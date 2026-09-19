#include "targets/qwen3_6/impl/runtime/yarn_rope.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail {

namespace {

constexpr double kPi                       = 3.14159265358979323846;
constexpr double kMaxEffectiveContext       = 1'048'576.0;
constexpr int kRegisteredNativeMax          = 262144;
// vLLM's MRotaryEmbedding.__init__ video-cache headroom quirk (see yarn_rope.h): the
// correction-range formula is evaluated against original_max * 4, not original_max directly.
constexpr double kEffectiveMaxPositionRatio = 4.0;

double yarn_find_correction_dim(double num_rotations, int rotary_dim, double theta,
                                 double effective_max_position) {
    return (static_cast<double>(rotary_dim) *
            std::log(effective_max_position / (num_rotations * 2.0 * kPi))) /
           (2.0 * std::log(theta));
}

void yarn_find_correction_range(double beta_fast, double beta_slow, int rotary_dim, double theta,
                                 double effective_max_position, double* low, double* high) {
    const double low_raw  = yarn_find_correction_dim(beta_fast, rotary_dim, theta,
                                                       effective_max_position);
    const double high_raw = yarn_find_correction_dim(beta_slow, rotary_dim, theta,
                                                       effective_max_position);
    const double low_t    = std::floor(low_raw);
    const double high_t   = std::ceil(high_raw);
    *low  = std::max(low_t, 0.0);
    *high = std::min(high_t, static_cast<double>(rotary_dim - 1));
}

double yarn_get_mscale(double factor) {
    if (factor <= 1.0) {
        return 1.0;
    }
    return 0.1 * std::log(factor) + 1.0;
}

}  // namespace

float yarn_rope_mscale(const YarnParams& p) {
    return static_cast<float>(yarn_get_mscale(static_cast<double>(p.factor)));
}

std::pair<std::vector<float>, float> yarn_scale(const YarnParams& p) {
    if (p.original_max != kRegisteredNativeMax) {
        throw std::invalid_argument(
            "yarn_scale: original_max must equal the artifact's registered native max (262144)");
    }
    const double factor = static_cast<double>(p.factor);
    const double origin = static_cast<double>(p.original_max);
    if (factor * origin > kMaxEffectiveContext + 1e-6) {
        throw std::invalid_argument(
            "yarn_scale: factor * original_max exceeds the 1,048,576 maximum effective context");
    }

    const int rotary_dim = p.rotary_pairs * 2;
    const double theta   = static_cast<double>(p.theta);

    // See yarn_rope.h: correction range is computed against original_max * 4 (vLLM
    // MRotaryEmbedding's unconditional video-cache position enlargement), not original_max.
    const double effective_max_position = origin * kEffectiveMaxPositionRatio;

    double low  = 0.0;
    double high = 0.0;
    yarn_find_correction_range(static_cast<double>(p.beta_fast), static_cast<double>(p.beta_slow),
                                rotary_dim, theta, effective_max_position, &low, &high);
    if (low == high) {
        high += 0.001;  // vLLM's singularity guard (yarn_linear_ramp_mask)
    }

    const double mscale                = yarn_get_mscale(factor);  // attn_factor pinned at 1
    const float attention_factor_a     = static_cast<float>(1.0 / (mscale * mscale));

    std::vector<float> inv_freq(static_cast<std::size_t>(p.rotary_pairs));
    for (int i = 0; i < p.rotary_pairs; ++i) {
        const double exponent      = static_cast<double>(2 * i) / static_cast<double>(rotary_dim);
        const double pos_freq      = std::pow(theta, exponent);
        const double extrapolation = 1.0 / pos_freq;
        const double interpolation = 1.0 / (factor * pos_freq);

        double ramp = (static_cast<double>(i) - low) / (high - low);
        ramp        = std::min(1.0, std::max(0.0, ramp));
        const double mask = 1.0 - ramp;

        const double value = interpolation * (1.0 - mask) + extrapolation * mask;
        inv_freq[static_cast<std::size_t>(i)] = static_cast<float>(value);
    }

    return {std::move(inv_freq), attention_factor_a};
}

std::uint32_t rope_effective_max_context(const EngineOptions& options, const RopeDomain& domain) {
    if (domain.native_max == 0) {
        throw std::invalid_argument("rope domain has no native context capacity");
    }
    std::uint32_t ceiling = domain.native_max;
    if (options.rope_mode == RopeMode::Yarn) {
        if (!domain.supports_yarn) {
            throw std::invalid_argument(
                "--rope yarn is not available for this target/profile: its text attention does not "
                "use the 256-dim head / 64-dim partial-rotary geometry the YaRN frequency table is "
                "computed for, and no YaRN rope_parameters are published for this checkpoint; use "
                "--rope native");
        }
        if (options.enable_vision) {
            throw std::invalid_argument(
                "--rope yarn cannot be combined with --vision: the Vision encoder ropes 2-D "
                "image-grid positions through its own 18-entry table, which the 32-entry text YaRN "
                "table does not describe; use --rope native for multimodal runs");
        }
        if (options.speculative.backend == SpeculativeBackend::DFlash) {
            throw std::invalid_argument(
                "--rope yarn cannot be combined with --spec dflash: the DFlash attention path "
                "carries its own 64-entry rotary table; use --spec mtp or --spec none");
        }
        if (options.yarn_origin != domain.native_max) {
            throw std::invalid_argument(
                "--yarn-origin must equal the artifact's registered native context capacity (" +
                std::to_string(domain.native_max) + "), got " +
                std::to_string(options.yarn_origin));
        }
        if (!std::isfinite(options.yarn_factor) || options.yarn_factor < 1.0) {
            throw std::invalid_argument("--yarn-factor must be a finite value >= 1.0");
        }
        const double extended =
            static_cast<double>(options.yarn_origin) * options.yarn_factor;
        if (extended > static_cast<double>(kMaximumYarnContext) + 1e-6) {
            throw std::invalid_argument(
                "--yarn-origin x --yarn-factor exceeds the " +
                std::to_string(kMaximumYarnContext) + "-token maximum extended context");
        }
        const auto extended_u32 = static_cast<std::uint32_t>(extended);
        if (static_cast<double>(extended_u32) != extended) {
            throw std::invalid_argument(
                "--yarn-origin x --yarn-factor must land on a whole number of tokens");
        }
        ceiling = extended_u32;
    }
    if (options.max_context == 0 || options.max_context > ceiling) {
        throw std::invalid_argument(
            "max_context " + std::to_string(options.max_context) + " exceeds the " +
            (options.rope_mode == RopeMode::Yarn ? std::string("YaRN-extended")
                                                 : std::string("variant native")) +
            " context capacity " + std::to_string(ceiling));
    }
    return ceiling;
}

}  // namespace ninfer::targets::qwen3_6::detail
