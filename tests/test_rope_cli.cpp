// The `--rope native|yarn` option surface and the effective context ceiling.
//
// Three things are checked here, all host-only (no CUDA device, no artifact):
//
//   1. CLI and ninfer-serve parse `--rope`, `--yarn-factor` and `--yarn-origin` identically and
//      default to the pre-YaRN behavior (`native`, factor 4.0, origin 262144 -- the latter two
//      inert unless yarn is selected).
//   2. `rope_effective_max_context` resolves the ceiling every context-derived extent in the
//      sequence plan is sized from: the variant's registered native capacity under Native,
//      `yarn_origin * yarn_factor` under Yarn. This is the value `--max-context` is admitted
//      against, so getting it wrong is either a silent cap at 262144 or an out-of-domain plan.
//   3. The complete accept/reject matrix that function owns. Each rejection exists because the
//      alternative is a wrong answer rather than an error: a mismatched origin would build a
//      correction table for the wrong window, `--vision` would rope 2-D image-grid positions
//      through a table that does not describe them, and an over-large factor would ask for
//      positions past the product's 1,048,576 ceiling.
//
// The domain values are the two registered rope domains: qwen3.6/3.8-27B (native 262144, YaRN
// available) and qwen3.6-35b-a3b (native 262144, no YaRN rope domain).

#include "options.h" // apps/cli
#include "serve/serve_options.h"
#include "targets/qwen3_6/impl/runtime/yarn_rope.h"

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::EngineOptions;
using ninfer::RopeMode;
using ninfer::SpeculativeBackend;
using ninfer::targets::qwen3_6::detail::RopeDomain;
using ninfer::targets::qwen3_6::detail::rope_effective_max_context;

constexpr RopeDomain kQwen27b{262144, true};
constexpr RopeDomain kQwen35bA3b{262144, false};

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

ninfer::cli::Options parse_cli(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

ninfer::serve::ServeOptions parse_serve(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

// Rejections are asserted by MESSAGE as well as by type. These messages are what an operator sees
// after a 20-minute weight load fails, so "it threw something" is not the contract -- "it named the
// flag and the number" is.
template <class Fn> bool rejected(Fn&& call, std::string_view expected_substring = {}) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        const std::string_view message(error.what());
        if (!expected_substring.empty() &&
            message.find(expected_substring) == std::string_view::npos) {
            std::cerr << "FAIL: rejection message \"" << message << "\" does not mention \""
                      << expected_substring << "\"\n";
            ++failures;
        }
        return true;
    }
    return false;
}

// One engine configuration, spelled out so each matrix row reads as its own statement.
EngineOptions options_for(RopeMode mode, std::uint32_t max_context, double factor,
                          std::uint32_t origin) {
    EngineOptions options;
    options.max_context = max_context;
    options.rope_mode   = mode;
    options.yarn_factor = factor;
    options.yarn_origin = origin;
    return options;
}

void check_cli_and_serve_parsing() {
    const ninfer::cli::Options cli_default = parse_cli({"ninfer", "m.ninfer", "--prompt", "hi"});
    check(cli_default.rope_mode == RopeMode::Native, "CLI default rope mode is not native");
    check(cli_default.yarn_factor == 4.0, "CLI default yarn factor is not 4.0");
    check(cli_default.yarn_origin == 262144U, "CLI default yarn origin is not 262144");

    const ninfer::cli::Options cli_yarn =
        parse_cli({"ninfer", "m.ninfer", "--prompt", "hi", "--rope", "yarn", "--yarn-factor", "2.0",
                   "--yarn-origin", "262144"});
    check(cli_yarn.rope_mode == RopeMode::Yarn, "CLI --rope yarn did not select yarn");
    check(cli_yarn.yarn_factor == 2.0, "CLI --yarn-factor did not parse");
    check(cli_yarn.yarn_origin == 262144U, "CLI --yarn-origin did not parse");

    check(rejected([] { (void)parse_cli({"ninfer", "m.ninfer", "--prompt", "hi", "--rope", "x"}); }),
          "CLI accepted an unknown --rope value");
    check(rejected([] {
              (void)parse_cli(
                  {"ninfer", "m.ninfer", "--prompt", "hi", "--yarn-factor", "not-a-number"});
          }),
          "CLI accepted a non-numeric --yarn-factor");
    check(rejected([] {
              (void)parse_cli({"ninfer", "m.ninfer", "--prompt", "hi", "--yarn-factor", "0.5"});
          }),
          "CLI accepted a --yarn-factor below 1.0");

    const ninfer::serve::ServeOptions serve_default = parse_serve({"ninfer-serve", "m.ninfer"});
    check(serve_default.rope_mode == RopeMode::Native, "serve default rope mode is not native");
    check(serve_default.yarn_factor == 4.0, "serve default yarn factor is not 4.0");
    check(serve_default.yarn_origin == 262144U, "serve default yarn origin is not 262144");

    const ninfer::serve::ServeOptions serve_yarn =
        parse_serve({"ninfer-serve", "m.ninfer", "--rope", "yarn", "--yarn-factor", "4.0",
                     "--yarn-origin", "262144", "--max-context", "1048576", "--kv-capacity",
                     "auto"});
    check(serve_yarn.rope_mode == RopeMode::Yarn, "serve --rope yarn did not select yarn");
    check(serve_yarn.yarn_factor == 4.0, "serve --yarn-factor did not parse");
    check(serve_yarn.yarn_origin == 262144U, "serve --yarn-origin did not parse");
    check(serve_yarn.max_context == 1048576U, "serve --max-context 1048576 did not parse");

    check(rejected([] { (void)parse_serve({"ninfer-serve", "m.ninfer", "--rope", "x"}); }),
          "serve accepted an unknown --rope value");
    check(rejected([] { (void)parse_serve({"ninfer-serve", "m.ninfer", "--yarn-factor", "0"}); }),
          "serve accepted a --yarn-factor below 1.0");
}

void check_effective_ceiling() {
    // Native: the ceiling is the variant's registered native capacity, whatever the (unread) yarn
    // fields say. A native run must be unaffected by them.
    check(rope_effective_max_context(options_for(RopeMode::Native, 2048, 4.0, 262144), kQwen27b) ==
              262144U,
          "native ceiling is not the variant native capacity");
    check(rope_effective_max_context(options_for(RopeMode::Native, 2048, 99.0, 7), kQwen27b) ==
              262144U,
          "native ceiling was influenced by yarn fields");
    check(rope_effective_max_context(options_for(RopeMode::Native, 262144, 4.0, 262144),
                                     kQwen35bA3b) == 262144U,
          "native ceiling differs on the 35B-A3B domain");

    // Yarn: origin x factor, exactly.
    check(rope_effective_max_context(options_for(RopeMode::Yarn, 1048576, 4.0, 262144), kQwen27b) ==
              1048576U,
          "yarn 4.0 x 262144 did not resolve to 1048576");
    check(rope_effective_max_context(options_for(RopeMode::Yarn, 524288, 2.0, 262144), kQwen27b) ==
              524288U,
          "yarn 2.0 x 262144 did not resolve to 524288");
    check(rope_effective_max_context(options_for(RopeMode::Yarn, 1, 1.0, 262144), kQwen27b) ==
              262144U,
          "yarn 1.0 x 262144 did not resolve to 262144");
    // A fractional factor is legal as long as the product is a whole token count.
    check(rope_effective_max_context(options_for(RopeMode::Yarn, 393216, 1.5, 262144), kQwen27b) ==
              393216U,
          "yarn 1.5 x 262144 did not resolve to 393216");
}

void check_accept_reject_matrix() {
    // Accepted: a native run at exactly the native ceiling, and a yarn run past it.
    check(!rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Native, 262144, 4.0, 262144),
                                               kQwen27b);
          }),
          "native max_context at the native ceiling was rejected");
    check(!rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Yarn, 300000, 4.0, 262144),
                                               kQwen27b);
          }),
          "yarn max_context past the native ceiling was rejected");
    check(!rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Yarn, 1048576, 4.0, 262144),
                                               kQwen27b);
          }),
          "yarn max_context at the 1M ceiling was rejected");

    // Rejected: native beyond the native ceiling (the pre-YaRN rule, unchanged).
    check(rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Native, 262145, 4.0, 262144),
                                               kQwen27b);
          }, "variant native context capacity 262144"),
          "native max_context beyond the native ceiling was accepted");
    check(rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Native, 0, 4.0, 262144),
                                               kQwen27b);
          }),
          "a zero max_context was accepted");

    // Rejected: yarn beyond its own extended ceiling.
    check(rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Yarn, 524289, 2.0, 262144),
                                               kQwen27b);
          }, "YaRN-extended context capacity 524288"),
          "yarn max_context beyond origin x factor was accepted");

    // Rejected: an origin that is not the artifact's registered native capacity.
    check(rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Yarn, 4096, 4.0, 131072),
                                               kQwen27b);
          }, "--yarn-origin must equal"),
          "a yarn origin below the registered native capacity was accepted");
    check(rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Yarn, 4096, 4.0, 524288),
                                               kQwen27b);
          }),
          "a yarn origin above the registered native capacity was accepted");

    // Rejected: a factor whose product overshoots the 1,048,576 product ceiling.
    check(rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Yarn, 4096, 8.0, 262144),
                                               kQwen27b);
          }, "maximum extended context"),
          "yarn 8.0 x 262144 (2,097,152) was accepted");
    check(rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Yarn, 4096, 0.5, 262144),
                                               kQwen27b);
          }, "--yarn-factor must be a finite value"),
          "a yarn factor below 1.0 was accepted");

    // Rejected: a factor whose product is not a whole token count. 1.1 x 262144 = 288,358.4, which
    // is inside the 1,048,576 ceiling and would otherwise be silently truncated to a ceiling the
    // operator did not ask for.
    check(rejected(
              [] {
                  (void)rope_effective_max_context(options_for(RopeMode::Yarn, 4096, 1.1, 262144),
                                                   kQwen27b);
              },
              "whole number of tokens"),
          "a yarn factor whose product is not a whole token count was accepted");
    // ... while a fractional factor that DOES land on a whole count is accepted (1.5 -> 393216).
    check(!rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Yarn, 393216, 1.5, 262144),
                                               kQwen27b);
          }),
          "yarn 1.5 x 262144 (a whole token count) was rejected");

    // Rejected: a target/profile with no YaRN rope domain. Native on the same domain is fine.
    check(rejected([] {
              (void)rope_effective_max_context(options_for(RopeMode::Yarn, 4096, 4.0, 262144),
                                               kQwen35bA3b);
          }, "--rope yarn is not available"),
          "yarn was accepted on a variant with no YaRN rope domain");

    // Rejected: yarn with Vision, at any tp width.
    check(rejected([] {
              EngineOptions options = options_for(RopeMode::Yarn, 4096, 4.0, 262144);
              options.enable_vision = true;
              (void)rope_effective_max_context(options, kQwen27b);
          }, "cannot be combined with --vision"),
          "yarn + vision was accepted");

    // Rejected: yarn with the DFlash speculative backend; accepted with MTP.
    check(rejected([] {
              EngineOptions options              = options_for(RopeMode::Yarn, 4096, 4.0, 262144);
              options.speculative.backend        = SpeculativeBackend::DFlash;
              options.speculative.draft_tokens   = 4;
              (void)rope_effective_max_context(options, kQwen27b);
          }, "cannot be combined with --spec dflash"),
          "yarn + dflash was accepted");
    check(!rejected([] {
              EngineOptions options            = options_for(RopeMode::Yarn, 4096, 4.0, 262144);
              options.speculative.backend      = SpeculativeBackend::Mtp;
              options.speculative.draft_tokens = 3;
              (void)rope_effective_max_context(options, kQwen27b);
          }),
          "yarn + mtp was rejected");

    // Explicitly NOT coupled to tensor parallelism: yarn is legal at tp 1 and tp 2 alike, and the
    // native path is likewise unaffected by tp. Both widths are asserted so a future guard that
    // ties yarn to --tp 2 fails here rather than in a 1M boot.
    for (const int tp : {1, 2}) {
        check(!rejected([tp] {
                  EngineOptions options = options_for(RopeMode::Yarn, 300000, 4.0, 262144);
                  options.tp            = tp;
                  options.devices       = tp == 2 ? std::vector<int>{0, 1} : std::vector<int>{0};
                  (void)rope_effective_max_context(options, kQwen27b);
              }),
              "yarn was rejected at tp " + std::to_string(tp));
    }
}

} // namespace

int main() {
    check_cli_and_serve_parsing();
    check_effective_ceiling();
    check_accept_reject_matrix();
    if (failures != 0) {
        std::cerr << failures << " rope option check(s) failed\n";
        return 1;
    }
    std::cout << "OK rope CLI/serve options and effective context ceiling\n";
    return 0;
}
