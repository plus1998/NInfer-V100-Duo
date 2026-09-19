// Drift guard for the committed vLLM YaRN rope-parity reference
// (tests/core/data/yarn_ref_4x.json). Host-only, no GPU, no artifact.
//
// This test is a thin trampoline around tools/tp2/dump_yarn_ref.py -- the single script that both
// produced the committed JSON and is re-run here to regenerate it, so the reference and its guard
// can never drift apart. All numeric computation stays in that script (real vLLM rope classes);
// this test's own job is only to (a) locate the serve env's python, (b) re-run the script there,
// (c) parse both the fresh dump and the committed JSON with nlohmann::json (already vendored
// under third_party/, used the same way by tests/test_ninfer_bench_support.cpp and the serve
// schema tests), and (d) diff every recorded field:
//
//   * any floating-point scalar or array (yarn_inv_freq/native_inv_freq's 32 pairs each,
//     mscale_effective, attention_factor_A, ...): <=1e-9 relative
//   * everything else (the config dict, correction_range_production, beta_fast/slow, head_dim,
//     rotary_dim, rotary_pairs, class_used_by_get_rope, the mrope/x4-quirk flags, ...): exact
//   * vllm_version: exact, and on mismatch this test fails with a message naming BOTH the
//     committed and regenerated version strings explicitly -- a version delta is the only
//     legitimate cause of numeric drift here and must never be silently accepted.
//
// If the serve env's python (NINFER_VLLM_PYTHON, defaulting to the known
// /home/pc/Projects/vllm/unsloth-nvfp4-env/bin/python path) does not exist, this test skips
// (ctest code 77) without invoking anything. If that python exists but vLLM (or a dependency,
// e.g. its torch build) is not importable there, dump_yarn_ref.py detects the ImportError at
// module load, prints "SKIP: ..." and exits 77 itself -- this test forwards that exit code
// verbatim, so "vLLM unavailable" always skips cleanly, never fails.

#include <nlohmann/json.hpp>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

std::string vllm_python_path() {
    if (const char* override_path = std::getenv("NINFER_VLLM_PYTHON");
        override_path != nullptr && *override_path != '\0') {
        return override_path;
    }
    return "/home/pc/Projects/vllm/unsloth-nvfp4-env/bin/python";
}

// Test-only escape hatch (mirrors the NINFER_QWEN3_6_27B_WEIGHTS-style overrides used by the
// real-artifact tests): lets a negative-path check point this test at a scratch copy of the
// reference JSON instead of the committed one, without touching the committed file. Defaults to
// the real committed reference.
std::filesystem::path reference_json_path(const std::filesystem::path& source_dir) {
    if (const char* override_path = std::getenv("NINFER_YARN_REF_JSON");
        override_path != nullptr && *override_path != '\0') {
        return override_path;
    }
    return source_dir / "tests" / "core" / "data" / "yarn_ref_4x.json";
}

// Single-quote a path for use inside a `/bin/sh -c` command line. Good enough for the fixed,
// space-free, quote-free paths this test ever builds (repo checkout path, temp directory).
std::string shell_quote(const std::string& value) { return "'" + value + "'"; }

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

constexpr double kRelTol = 1e-9;

bool nearly_equal(double a, double b, double tol) {
    const double denom = std::abs(a) > 0.0 ? std::abs(a) : 1.0;
    return std::abs(a - b) / denom <= tol;
}

// std::to_string truncates to 6 decimal places, which prints as "0.000000" for both kRelTol
// (1e-9) and the small-magnitude inv_freq entries (down to ~4e-8) -- useless in a failure
// message. Format with full round-trippable precision instead.
std::string format_double(double value) {
    std::ostringstream stream;
    stream.precision(17);
    stream << value;
    return stream.str();
}

bool is_numeric_array(const Json& value) {
    if (!value.is_array()) { return false; }
    for (const auto& element : value) {
        if (!element.is_number()) { return false; }
    }
    return true;
}

// Exhaustively diffs one committed field against its regenerated counterpart, appending a
// human-readable message to `failures` for every mismatch. Floating scalars/arrays get a
// relative tolerance; everything else (strings, bools, nested objects/arrays of non-numbers)
// must match exactly -- this generically covers every field in yarn_ref_4x.json today and any
// field added to it later, without needing a hand-maintained key list.
void compare_field(const std::string& key, const Json& committed, const Json& regenerated,
                    std::vector<std::string>& failures) {
    if (committed.is_number() && regenerated.is_number()) {
        const double c = committed.get<double>();
        const double r = regenerated.get<double>();
        if (!nearly_equal(c, r, kRelTol)) {
            failures.push_back(key + ": mismatch beyond " + format_double(kRelTol) +
                                " relative (committed=" + format_double(c) +
                                " regenerated=" + format_double(r) + ")");
        }
        return;
    }
    if (is_numeric_array(committed) && is_numeric_array(regenerated)) {
        if (committed.size() != regenerated.size()) {
            failures.push_back(key + ": length mismatch (committed=" +
                                std::to_string(committed.size()) + " regenerated=" +
                                std::to_string(regenerated.size()) + ")");
            return;
        }
        for (std::size_t i = 0; i < committed.size(); ++i) {
            const double c = committed[i].get<double>();
            const double r = regenerated[i].get<double>();
            if (!nearly_equal(c, r, kRelTol)) {
                failures.push_back(key + "[" + std::to_string(i) + "]: mismatch beyond " +
                                    format_double(kRelTol) + " relative (committed=" +
                                    format_double(c) + " regenerated=" + format_double(r) + ")");
            }
        }
        return;
    }
    if (committed != regenerated) {
        failures.push_back(key + ": mismatch (committed=" + committed.dump() +
                            " regenerated=" + regenerated.dump() + ")");
    }
}

}  // namespace

int main() {
    const std::string python = vllm_python_path();
    if (!std::filesystem::exists(python)) {
        std::cerr << "skip: vLLM serve-env python not found at " << python
                   << " (override with NINFER_VLLM_PYTHON)\n";
        return 77;
    }

    const std::filesystem::path source_dir(NINFER_SOURCE_DIR);
    const std::filesystem::path dump_script    = source_dir / "tools" / "tp2" / "dump_yarn_ref.py";
    const std::filesystem::path reference_json = reference_json_path(source_dir);
    if (!std::filesystem::is_regular_file(dump_script)) {
        std::cerr << "dump script missing: " << dump_script << '\n';
        return 1;
    }
    if (!std::filesystem::is_regular_file(reference_json)) {
        std::cerr << "committed reference JSON missing: " << reference_json << '\n';
        return 1;
    }

    const std::filesystem::path stdout_path =
        std::filesystem::temp_directory_path() /
        ("ninfer_yarn_ref_drift_stdout_" + std::to_string(::getpid()) + ".json");
    const std::filesystem::path stderr_path =
        std::filesystem::temp_directory_path() /
        ("ninfer_yarn_ref_drift_stderr_" + std::to_string(::getpid()) + ".txt");

    const std::string command = "VLLM_LOGGING_LEVEL=WARNING " + shell_quote(python) + " " +
                                 shell_quote(dump_script.string()) + " >" +
                                 shell_quote(stdout_path.string()) + " 2>" +
                                 shell_quote(stderr_path.string());

    std::cout << "running: " << command << '\n';
    const int status = std::system(command.c_str());

    auto cleanup = [&]() {
        std::error_code ignored;
        std::filesystem::remove(stdout_path, ignored);
        std::filesystem::remove(stderr_path, ignored);
    };

    if (status == -1) {
        std::cerr << "failed to launch dump_yarn_ref.py\n";
        cleanup();
        return 1;
    }
    if (!WIFEXITED(status)) {
        std::cerr << "dump_yarn_ref.py terminated abnormally (raw status=" << status << ")\n";
        std::cerr << read_file(stderr_path);
        cleanup();
        return 1;
    }
    const int exit_code = WEXITSTATUS(status);
    if (exit_code == 77) {
        // dump_yarn_ref.py itself detected that vLLM (or a dependency) is not importable in
        // this python and printed its own "SKIP: ..." message -- surface it and forward the skip.
        std::cerr << read_file(stderr_path);
        cleanup();
        return 77;
    }
    if (exit_code != 0) {
        std::cerr << "dump_yarn_ref.py failed with exit code " << exit_code << ":\n"
                   << read_file(stderr_path);
        cleanup();
        return 1;
    }

    Json regenerated;
    Json committed;
    try {
        std::ifstream regenerated_stream(stdout_path);
        regenerated_stream >> regenerated;
        std::ifstream committed_stream(reference_json);
        committed_stream >> committed;
    } catch (const std::exception& error) {
        std::cerr << "failed to parse JSON: " << error.what() << '\n';
        cleanup();
        return 1;
    }
    cleanup();

    std::vector<std::string> failures;

    // Named-both-versions requirement: handled first and explicitly, independent of the
    // generic field diff below (which would also catch it, but not with this exact wording).
    const std::string committed_version   = committed.value("vllm_version", std::string{});
    const std::string regenerated_version = regenerated.value("vllm_version", std::string{});
    if (committed_version != regenerated_version) {
        failures.push_back(
            "vllm_version mismatch: committed=" + committed_version +
            " regenerated=" + regenerated_version +
            " -- the committed tests/core/data/yarn_ref_4x.json was generated against a "
            "different vLLM version than is installed now; a version delta is the only "
            "legitimate cause of numeric drift here and must be reviewed and regenerated "
            "deliberately, not silently accepted");
    }

    // Documentation-only fields, not values: excluded from the comparison.
    static const std::vector<std::string> kExcludedKeys = {"_comment", "generator_invocation",
                                                             "vllm_version"};
    for (const auto& [key, committed_value] : committed.items()) {
        if (std::find(kExcludedKeys.begin(), kExcludedKeys.end(), key) != kExcludedKeys.end()) {
            continue;
        }
        if (!regenerated.contains(key)) {
            failures.push_back(key + ": present in committed JSON but missing from regenerated dump");
            continue;
        }
        compare_field(key, committed_value, regenerated.at(key), failures);
    }

    if (!failures.empty()) {
        std::cerr << "FAIL: regenerated vLLM YaRN rope reference drifted from " << reference_json
                  << ":\n";
        for (const auto& failure : failures) { std::cerr << "  - " << failure << '\n'; }
        return 1;
    }

    std::cout << "ok: regenerated vLLM YaRN rope reference matches " << reference_json
              << " (vllm_version=" << regenerated_version << ")\n";
    return 0;
}
