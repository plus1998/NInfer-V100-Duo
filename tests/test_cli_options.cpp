// The CLI stop-policy and diagnostic option surface that is not the rope surface.
//
// Host-only: no artifact, no device, no engine construction. The subject is
// `ninfer::cli::parse_options` plus the one public default it has to agree with.
//
// `--ignore-eos` exists because a fixed-length decode to the context ceiling is impossible
// otherwise: the checkpoint's own `eos_token_id` set ends generation after ~1,900 tokens, it comes
// from `generation_config.json` inside the artifact, and nothing else on the CLI or the serve
// surface can drop it. Getting its default wrong is not a visible error -- it is a soak that ends
// early, or a product request that never terminates -- so the default is asserted here as
// explicitly as the flag itself.
//
// SCOPE NOTE, deliberate: the wiring `request.stop.include_model_defaults = !cli.ignore_eos` lives
// in `apps/cli/main.cpp`'s `main()`, which cannot be linked into a test without also constructing
// an Engine and loading an artifact. So this file tests the parsed option, and pins the other half
// of that expression -- that `StopPolicy::include_model_defaults` defaults to `true`, i.e. that
// `!ignore_eos` reproduces the pre-existing behavior when the flag is absent. If the assignment in
// main.cpp is ever deleted, these checks still pass; what they prevent is the option surface
// drifting under it.

#include "options.h" // apps/cli

#include "ninfer/types.h"

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

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

// The minimum legal invocation: an artifact path and exactly one of --prompt/--messages.
std::vector<std::string> base() { return {"ninfer", "m.ninfer", "--prompt", "hi"}; }

std::vector<std::string> with(std::vector<std::string> extra) {
    std::vector<std::string> arguments = base();
    for (std::string& item : extra) { arguments.push_back(std::move(item)); }
    return arguments;
}

void check_ignore_eos() {
    const ninfer::cli::Options omitted = parse_cli(base());
    check(!omitted.ignore_eos, "--ignore-eos is not off by default");

    const ninfer::cli::Options set = parse_cli(with({"--ignore-eos"}));
    check(set.ignore_eos, "--ignore-eos did not set ignore_eos");

    // The public default the CLI's `!ignore_eos` has to reproduce when the flag is absent: with no
    // flag, the request must keep the checkpoint's own stop tokens.
    const ninfer::StopPolicy policy_default;
    check(policy_default.include_model_defaults,
          "StopPolicy::include_model_defaults no longer defaults to true, so `!ignore_eos` no "
          "longer reproduces the default stop behavior");
    check(policy_default.include_model_defaults == !omitted.ignore_eos,
          "the CLI default and the StopPolicy default disagree about model stop tokens");

    // It is a valueless flag: it must not swallow the following argument.
    const ninfer::cli::Options followed = parse_cli(with({"--ignore-eos", "--max-new", "64"}));
    check(followed.ignore_eos && followed.max_new == 64U,
          "--ignore-eos consumed the option that followed it");

    // Position independence, and no interference with the flags the 1M soak runs it beside.
    const ninfer::cli::Options soak = parse_cli(
        with({"--greedy", "--ignore-eos", "--no-thinking", "--print-token-ids", "--max-new", "16"}));
    check(soak.ignore_eos, "--ignore-eos was lost among the other soak flags");
    check(soak.greedy && soak.sampling.temperature.has_value() &&
              soak.sampling.temperature.value() == 0.0F,
          "--greedy did not survive alongside --ignore-eos");
    check(!soak.enable_thinking, "--no-thinking did not survive alongside --ignore-eos");
    check(soak.print_token_ids, "--print-token-ids did not survive alongside --ignore-eos");
    check(soak.max_new == 16U, "--max-new did not survive alongside --ignore-eos");

    // Explicit stop conditions are a separate mechanism and must remain usable with it: the flag
    // drops the CHECKPOINT's stop tokens, not the caller's.
    const ninfer::cli::Options with_stops =
        parse_cli(with({"--ignore-eos", "--stop-token-id", "7", "--stop", "END"}));
    check(with_stops.ignore_eos, "--ignore-eos was lost when explicit stops were present");
    check(with_stops.stop_token_ids.size() == 1 && with_stops.stop_token_ids.front() == 7,
          "--stop-token-id was lost alongside --ignore-eos");
    check(with_stops.stop_strings.size() == 1 && with_stops.stop_strings.front().text == "END",
          "--stop was lost alongside --ignore-eos");
}

// The option surface is a contract with the operator, not only with the parser: an option that
// parses but is undocumented is an option nobody finds.
void check_documented() {
    const std::string usage = ninfer::cli::usage_text("ninfer");
    check(usage.find("--ignore-eos") != std::string::npos,
          "--ignore-eos is missing from the CLI usage text");
}

} // namespace

int main() {
    check_ignore_eos();
    check_documented();
    if (failures != 0) {
        std::cerr << failures << " CLI option check(s) failed\n";
        return 1;
    }
    std::cout << "OK CLI stop-policy and diagnostic options\n";
    return 0;
}
