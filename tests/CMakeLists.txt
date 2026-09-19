find_package(Python3 REQUIRED COMPONENTS Interpreter)

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the installed product headers stand alone.
add_executable(ninfer_public_api_test test_public_api.cpp)
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

function(ninfer_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc
    ${CUDAToolkit_INCLUDE_DIRS})
  if(arg_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  else()
    target_link_libraries(${name} PRIVATE ninfer_core)
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

function(ninfer_add_op_test name)
  ninfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ninfer_add_linear_test name source)
  ninfer_add_op_test(${name}
    SOURCES ${source}
            ops/linear/linear_test_common.cpp
    LIBRARIES ninfer_ops)
endfunction()

ninfer_add_op_test(ninfer_ggml_k_test
  SOURCES ops/test_ggml_k.cpp
  LIBRARIES ninfer_ops)

function(ninfer_add_fused_linear_test name source common)
  ninfer_add_op_test(${name}
    SOURCES ${source} ${common}
    LIBRARIES ninfer_ops)
endfunction()

# Artifact framing and model-independent runtime contracts.
ninfer_add_test(ninfer_artifact_reader_test
  SOURCES test_ninfer_artifact_reader.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_artifact_materialization_test
  SOURCES test_artifact_materialization.cpp
  LIBRARIES ninfer_artifact)
# Host-only: no artifact, no device. Byte-level TP2 shard slice geometry per storage layout.
ninfer_add_test(ninfer_tensor_slice_test
  SOURCES test_tensor_slice.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_media_decode_test
  SOURCES test_media_decode.cpp
  LIBRARIES ninfer_media_decode)
ninfer_add_test(ninfer_device_test       SOURCES test_device.cpp)
ninfer_add_test(ninfer_decode_graph_test SOURCES test_decode_graph.cpp)
ninfer_add_test(ninfer_tensor_test       SOURCES test_tensor.cpp)
ninfer_add_test(ninfer_arena_test        SOURCES test_arena.cpp)
ninfer_add_test(ninfer_request_memory_test SOURCES test_request_memory.cpp)
ninfer_add_test(ninfer_admission_policy_test SOURCES test_admission_policy.cpp)
ninfer_add_test(ninfer_kv_capacity_test SOURCES test_kv_capacity.cpp)
# Host-only: no artifact, no device, no Engine construction. tp2 per-device KV pool sizing (4.1).
ninfer_add_test(ninfer_kv_capacity_tp2_test SOURCES test_kv_capacity_tp2.cpp)
ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES test_sampling_defaults.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_sampling_defaults_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
ninfer_add_test(ninfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_state_store_test
  SOURCES test_state_store.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES test_gdn_replay_records.cpp
  LIBRARIES ninfer_core)
set_tests_properties(
  ninfer_artifact_materialization_test
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_request_memory_test
  ninfer_kv_cache_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)
# Family frontend and one opt-in real-artifact prefix/MTP integration test.
ninfer_add_test(ninfer_qwen3_6_frontend_test
  SOURCES targets/qwen3_6/test_frontend.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_frontend_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
ninfer_add_test(ninfer_qwen3_6_runtime_mechanisms_test
  SOURCES targets/qwen3_6/test_runtime_mechanisms.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_runtime_mechanisms_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
# Host-only: no artifact, no device. YaRN host-side frequency/attention-factor math, family-owned
# because src/core owns device primitives only and the rope semantics belong to
# src/targets/qwen3_6; checked against a committed vLLM-reference dump
# (tests/core/data/yarn_ref_4x.json).
ninfer_add_test(ninfer_qwen3_6_yarn_rope_test
  SOURCES targets/qwen3_6/test_yarn_rope.cpp
  LIBRARIES ninfer_engine ninfer_core)
# Drift guard for the committed vLLM reference dump above
# (tests/core/data/yarn_ref_4x.json). Host-only; re-runs tools/tp2/dump_yarn_ref.py -- the same
# single script that produced the dump -- in the serve env's python and diffs every recorded
# field against the committed JSON, so vLLM-side yarn formula/version drift is caught instead
# of silently accepted. Skips (77) when that python or vLLM is unavailable.
ninfer_add_test(ninfer_qwen3_6_yarn_rope_drift_test
  SOURCES targets/qwen3_6/test_yarn_rope_drift.cpp
  NEEDS_SOURCE_DIR)
set_tests_properties(ninfer_qwen3_6_yarn_rope_drift_test PROPERTIES SKIP_RETURN_CODE 77)
# Engine worker-loop exception boundary, driven through the real ConcurrentExecutor with a
# scripted in-process Program: no artifact and no model, but the worker binds itself to a CUDA
# device, so this skips (77) where none is visible.
ninfer_add_test(ninfer_concurrent_executor_test
  SOURCES test_concurrent_executor.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_concurrent_executor_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
set_tests_properties(ninfer_concurrent_executor_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_prefix_real_test
  SOURCES targets/qwen3_6_27b/test_engine_prefix_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_8_27b_tp2_real_test
  SOURCES targets/qwen3_6_27b/test_engine_tp2_real.cpp
  LIBRARIES ninfer_engine)
target_link_libraries(ninfer_qwen3_8_27b_tp2_real_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_8_27b_tp2_real_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
# MTP3 speculative decoding under TP2. Opt-in on the same env/device gate as its siblings; the
# timeout is larger because every leg runs four engines (tp1/tp2 x MTP/off).
ninfer_add_test(ninfer_qwen3_8_27b_mtp_tp2_real_test
  SOURCES targets/qwen3_6_27b/test_engine_mtp_tp2_real.cpp
  LIBRARIES ninfer_engine ninfer_product_prompt_input)
target_link_libraries(ninfer_qwen3_8_27b_mtp_tp2_real_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_8_27b_mtp_tp2_real_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600)
# Full-head GGUF Q4_K_M MTP3 and graph/eager state gate. Every engine uses TP2,
# so it can run on two 16 GiB V100s without loading the whole artifact on one card.
ninfer_add_test(ninfer_qwen3_8_27b_v100-duo_real_test
  SOURCES targets/qwen3_6_27b/test_engine_v100-duo_real.cpp
  LIBRARIES ninfer_engine)
target_link_libraries(ninfer_qwen3_8_27b_v100-duo_real_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_8_27b_v100-duo_real_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
# TP2 INT8 KV + optimized MTP3 cache continuation and response-checkpoint replay.
ninfer_add_test(ninfer_qwen3_8_27b_v100-duo_prefix_real_test
  SOURCES targets/qwen3_6_27b/test_engine_v100-duo_prefix_real.cpp
  LIBRARIES ninfer_engine)
target_link_libraries(ninfer_qwen3_8_27b_v100-duo_prefix_real_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_8_27b_v100-duo_prefix_real_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
# `--rope yarn` past the native 262144-position window, on real weights, at both tp widths.
# Artifact-gated; the tp2 leg additionally needs two devices.
ninfer_add_test(ninfer_qwen3_8_27b_yarn_real_test
  SOURCES targets/qwen3_6_27b/test_engine_yarn_real.cpp
  LIBRARIES ninfer_engine)
target_link_libraries(ninfer_qwen3_8_27b_yarn_real_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_8_27b_yarn_real_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 5400)
ninfer_add_test(ninfer_qwen3_8_27b_graph_tp2_test
  SOURCES targets/qwen3_6_27b/test_graph_tp2.cpp
  LIBRARIES ninfer_engine)
target_link_libraries(ninfer_qwen3_8_27b_graph_tp2_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_8_27b_graph_tp2_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
# Greedy tp1-vs-tp2 numerical parity harness. Source lives under tools/tp2/ because it is a
# re-runnable host tool, not only a test; wired here as an opt-in ctest as well, on the same
# env/device gate as its siblings above.
ninfer_add_test(ninfer_qwen3_8_27b_tp2_parity_test
  SOURCES ${PROJECT_SOURCE_DIR}/tools/tp2/parity.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_product_prompt_input)
target_link_libraries(ninfer_qwen3_8_27b_tp2_parity_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_8_27b_tp2_parity_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 5400)
ninfer_add_test(ninfer_qwen3_6_27b_load_plan_test
  SOURCES targets/qwen3_6_27b/test_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_27b_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ninfer_qwen3_6_27b_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_27b_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_sharded_materialization_test
  SOURCES targets/qwen3_6_27b/test_sharded_materialization.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_27b_sharded_materialization_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/impl
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(ninfer_qwen3_6_27b_sharded_materialization_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_27b_sharded_materialization_test PROPERTIES SKIP_RETURN_CODE 77)
# Host-only: no artifact, no device. Pure TP2 ShardPlan table test.
ninfer_add_test(ninfer_qwen3_6_27b_shard_map_test
  SOURCES targets/test_shard_map.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_27b_shard_map_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/impl
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_dflash_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  SOURCES targets/qwen3_6_35b_a3b/test_dflash_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
target_link_libraries(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Product protocol contracts.
ninfer_add_test(ninfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_responses_schema_test
  SOURCES test_responses_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_response_store_test
  SOURCES test_response_store.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES test_tool_call_parser.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_serve_options_test
  SOURCES test_serve_options.cpp
  LIBRARIES ninfer_serve)
# Host-only: no artifact, no device. `--rope native|yarn` parsing on BOTH product front ends plus
# the effective-context-ceiling resolution and its accept/reject matrix. The CLI parser is
# compiled in directly because apps/ninfer is an executable, not a library.
ninfer_add_test(ninfer_rope_cli_test
  SOURCES test_rope_cli.cpp ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
  LIBRARIES ninfer_serve ninfer_engine)
target_include_directories(ninfer_rope_cli_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)
# Host-only: no artifact, no device. The CLI stop-policy and diagnostic option surface that the rope
# test does not cover -- principally `--ignore-eos`, whose default decides whether a request honors
# the checkpoint's own stop tokens. Same compile-in as above: apps/ninfer is an executable, not a
# library.
ninfer_add_test(ninfer_cli_options_test
  SOURCES test_cli_options.cpp ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)
ninfer_add_test(ninfer_request_log_test
  SOURCES test_request_log.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_http_error_handler_test
  SOURCES test_http_error_handler.cpp
  LIBRARIES ninfer_serve)
target_include_directories(ninfer_http_error_handler_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
ninfer_add_test(ninfer_bench_support_test
  SOURCES test_ninfer_bench_support.cpp
          ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b/ninfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b)

# Shared Op numerical tests. Target-composed schedule tests are registered separately below.
set(ninfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  gated_rmsnorm
  l2norm
  gated_delta_net
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  vision_attention
  sampling
  scalar
  cast
  scatter
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES ops/test_${op}.cpp
    LIBRARIES ninfer_ops)
endforeach()

# RoPE's YaRN frequency-override mode. Listed separately from the loop above because it also
# consumes the family-owned host YaRN math (yarn_scale/yarn_rope_mscale, built into
# ninfer_engine) as the source of both the native and the corrected frequency table, instead of
# transcribing either into the test.
ninfer_add_op_test(ninfer_rope_yarn_test
  SOURCES ops/test_rope_yarn.cpp
  LIBRARIES ninfer_ops ninfer_engine)

# Cross-device collectives. The suite needs two CUDA devices in one process and reports the
# shared skip code when fewer are visible.
ninfer_add_op_test(ninfer_allreduce_test
  SOURCES ops/test_allreduce.cpp
  LIBRARIES ninfer_ops)

# Tensor-parallel split forms. Same two-device requirement and skip code as the collectives above.
ninfer_add_op_test(ninfer_linear_split_test
  SOURCES ops/test_linear_split.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_linear_add_split_test
  SOURCES ops/test_linear_add_split.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_linear_swiglu_split_test
  SOURCES ops/test_linear_swiglu_split.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_attn_input_proj_split_test
  SOURCES ops/test_attn_input_proj_split.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_projections_split_test
  SOURCES ops/test_gdn_projections_split.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_output_head_split_test
  SOURCES ops/test_output_head_split.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_attention_headlocal_test
  SOURCES ops/test_attention_headlocal.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_headsplit_test
  SOURCES ops/test_gdn_headsplit.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_split_test
  SOURCES ops/test_mtp_split.cpp
  LIBRARIES ninfer_ops)

# Stateful, fused, and exact-shape Ops remain central even when the current implementation domain
# is one checkpoint/device.
ninfer_add_op_test(ninfer_gqa_attention_test
  SOURCES ops/test_gqa_attention.cpp
  LIBRARIES ninfer_ops)
# The visible-key domain past the pre-YaRN 262144 bound: its own target because it needs multi-GiB
# KV pools and a long FP64 oracle, and skips per case when the device cannot hold one.
ninfer_add_op_test(ninfer_gqa_attention_long_context_test
  SOURCES ops/test_gqa_attention_long_context.cpp
  LIBRARIES ninfer_ops)
set_tests_properties(ninfer_gqa_attention_long_context_test PROPERTIES TIMEOUT 1800)
ninfer_add_op_test(ninfer_bidirectional_gqa_attention_test
  SOURCES ops/test_bidirectional_gqa_attention.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_swa_test
  SOURCES ops/test_swa.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_kv_cache_append_prefix_test
  SOURCES ops/test_kv_cache_append_prefix.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES ops/test_prepare_masked_block.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES ops/test_sparse_moe.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES ops/test_mtp_pack.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES ops/test_mtp_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES ops/test_speculative_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_linear_test(ninfer_linear_q4_a16_test ops/linear/test_q4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q5_a16_test ops/linear/test_q5_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q6_a16_test ops/linear/test_q6_a16.cpp)
ninfer_add_linear_test(ninfer_linear_w8_a16_test ops/linear/test_w8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a16_test ops/linear/test_nvfp4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a4_test ops/linear/test_nvfp4_a4.cpp)
ninfer_add_linear_test(ninfer_linear_fp8_a16_test ops/linear/test_fp8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_fp8_a8_test ops/linear/test_fp8_a8.cpp)
ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES ops/linear/test_bf16_a16.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES ops/test_attn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES ops/test_gdn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES ops/test_gdn_input_proj_conv_snapshot.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES ops/test_gdn_input_proj_conv_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES ops/test_gated_delta_net_replay_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES ops/test_gdn_replay_fold.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_add_q5_a16_test
  ops/linear_add/test_q5_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_bf16_a16_test
  ops/linear_add/test_bf16_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_w8_a16_test
  ops/linear_add/test_w8_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_op_test(ninfer_linear_add_nvfp4_test
  SOURCES ops/linear_add/test_nvfp4.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_linear_add_fp8_test
  SOURCES ops/linear_add/test_fp8.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_pair_w8_a16_test
  ops/linear_pair/test_w8_a16.cpp
  ops/linear_pair/linear_pair_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_q4_a16_test
  ops/linear_swiglu/test_q4_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_w8_a16_test
  ops/linear_swiglu/test_w8_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_nvfp4_test
  ops/linear_swiglu/test_nvfp4.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_fp8_test
  ops/linear_swiglu/test_fp8.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)

ninfer_add_test(ninfer_qwen3_6_27b_visual_scatter_test
  SOURCES targets/qwen3_6_27b/test_visual_scatter.cpp
  LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_qwen3_6_27b_visual_scatter_test PROPERTIES SKIP_RETURN_CODE 77)

# These tests deliberately compile against the exact target's internal contracts; the target does
# not export impl/ to ordinary consumers.
set(ninfer_qwen3_6_27b_internal_tests
  ninfer_qwen3_6_27b_visual_scatter_test)
foreach(test_target IN LISTS ninfer_qwen3_6_27b_internal_tests)
  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/impl
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
endforeach()
