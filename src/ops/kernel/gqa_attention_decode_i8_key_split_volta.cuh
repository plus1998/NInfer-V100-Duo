#pragma once

#include "ops/kernel/gqa_attention_decode_i8_tc_volta.cuh"

namespace ninfer::ops {

__device__ __forceinline__ void gqa_key_split_mma_pv_f32(float (&d)[8], const half2 (&p)[4],
                                                        const half2 (&v)[4]) {
    const int* Pxi = reinterpret_cast<const int*>(p);
    const int* Vxi = reinterpret_cast<const int*>(v);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(Pxi[0]), "r"(Pxi[1]), "r"(Vxi[0]), "r"(Vxi[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(Pxi[2]), "r"(Pxi[3]), "r"(Vxi[2]), "r"(Vxi[3]));
}

inline constexpr int kGqaKeySplitWarps   = 8;
inline constexpr int kGqaKeySplitBc      = 64;
inline constexpr int kGqaKeySplitStride  = kGqaHeadDim + 8;
inline constexpr int kGqaKeySplitPStride = kGqaKeySplitBc + 8;
inline constexpr int kGqaKeySplitPages   = 64;
inline constexpr std::size_t kGqaKeySplitSmemBytes =
    sizeof(half) * (32 * kGqaKeySplitStride + 2 * kGqaKeySplitBc * kGqaKeySplitStride +
                    32 * kGqaKeySplitPStride) +
    sizeof(float) * kGqaKeySplitWarps * 32 + sizeof(std::int32_t) * kGqaKeySplitPages;

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(256, 1) __global__ void gqa_attention_small_t_tc_volta_key_split_i8_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
    std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta == 8);

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int DimSplit      = WarpsPerCta;
    constexpr int Br            = 32;
    constexpr int Bc            = 64;
    constexpr int D             = kGqaHeadDim;
    constexpr int Threads       = DimSplit * 32;
    constexpr int DChunks       = D / 8;
    constexpr int DSlice        = D / DimSplit;
    constexpr int DChunksLocal  = DSlice / 8;
    constexpr int PageIds       = 64;
    constexpr int Groups        = D / kGqaKvQuantGroup;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(D % kGqaKvQuantGroup == 0, "head dim must divide into whole quant groups");
    static_assert(kGqaKvQuantGroup % 8 == 0,
                  "an 8-wide staging chunk must sit inside one quant group, so it needs one scale");

    constexpr int SmemPad    = 8;
    constexpr int SmemStride = D + SmemPad;
    static_assert(SmemPad % 8 == 0, "pad must preserve 16-byte alignment of the staging stores");
    constexpr int Stride = SmemStride;
    constexpr int PStride = Bc + 8;
    constexpr int Warps = WarpsPerCta;
    constexpr int KeyGroups = Bc / 8;
    extern __shared__ __align__(16) unsigned char shared_data[];
    half* q_s = reinterpret_cast<half*>(shared_data);
    half* k_s = q_s + Br * Stride;
    half* v_s = k_s + Bc * Stride;
    half* p_s = v_s + Bc * Stride;
    float* red_s = reinterpret_cast<float*>(p_s + Br * PStride);
    auto* physical_pages_s = reinterpret_cast<std::int32_t*>(red_s + Warps * Br);

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int dim_warp    = warp;
    int valid_tokens       = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens         = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;

    const int active_split_count =
        gqa_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {

        const int warps = Threads / 32;
        for (int pair = warp; pair < valid_tokens * Groups; pair += warps) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end || position < 0 ||
                position >= logical_capacity) {
                continue;
            }
            int physical_page     = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            physical_page          = __shfl_sync(FullMask, physical_page, 0);
            const int page_offset = position & kPagedKVPageMask;
            const int d0          = grp * kGqaKvQuantGroup + lane;
            const int d1          = d0 + 32;
            const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
            const float kv0         = __bfloat162float(input.k[src0]);
            const float kv1         = __bfloat162float(input.k[src1]);
            const float vv0         = __bfloat162float(input.v[src0]);
            const float vv1         = __bfloat162float(input.v[src1]);
            float kamax             = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax             = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax                   = warp_max(kamax, FullMask);
            vamax                   = warp_max(vamax, FullMask);
            const __half ksh        = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
            const __half vsh        = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
            const float k_inv       = __half2float(ksh) > 0.0f ? 1.0f / __half2float(ksh) : 0.0f;
            const float v_inv       = __half2float(vsh) > 0.0f ? 1.0f / __half2float(vsh) : 0.0f;
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(kv0, k_inv);
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(kv1, k_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(vv0, v_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(vv1, v_inv);
            if (lane == 0) {
                const std::int64_t so =
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
                cache_k_scale[so] = ksh;
                cache_v_scale[so] = vsh;
            }
        }
        __syncthreads();
    }

        for (int idx = tid; idx < Br * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head     = 0;
            int token      = 0;
            __nv_bfloat16 value = __float2bfloat16(0.0f);
            if (row < row_count) {
                gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
                if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                    value = q[gqa_q_index<Geometry>(q_head, d, token)];
                }
            }
            q_s[row * SmemStride + d] = __float2half(__bfloat162float(value));
        }
        __syncthreads();

    const int r_lo = volta_d_get_i(0) & ~2;
    const int r_hi = volta_d_get_i(0) | 2;
    int q_head_lo = 0, tok_lo = 0, q_head_hi = 0, tok_hi = 0;
    gqa_small_t_tc_row_to_qt<Geometry>(r_lo, tokens, kv_head, q_head_lo, tok_lo);
    gqa_small_t_tc_row_to_qt<Geometry>(r_hi, tokens, kv_head, q_head_hi, tok_hi);
    const int qabs_lo = (r_lo < row_count) ? pos[tok_lo] : -1;
    const int qabs_hi = (r_hi < row_count) ? pos[tok_hi] : -1;

    float acc_f[DChunksLocal][8];
#pragma unroll
    for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc_f[c][i] = 0.0f; }
    }
    float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F;
    float l_lo = 0.0f, l_hi = 0.0f;

    constexpr int ChunksPerThread = Bc * (D / 8) / Threads;
    static_assert(Bc * (D / 8) % Threads == 0);
    static_assert(Bc == kPagedKVPageSize && Threads == 256 && D / 8 == 32,
                  "fast tile load assumes one page per tile and key = warp + 8i");
    int2 k_codes[ChunksPerThread], v_codes[ChunksPerThread];
    half k_scales[ChunksPerThread], v_scales[ChunksPerThread];
    auto load_tile = [&](int kb) {
        const int k0   = first_tile + kb * Bc;
        const int page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];

        if (k0 >= split_start && k0 + Bc <= split_end) {
            const std::int64_t code_base =
                paged_kv_page_head_offset<kGqaKvQuantHeadDim, Geometry::KVHeads>(page, kv_head) +
                warp * kGqaKvQuantHeadDim + lane * 8;
            const std::int64_t scale_base =
                paged_kv_page_head_offset<kGqaKvQuantGroups, Geometry::KVHeads>(page, kv_head) +
                warp * kGqaKvQuantGroups + (lane * 8) / kGqaKvQuantGroup;
#pragma unroll
            for (int i = 0; i < ChunksPerThread; ++i) {
                k_codes[i]  = load_vec<int2>(cache_k_i8 + code_base + i * 8 * kGqaKvQuantHeadDim);
                v_codes[i]  = load_vec<int2>(cache_v_i8 + code_base + i * 8 * kGqaKvQuantHeadDim);
                k_scales[i] = cache_k_scale[scale_base + i * 8 * kGqaKvQuantGroups];
                v_scales[i] = cache_v_scale[scale_base + i * 8 * kGqaKvQuantGroups];
            }
            return;
        }
#pragma unroll
        for (int i = 0; i < ChunksPerThread; ++i) {
            const int chunk = tid + i * Threads;
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            const int key   = k0 + key_l;
            if (key >= split_start && key < split_end) {
                const int page_offset = key & kPagedKVPageMask;
                const std::int64_t code_off =
                    gqa_kv_quant_code_index<Geometry>(page, kv_head, d, page_offset);
                const std::int64_t scale_off = gqa_kv_quant_scale_index<Geometry>(
                    page, kv_head, d / kGqaKvQuantGroup, page_offset);
                k_codes[i]  = load_vec<int2>(&cache_k_i8[code_off]);
                v_codes[i]  = load_vec<int2>(&cache_v_i8[code_off]);
                k_scales[i] = cache_k_scale[scale_off];
                v_scales[i] = cache_v_scale[scale_off];
            } else {
                k_codes[i]  = make_int2(0, 0);
                v_codes[i]  = make_int2(0, 0);
                k_scales[i] = __float2half(0.0f);
                v_scales[i] = __float2half(0.0f);
            }
        }
    };

    auto dequant8 = [](int2 raw, half sc) {
        const __half2 scale2 = __half2half2(sc);
        const __half2 bias2  = __float2half2_rn(1152.0f);
        const unsigned lo = static_cast<unsigned>(raw.x) ^ 0x80808080u;
        const unsigned hi = static_cast<unsigned>(raw.y) ^ 0x80808080u;
        unsigned h[4];
        h[0] = __byte_perm(lo, 0x64646464u, 0x5150);
        h[1] = __byte_perm(lo, 0x64646464u, 0x5352);
        h[2] = __byte_perm(hi, 0x64646464u, 0x5150);
        h[3] = __byte_perm(hi, 0x64646464u, 0x5352);
        __half2 out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[j] = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h[j]), bias2), scale2);
        }
        return *reinterpret_cast<const int4*>(out);
    };

    load_tile(0);
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

#pragma unroll
        for (int i = 0; i < ChunksPerThread; ++i) {
            const int chunk = tid + i * Threads;
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            store_vec(&k_s[key_l * Stride + d], dequant8(k_codes[i], k_scales[i]));
            store_vec(&v_s[key_l * Stride + d], dequant8(v_codes[i], v_scales[i]));
        }
        if (kb + 1 < key_blocks) { load_tile(kb + 1); }
        __syncthreads();

        const int sub_k0 = k0 + warp * 8;
        float d_score[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int c = 0; c < DChunks; ++c) {
            half2 qf[4];
            volta_load_qp(qf, reinterpret_cast<const half2*>(&q_s[c * 8]), Stride / 2);
            half2 kf[4];
            volta_load_k(kf, reinterpret_cast<const half2*>(&k_s[warp * 8 * Stride + c * 8]),
                         Stride / 2);
            volta_mma_qk(d_score, qf, kf);
        }
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int row  = volta_d_get_i(l);
            const int key  = sub_k0 + volta_d_get_j(l);
            const int qabs = ((l & 2) == 0) ? qabs_lo : qabs_hi;
            const bool ok  = row < row_count && key >= split_start && key < split_end && key <= qabs;
            d_score[l]     = ok ? d_score[l] * scale : -CUDART_INF_F;
        }
        float bm_lo = fmaxf(fmaxf(d_score[0], d_score[1]), fmaxf(d_score[4], d_score[5]));
        float bm_hi = fmaxf(fmaxf(d_score[2], d_score[3]), fmaxf(d_score[6], d_score[7]));
        bm_lo       = fmaxf(bm_lo, __shfl_xor_sync(FullMask, bm_lo, 2, 32));
        bm_hi       = fmaxf(bm_hi, __shfl_xor_sync(FullMask, bm_hi, 2, 32));

        red_s[warp * 32 + (((lane & 2) == 0) ? r_lo : r_hi)] = ((lane & 2) == 0) ? bm_lo : bm_hi;
        __syncthreads();

        float tm_lo = -CUDART_INF_F, tm_hi = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            tm_lo = fmaxf(tm_lo, red_s[w * 32 + r_lo]);
            tm_hi = fmaxf(tm_hi, red_s[w * 32 + r_hi]);
        }
        const float new_m_lo = fmaxf(m_lo, tm_lo);
        const float new_m_hi = fmaxf(m_hi, tm_hi);
        const float alpha_lo =
            (m_lo == -CUDART_INF_F) ? 0.0f : exp2_approx((m_lo - new_m_lo) * Log2E);
        const float alpha_hi =
            (m_hi == -CUDART_INF_F) ? 0.0f : exp2_approx((m_hi - new_m_hi) * Log2E);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const float new_m = ((l & 2) == 0) ? new_m_lo : new_m_hi;
            d_score[l] = (new_m > -CUDART_INF_F && d_score[l] > -CUDART_INF_F)
                             ? exp2_approx((d_score[l] - new_m) * Log2E)
                             : 0.0f;
        }
        float bl_lo = d_score[0] + d_score[1] + d_score[4] + d_score[5];
        float bl_hi = d_score[2] + d_score[3] + d_score[6] + d_score[7];
        bl_lo       = bl_lo + __shfl_xor_sync(FullMask, bl_lo, 2, 32);
        bl_hi       = bl_hi + __shfl_xor_sync(FullMask, bl_hi, 2, 32);
        l_lo        = l_lo * alpha_lo + bl_lo;
        l_hi        = l_hi * alpha_hi + bl_hi;
        m_lo        = new_m_lo;
        m_hi        = new_m_hi;

#pragma unroll
        for (int l = 0; l < 8; ++l) {
            p_s[volta_d_get_i(l) * PStride + warp * 8 + volta_d_get_j(l)] = __float2half(d_score[l]);
        }
        __syncthreads();

#pragma unroll
        for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc_f[c][i] *= ((i & 2) == 0) ? alpha_lo : alpha_hi; }
        }
#pragma unroll
        for (int g = 0; g < KeyGroups; ++g) {
            half2 p[4];
            *reinterpret_cast<int4*>(p) =
                *reinterpret_cast<const int4*>(&p_s[lane * PStride + g * 8]);
#pragma unroll
            for (int c = 0; c < DChunksLocal; ++c) {
                half2 vf[4];
                volta_load_v(vf,
                             reinterpret_cast<const half2*>(
                                 &v_s[g * 8 * Stride + warp * DSlice + c * 8]),
                             Stride / 2);
                gqa_key_split_mma_pv_f32(acc_f[c], p, vf);
            }
        }
        __syncthreads();
    }

    red_s[warp * 32 + (((lane & 2) == 0) ? r_lo : r_hi)] = ((lane & 2) == 0) ? l_lo : l_hi;
    __syncthreads();
    const int row      = lane;
    const float own_m  = ((lane & 2) == 0) ? m_lo : m_hi;
    if (row < row_count) {
        int q_head = 0;
        int token  = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            if (warp == 0) {
                float own_l = 0.0f;
#pragma unroll
                for (int w = 0; w < Warps; ++w) { own_l += red_s[w * 32 + row]; }
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = own_m;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = own_l;
            }
        }
    }

#pragma unroll
    for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = volta_d_get_i(i);
            if (r >= row_count) { continue; }
            int qh = 0;
            int tk = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(r, tokens, kv_head, qh, tk);
            if (!gqa_valid_q_head<Geometry>(kv_head, qh)) { continue; }
            const int d = warp * DSlice + c * 8 + volta_d_get_j(i);
            partial_acc[gqa_partial_acc_index<Geometry>(qh, d, tk, split, tokens)] = __float2bfloat16(acc_f[c][i]);
        }
    }
#endif
}

}
