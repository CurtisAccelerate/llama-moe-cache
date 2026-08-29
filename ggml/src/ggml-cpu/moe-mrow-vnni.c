// Expert-major M-row IQ1_M kernel (Lane 2).
//
// Architecture borrowed from ExLlamaV3 CPU MoE, not a port:
//   - unique expert + assigned rows from the existing grouped scheduler
//   - sequential DRAM stream of GGML [N][K] expert weights (storage order)
//   - K-inner microkernel: decode each IQ1_M superblock once, M accumulators
//   - M>=2 specialized (M=1 stays stock GEMV) + M=5-8 unrolled tile
//   - fuse decode with integer dots against Q8_K activations
//   - AVX2-VNNI vpdpbusd 256-bit when compiled in; else maddubs
//
// ExLlama's measured 3.4× "K-major vs per-output-column" is sequential-for-their
// packed layout. GGML IQ1_M is [N][K]; packing/transpose is what ate the failed
// pair kernel, so we do not pack. Sequential here is one weight row's K, then
// the next row. The microkernel is K-inner with live M accs — not nrc=1 GEMV
// per assigned row (that is the rejected grouped-cpu speed path).
//
// Threading: existing ggml persistent threadpool, split on output rows.
// No new worker pool. P-only vs P+E is a later A/B; keep current nth.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "moe-mrow-vnni.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "simd-mappings.h"
#include "quants.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
typedef volatile LONG mrow_atomic_int;
static LONG mrow_atomic_fetch_add(mrow_atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG mrow_atomic_load(mrow_atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static int mrow_atomic_claim_zero(mrow_atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 1, 0) == 0;
}
static void mrow_atomic_store(mrow_atomic_int * ptr, LONG value) {
    InterlockedExchange(ptr, value);
}
#else
#include <stdatomic.h>
typedef atomic_int mrow_atomic_int;
static int mrow_atomic_fetch_add(mrow_atomic_int * ptr, int inc) {
    return atomic_fetch_add_explicit(ptr, inc, memory_order_relaxed);
}
static int mrow_atomic_load(mrow_atomic_int * ptr) {
    return atomic_load_explicit(ptr, memory_order_acquire);
}
static int mrow_atomic_claim_zero(mrow_atomic_int * ptr) {
    int expected = 0;
    return atomic_compare_exchange_strong_explicit(
        ptr, &expected, 1, memory_order_acq_rel, memory_order_acquire);
}
static void mrow_atomic_store(mrow_atomic_int * ptr, int value) {
    atomic_store_explicit(ptr, value, memory_order_release);
}
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define MROW_MAX_M 8
// Break-even: M=1 is stock GEMV (no reuse). N=8 verify is ~75% M=1 assignments.
#define MROW_MIN_M 2

static int g_moe_mrow_vnni = -1;
static int g_moe_mrow_vnni_dynamic = -1;
static int g_moe_mrow_vnni_min_m = -1;
static int g_moe_mrow_vnni_tiles_per_thread = -1;
static struct ggml_moe_mrow_vnni_counters g_mrow_ct;
static uint64_t g_iq2_signs64[128];
static mrow_atomic_int g_iq2_signs_state;

static void mrow_iq2_signs_init(void) {
    if (mrow_atomic_load(&g_iq2_signs_state) == 2) {
        return;
    }
    if (mrow_atomic_claim_zero(&g_iq2_signs_state)) {
        for (int code = 0; code < 128; ++code) {
            const uint8_t signs = ksigns_iq2xs[code];
            uint64_t packed = 0;
            for (int j = 0; j < 8; ++j) {
                const uint64_t lane = (signs & kmask_iq2xs[j]) ? 0xffu : 0x01u;
                packed |= lane << (8 * j);
            }
            g_iq2_signs64[code] = packed;
        }
        mrow_atomic_store(&g_iq2_signs_state, 2);
        return;
    }
    while (mrow_atomic_load(&g_iq2_signs_state) != 2) {
    }
}

static int env_flag(const char * name) {
    const char * e = getenv(name);
    return (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
}

void ggml_moe_mrow_vnni_set(int enabled) {
    g_moe_mrow_vnni = enabled ? 1 : 0;
    if (enabled) {
        mrow_iq2_signs_init();
    }
}

int ggml_moe_mrow_vnni_get(void) {
    if (g_moe_mrow_vnni < 0) {
        g_moe_mrow_vnni = env_flag("LLAMA_MOE_MROW_VNNI");
    }
    if (g_moe_mrow_vnni) {
        mrow_iq2_signs_init();
    }
    return g_moe_mrow_vnni;
}

static int env_int_clamped(const char * name, int fallback, int lo, int hi) {
    const char * e = getenv(name);
    if (!e || !e[0]) {
        return fallback;
    }
    const int value = atoi(e);
    return value < lo ? lo : (value > hi ? hi : value);
}

void ggml_moe_mrow_vnni_dynamic_set(int enabled) {
    g_moe_mrow_vnni_dynamic = enabled ? 1 : 0;
}

int ggml_moe_mrow_vnni_dynamic_get(void) {
    if (g_moe_mrow_vnni_dynamic < 0) {
        g_moe_mrow_vnni_dynamic = env_flag("LLAMA_MOE_MROW_VNNI_DYNAMIC");
    }
    return g_moe_mrow_vnni_dynamic;
}

int ggml_moe_mrow_vnni_supported(enum ggml_type wtype) {
    return wtype == GGML_TYPE_IQ1_M || wtype == GGML_TYPE_IQ2_XXS || wtype == GGML_TYPE_IQ3_XXS || wtype == GGML_TYPE_F32;
}

#if defined(__AVX2__) && (defined(__AVXVNNI__) || defined(GGML_AVX_VNNI))
#define MROW_VNNI 1
#else
#define MROW_VNNI 0
#endif

int ggml_moe_mrow_vnni_uses_vpdpbusd(void) {
    return MROW_VNNI;
}

const char * ggml_moe_mrow_vnni_kernel_name(void) {
#if defined(__AVX2__)
#if MROW_VNNI
    return "iq1m-iq2xxs-iq3xxs-mrow-avx2-vnni";
#else
    return "iq1m-iq2xxs-iq3xxs-mrow-avx2-maddubs";
#endif
#else
    return "iq1m-iq2xxs-iq3xxs-mrow-generic";
#endif
}

int ggml_moe_mrow_vnni_min_m(void) {
    if (g_moe_mrow_vnni_min_m < 0) {
        g_moe_mrow_vnni_min_m = env_int_clamped("LLAMA_MOE_MROW_VNNI_MIN_M", MROW_MIN_M, 2, MROW_MAX_M);
    }
    return g_moe_mrow_vnni_min_m;
}

static int ggml_moe_mrow_vnni_tiles_per_thread(void) {
    if (g_moe_mrow_vnni_tiles_per_thread < 0) {
        g_moe_mrow_vnni_tiles_per_thread =
            env_int_clamped("LLAMA_MOE_MROW_VNNI_TILES_PER_THREAD", 4, 1, 16);
    }
    return g_moe_mrow_vnni_tiles_per_thread;
}

void ggml_moe_mrow_vnni_counters_reset(void) {
    memset(&g_mrow_ct, 0, sizeof(g_mrow_ct));
}

void ggml_moe_mrow_vnni_counters_get(struct ggml_moe_mrow_vnni_counters * out) {
    if (out) {
        *out = g_mrow_ct;
    }
}

static void mrow_count_bin(uint64_t * bins, int M) {
    if (M <= 0) {
        return;
    }
    const int bin = M > MROW_MAX_M ? MROW_MAX_M : M;
    bins[bin]++;
}

void ggml_moe_mrow_vnni_note_stock(const int64_t * counts, int n_as) {
    g_mrow_ct.ops_stock++;
    if (!counts || n_as <= 0) {
        return;
    }
    for (int i = 0; i < n_as; ++i) {
        mrow_count_bin(g_mrow_ct.fallback, (int) counts[i]);
    }
}

// ---------------------------------------------------------------------------
// F32: weight element loaded once, M accumulators. Tests / fallback.
// ---------------------------------------------------------------------------

static void mrow_f32(
        int64_t ne00,
        const float * wrow,
        int M,
        const float * cols[],
        float * dst_cols[],
        int64_t ir0) {
    float acc[MROW_MAX_M];
    for (int m = 0; m < M; ++m) {
        acc[m] = 0.0f;
    }
    for (int64_t k = 0; k < ne00; ++k) {
        const float wk = wrow[k];
        for (int m = 0; m < M; ++m) {
            acc[m] += wk * cols[m][k];
        }
    }
    for (int m = 0; m < M; ++m) {
        dst_cols[m][ir0] = acc[m];
    }
}

// ---------------------------------------------------------------------------
// IQ1_M generic: decode each superblock once, apply M Q8_K rows.
// ---------------------------------------------------------------------------

static void mrow_iq1m_generic(
        int64_t n,
        const block_iq1_m * x,
        int M,
        const block_q8_K * ys[],
        float * dst_cols[],
        int64_t ir0) {
    assert(n % QK_K == 0);
    const int nb = (int) (n / QK_K);
    float acc[MROW_MAX_M];
    for (int m = 0; m < M; ++m) {
        acc[m] = 0.0f;
    }

    iq1m_scale_t scale;

    for (int i = 0; i < nb; ++i) {
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *) x[i].scales;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d0 = GGML_CPU_FP16_TO_FP32(scale.f16);

        const int8_t * q8s[MROW_MAX_M];
        float dm[MROW_MAX_M];
        for (int m = 0; m < M; ++m) {
            q8s[m] = ys[m][i].qs;
            dm[m] = d0 * ys[m][i].d;
        }

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int dlt[4] = {
                qh[0] & 0x08 ? -1 : 1,
                qh[0] & 0x80 ? -1 : 1,
                qh[1] & 0x08 ? -1 : 1,
                qh[1] & 0x80 ? -1 : 1,
            };
            const int8_t * grid[4];
            for (int l = 0; l < 4; ++l) {
                grid[l] = (const int8_t *) (iq1s_grid + (qs[l] | (((uint16_t) qh[l / 2] << (8 - 4 * (l % 2))) & 0x700)));
            }
            const int ls1 = 2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1;
            const int ls2 = 2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1;

            for (int m = 0; m < M; ++m) {
                const int8_t * q8 = q8s[m];
                int sum1[2] = {0, 0};
                int sum2[2] = {0, 0};
                for (int l = 0; l < 4; ++l) {
                    int lsum1 = 0, lsum2 = 0;
                    for (int j = 0; j < 8; ++j) {
                        lsum1 += q8[j] * grid[l][j];
                        lsum2 += q8[j];
                    }
                    q8 += 8;
                    sum1[l / 2] += lsum1;
                    sum2[l / 2] += lsum2 * dlt[l];
                }
                q8s[m] = q8;
                acc[m] += dm[m] * ((sum1[0] * ls1 + sum1[1] * ls2) + IQ1M_DELTA * (sum2[0] * ls1 + sum2[1] * ls2));
            }
            qs += 4;
            qh += 2;
        }
    }

    for (int m = 0; m < M; ++m) {
        dst_cols[m][ir0] = acc[m];
    }
}

#if defined(__AVX2__)

#if defined(_MSC_VER) && !defined(__FMA__)
#define __FMA__
#endif
#if defined(_MSC_VER) && !defined(__F16C__)
#define __F16C__
#endif

#include <immintrin.h>

static inline float mrow_hsum_ps(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i mrow_dot_i8(const __m256i w, const __m256i a) {
    const __m256i aw = _mm256_sign_epi8(w, w);
    const __m256i sa = _mm256_sign_epi8(a, w);
#if MROW_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
#else
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(ones, _mm256_maddubs_epi16(aw, sa));
#endif
}

static inline __m256i mrow_dot_delta(const __m256i a, const __m256i delta, const __m256i mone8) {
    const __m256i sa = _mm256_sign_epi8(a, delta);
#if MROW_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), mone8, sa);
#else
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(ones, _mm256_maddubs_epi16(mone8, sa));
#endif
}

static inline __m256i mrow_scale_i32(const __m256i sum32, const __m256i scale16) {
    // VPSHUFB is per 128-bit lane, so the two halves of scale16 can differ.
    // vpdpbusd/maddubs→epi32 lanes 0-3 are the low half, 4-7 the high half.
    const __m128i s32_lo = _mm_cvtepi16_epi32(_mm256_castsi256_si128(scale16));
    const __m128i s32_hi = _mm_cvtepi16_epi32(_mm256_extracti128_si256(scale16, 1));
    const __m256i s32 = _mm256_inserti128_si256(_mm256_castsi128_si256(s32_lo), s32_hi, 1);
    return _mm256_mullo_epi32(sum32, s32);
}

// Decode one 64-weight chunk (2×32) of an IQ1_M superblock.
static inline void mrow_iq1m_decode64(
        const uint8_t * qs,
        const uint8_t * qh,
        __m256i * q1b_1,
        __m256i * q1b_2,
        __m256i * delta1,
        __m256i * delta2) {
    const __m256i mone8 = _mm256_set1_epi8(1);
#ifdef __BMI2__
    const uint64_t packed_idx1 = _pdep_u64(*(const uint32_t *) qs, 0x00ff00ff00ff00ffULL)
                               | _pdep_u64(*(const uint16_t *) (qh) & 0x7777, 0xf000f000f000f00ULL);
    const uint64_t packed_idx2 = _pdep_u64(*(const uint32_t *) (qs + 4), 0x00ff00ff00ff00ffULL)
                               | _pdep_u64(*(const uint16_t *) (qh + 2) & 0x7777, 0xf000f000f000f00ULL);
    const uint16_t * idx1 = (const uint16_t *) (&packed_idx1);
    const uint16_t * idx2 = (const uint16_t *) (&packed_idx2);
    *q1b_1 = _mm256_set_epi64x(iq1s_grid[idx1[3]], iq1s_grid[idx1[2]], iq1s_grid[idx1[1]], iq1s_grid[idx1[0]]);
    *q1b_2 = _mm256_set_epi64x(iq1s_grid[idx2[3]], iq1s_grid[idx2[2]], iq1s_grid[idx2[1]], iq1s_grid[idx2[0]]);
    const uint64_t delta_sign = _pdep_u64(*(const uint32_t *) (qh) & 0x88888888, 0xf0f0f0f0f0f0f0f0ULL);
    *delta1 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32((int) delta_sign)));
    *delta2 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32((int) (delta_sign >> 32))));
#else
    UNUSED(mone8);
    *q1b_1 = _mm256_set_epi64x(
            iq1s_grid[qs[3] | (((uint16_t) qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t) qh[1] << 8) & 0x700)],
            iq1s_grid[qs[1] | (((uint16_t) qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t) qh[0] << 8) & 0x700)]);
    *q1b_2 = _mm256_set_epi64x(
            iq1s_grid[qs[7] | (((uint16_t) qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t) qh[3] << 8) & 0x700)],
            iq1s_grid[qs[5] | (((uint16_t) qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t) qh[2] << 8) & 0x700)]);
    *delta1 = _mm256_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffffULL : 0x0101010101010101ULL,
                                qh[1] & 0x08 ? 0xffffffffffffffffULL : 0x0101010101010101ULL,
                                qh[0] & 0x80 ? 0xffffffffffffffffULL : 0x0101010101010101ULL,
                                qh[0] & 0x08 ? 0xffffffffffffffffULL : 0x0101010101010101ULL);
    *delta2 = _mm256_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffffULL : 0x0101010101010101ULL,
                                qh[3] & 0x08 ? 0xffffffffffffffffULL : 0x0101010101010101ULL,
                                qh[2] & 0x80 ? 0xffffffffffffffffULL : 0x0101010101010101ULL,
                                qh[2] & 0x08 ? 0xffffffffffffffffULL : 0x0101010101010101ULL);
#endif
}

#define MROW_IQ1M_APPLY_M(m, q8, accum1, accum2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8) \
    do { \
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *) (q8)); \
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *) ((q8) + 32)); \
        const __m256i p1 = mrow_scale_i32(mrow_dot_i8((q1b_1), q8b_1), (scale1)); \
        const __m256i p2 = mrow_scale_i32(mrow_dot_i8((q1b_2), q8b_2), (scale2)); \
        const __m256i p3 = mrow_scale_i32(mrow_dot_delta(q8b_1, (delta1), (mone8)), (scale1)); \
        const __m256i p4 = mrow_scale_i32(mrow_dot_delta(q8b_2, (delta2), (mone8)), (scale2)); \
        (accum1)[m] = _mm256_add_epi32((accum1)[m], _mm256_add_epi32(p1, p2)); \
        (accum2)[m] = _mm256_add_epi32((accum2)[m], _mm256_add_epi32(p3, p4)); \
    } while (0)

#define DEFINE_MROW_IQ1M_AVX2(M) \
static void mrow_iq1m_avx2_##M( \
        int64_t n, \
        const block_iq1_m * x, \
        const block_q8_K * ys[], \
        float * dst_cols[], \
        int64_t ir0) { \
    assert(n % QK_K == 0); \
    const int nb = (int) (n / QK_K); \
    const __m256i mask = _mm256_set1_epi16(0x7); \
    const __m256i mone = _mm256_set1_epi16(1); \
    const __m256i mone8 = _mm256_set1_epi8(1); \
    const __m256i mtwo8 = _mm256_set1_epi8(2); \
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0); \
    __m256 facc1[M]; \
    __m256 facc2[M]; \
    for (int m = 0; m < (M); ++m) { \
        facc1[m] = _mm256_setzero_ps(); \
        facc2[m] = _mm256_setzero_ps(); \
    } \
    iq1m_scale_t scale; \
    for (int i = 0; i < nb; ++i) { \
        const uint8_t  * qs = x[i].qs; \
        const uint8_t  * qh = x[i].qh; \
        const uint16_t * sc = (const uint16_t *) x[i].scales; \
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000); \
        __m256i scales = _mm256_set1_epi64x((long long) *(const uint64_t *) sc); \
        scales = _mm256_srlv_epi64(scales, scales_shift); \
        scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone); \
        __m256i scales_idx1 = _mm256_set1_epi16(0x0100); \
        __m256i scales_idx2 = _mm256_add_epi8(scales_idx1, _mm256_set1_epi8(8)); \
        const int8_t * q8[8]; \
        float yd[8]; \
        for (int m = 0; m < (M); ++m) { \
            q8[m] = ys[m][i].qs; \
            yd[m] = ys[m][i].d * GGML_CPU_FP16_TO_FP32(scale.f16); \
        } \
        __m256i sumi1[8]; \
        __m256i sumi2[8]; \
        for (int m = 0; m < (M); ++m) { \
            sumi1[m] = _mm256_setzero_si256(); \
            sumi2[m] = _mm256_setzero_si256(); \
        } \
        for (int ib = 0; ib < QK_K / 32; ib += 2) { \
            __m256i q1b_1, q1b_2, delta1, delta2; \
            mrow_iq1m_decode64(qs, qh, &q1b_1, &q1b_2, &delta1, &delta2); \
            const __m256i scale1 = _mm256_shuffle_epi8(scales, scales_idx1); \
            const __m256i scale2 = _mm256_shuffle_epi8(scales, scales_idx2); \
            scales_idx1 = _mm256_add_epi8(scales_idx1, mtwo8); \
            scales_idx2 = _mm256_add_epi8(scales_idx2, mtwo8); \
            if ((M) == 1) { \
                MROW_IQ1M_APPLY_M(0, q8[0], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                q8[0] += 64; \
            } else if ((M) == 2) { \
                MROW_IQ1M_APPLY_M(0, q8[0], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(1, q8[1], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                q8[0] += 64; q8[1] += 64; \
            } else if ((M) == 3) { \
                MROW_IQ1M_APPLY_M(0, q8[0], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(1, q8[1], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(2, q8[2], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                q8[0] += 64; q8[1] += 64; q8[2] += 64; \
            } else if ((M) == 4) { \
                MROW_IQ1M_APPLY_M(0, q8[0], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(1, q8[1], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(2, q8[2], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(3, q8[3], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                q8[0] += 64; q8[1] += 64; q8[2] += 64; q8[3] += 64; \
            } else { \
                /* M=5-8 unrolled tile */ \
                MROW_IQ1M_APPLY_M(0, q8[0], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(1, q8[1], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(2, q8[2], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(3, q8[3], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                MROW_IQ1M_APPLY_M(4, q8[4], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); \
                q8[0] += 64; q8[1] += 64; q8[2] += 64; q8[3] += 64; q8[4] += 64; \
                if ((M) >= 6) { MROW_IQ1M_APPLY_M(5, q8[5], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); q8[5] += 64; } \
                if ((M) >= 7) { MROW_IQ1M_APPLY_M(6, q8[6], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); q8[6] += 64; } \
                if ((M) >= 8) { MROW_IQ1M_APPLY_M(7, q8[7], sumi1, sumi2, q1b_1, q1b_2, delta1, delta2, scale1, scale2, mone8); q8[7] += 64; } \
            } \
            qs += 8; \
            qh += 4; \
        } \
        for (int m = 0; m < (M); ++m) { \
            const __m256 d = _mm256_set1_ps(yd[m]); \
            facc1[m] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1[m]), facc1[m]); \
            facc2[m] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2[m]), facc2[m]); \
        } \
    } \
    for (int m = 0; m < (M); ++m) { \
        dst_cols[m][ir0] = mrow_hsum_ps(facc1[m]) + IQ1M_DELTA * mrow_hsum_ps(facc2[m]); \
    } \
}

DEFINE_MROW_IQ1M_AVX2(1)
DEFINE_MROW_IQ1M_AVX2(2)
DEFINE_MROW_IQ1M_AVX2(3)
DEFINE_MROW_IQ1M_AVX2(4)
DEFINE_MROW_IQ1M_AVX2(5)
DEFINE_MROW_IQ1M_AVX2(6)
DEFINE_MROW_IQ1M_AVX2(7)
DEFINE_MROW_IQ1M_AVX2(8)

static void mrow_iq1m_avx2(
        int64_t n,
        const block_iq1_m * x,
        int M,
        const block_q8_K * ys[],
        float * dst_cols[],
        int64_t ir0) {
    switch (M) {
        case 1: mrow_iq1m_avx2_1(n, x, ys, dst_cols, ir0); break;
        case 2: mrow_iq1m_avx2_2(n, x, ys, dst_cols, ir0); break;
        case 3: mrow_iq1m_avx2_3(n, x, ys, dst_cols, ir0); break;
        case 4: mrow_iq1m_avx2_4(n, x, ys, dst_cols, ir0); break;
        case 5: mrow_iq1m_avx2_5(n, x, ys, dst_cols, ir0); break;
        case 6: mrow_iq1m_avx2_6(n, x, ys, dst_cols, ir0); break;
        case 7: mrow_iq1m_avx2_7(n, x, ys, dst_cols, ir0); break;
        case 8: mrow_iq1m_avx2_8(n, x, ys, dst_cols, ir0); break;
        default: mrow_iq1m_generic(n, x, M, ys, dst_cols, ir0); break;
    }
}

static inline uint64_t mrow_iq2xxs_signs64(uint8_t code) {
    return g_iq2_signs64[code & 127];
}

static void mrow_iq2xxs_avx2_m2(
        int64_t n,
        const block_iq2_xxs * x,
        const block_q8_K * ys[2],
        float * dst_cols[2],
        int64_t ir0) {
    assert(n % QK_K == 0);
    const int nb = (int) (n / QK_K);
    __m256 accumf[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };

    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const uint16_t * q2 = x[i].qs;
        const int8_t * q8[2] = { ys[0][i].qs, ys[1][i].qs };
        __m256i sumi1[2] = { _mm256_setzero_si256(), _mm256_setzero_si256() };
        __m256i sumi2[2] = { _mm256_setzero_si256(), _mm256_setzero_si256() };

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            uint32_t aux32[4];
            memcpy(aux32, q2, sizeof(aux32));
            q2 += 8;
            const uint8_t * aux8 = (const uint8_t *) aux32;
            const __m256i q2_1 = _mm256_set_epi64x(
                iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]],
                iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(
                iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]],
                iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(
                (long long) mrow_iq2xxs_signs64((aux32[1] >> 21) & 127),
                (long long) mrow_iq2xxs_signs64((aux32[1] >> 14) & 127),
                (long long) mrow_iq2xxs_signs64((aux32[1] >>  7) & 127),
                (long long) mrow_iq2xxs_signs64((aux32[1] >>  0) & 127));
            const __m256i s2_2 = _mm256_set_epi64x(
                (long long) mrow_iq2xxs_signs64((aux32[3] >> 21) & 127),
                (long long) mrow_iq2xxs_signs64((aux32[3] >> 14) & 127),
                (long long) mrow_iq2xxs_signs64((aux32[3] >>  7) & 127),
                (long long) mrow_iq2xxs_signs64((aux32[3] >>  0) & 127));
            const __m256i ls1 = _mm256_set1_epi16((short) (2 * (aux32[1] >> 28) + 1));
            const __m256i ls2 = _mm256_set1_epi16((short) (2 * (aux32[3] >> 28) + 1));

            for (int m = 0; m < 2; ++m) {
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8[m]); q8[m] += 32;
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) q8[m]); q8[m] += 32;
                const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
                const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
                sumi1[m] = _mm256_add_epi32(sumi1[m], _mm256_madd_epi16(dot1, ls1));
                sumi2[m] = _mm256_add_epi32(sumi2[m], _mm256_madd_epi16(dot2, ls2));
            }
        }

        for (int m = 0; m < 2; ++m) {
            const float d = xd * ys[m][i].d;
            accumf[m] = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[m], sumi2[m])),
                accumf[m]);
        }
    }

    dst_cols[0][ir0] = 0.125f * mrow_hsum_ps(accumf[0]);
    dst_cols[1][ir0] = 0.125f * mrow_hsum_ps(accumf[1]);
}

#endif // __AVX2__

static void mrow_iq1m(
        int64_t n,
        const block_iq1_m * x,
        int M,
        const block_q8_K * ys[],
        float * dst_cols[],
        int64_t ir0) {
#if defined(__AVX2__)
    mrow_iq1m_avx2(n, x, M, ys, dst_cols, ir0);
#else
    mrow_iq1m_generic(n, x, M, ys, dst_cols, ir0);
#endif
}

static void mrow_iq3xxs_m2_generic(
        int64_t n,
        const block_iq3_xxs * x,
        const block_q8_K * y0,
        const block_q8_K * y1,
        float * dst0,
        float * dst1,
        int64_t ir0) {
    assert(n % QK_K == 0);
    const int nb = (int) (n / QK_K);
    float sumf0 = 0.0f;
    float sumf1 = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * q3 = x[i].qs;
        const uint8_t * gas = x[i].qs + QK_K / 4;
        const int8_t * q80 = y0[i].qs;
        const int8_t * q81 = y1[i].qs;
        int32_t bsum0 = 0;
        int32_t bsum1 = 0;

        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            uint32_t aux32;
            memcpy(&aux32, gas, sizeof(aux32));
            gas += sizeof(aux32);
            const int32_t ls = 2 * (int32_t) (aux32 >> 28) + 1;
            int32_t sumi0 = 0;
            int32_t sumi1 = 0;

            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid + q3[2 * l + 0]);
                const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid + q3[2 * l + 1]);
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                for (int j = 0; j < 4; ++j) {
                    const int sign0 = signs & kmask_iq2xs[j + 0] ? -1 : 1;
                    const int sign1 = signs & kmask_iq2xs[j + 4] ? -1 : 1;
                    sumi0 += grid1[j] * q80[j + 0] * sign0 + grid2[j] * q80[j + 4] * sign1;
                    sumi1 += grid1[j] * q81[j + 0] * sign0 + grid2[j] * q81[j + 4] * sign1;
                }
                q80 += 8;
                q81 += 8;
            }
            q3 += 8;
            bsum0 += sumi0 * ls;
            bsum1 += sumi1 * ls;
        }

        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
        sumf0 += dx * y0[i].d * bsum0;
        sumf1 += dx * y1[i].d * bsum1;
    }

    dst0[ir0] = 0.25f * sumf0;
    dst1[ir0] = 0.25f * sumf1;
}

#if defined(__AVX2__)
static inline uint64_t mrow_iq3xxs_sign64(uint8_t signs) {
#ifdef __BMI2__
    return _pdep_u64(signs, 0x8080808080808080ULL) | 0x0101010101010101ULL;
#else
    uint64_t expanded = 0;
    for (int j = 0; j < 8; ++j) {
        expanded |= (uint64_t) ((signs & (1u << j)) ? 0x81 : 0x01) << (8 * j);
    }
    return expanded;
#endif
}

static inline __m256i mrow_iq3xxs_dot32(const __m256i q3, const __m256i q8, int scale) {
#if MROW_VNNI
    const __m256i dot = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q3, q8);
    return _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale));
#else
    const __m256i dot = _mm256_maddubs_epi16(q3, q8);
    return _mm256_madd_epi16(dot, _mm256_set1_epi16((short) scale));
#endif
}

static void mrow_iq3xxs_m2_avx2(
        int64_t n,
        const block_iq3_xxs * x,
        const block_q8_K * y0,
        const block_q8_K * y1,
        float * dst0,
        float * dst1,
        int64_t ir0) {
    assert(n % QK_K == 0);
    const int nb = (int) (n / QK_K);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const uint8_t * q3 = x[i].qs;
        const uint8_t * gas = x[i].qs + QK_K / 4;
        const int8_t * q80 = y0[i].qs;
        const int8_t * q81 = y1[i].qs;
        __m256i sumi0a = _mm256_setzero_si256();
        __m256i sumi0b = _mm256_setzero_si256();
        __m256i sumi1a = _mm256_setzero_si256();
        __m256i sumi1b = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q3a = _mm256_set_epi32(
                    iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                    iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q3b = _mm256_set_epi32(
                    iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                    iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;

            uint32_t aux32[2];
            memcpy(aux32, gas, sizeof(aux32));
            gas += sizeof(aux32);
            const __m256i signs_a = _mm256_set_epi64x(
                    mrow_iq3xxs_sign64(ksigns_iq2xs[(aux32[0] >> 21) & 127]),
                    mrow_iq3xxs_sign64(ksigns_iq2xs[(aux32[0] >> 14) & 127]),
                    mrow_iq3xxs_sign64(ksigns_iq2xs[(aux32[0] >>  7) & 127]),
                    mrow_iq3xxs_sign64(ksigns_iq2xs[(aux32[0] >>  0) & 127]));
            const __m256i signs_b = _mm256_set_epi64x(
                    mrow_iq3xxs_sign64(ksigns_iq2xs[(aux32[1] >> 21) & 127]),
                    mrow_iq3xxs_sign64(ksigns_iq2xs[(aux32[1] >> 14) & 127]),
                    mrow_iq3xxs_sign64(ksigns_iq2xs[(aux32[1] >>  7) & 127]),
                    mrow_iq3xxs_sign64(ksigns_iq2xs[(aux32[1] >>  0) & 127]));
            const int scale_a = 2 * (int) (aux32[0] >> 28) + 1;
            const int scale_b = 2 * (int) (aux32[1] >> 28) + 1;

            const __m256i q80a = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q80), signs_a);
            const __m256i q81a = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q81), signs_a);
            q80 += 32;
            q81 += 32;
            const __m256i q80b = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q80), signs_b);
            const __m256i q81b = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q81), signs_b);
            q80 += 32;
            q81 += 32;

            sumi0a = _mm256_add_epi32(sumi0a, mrow_iq3xxs_dot32(q3a, q80a, scale_a));
            sumi0b = _mm256_add_epi32(sumi0b, mrow_iq3xxs_dot32(q3b, q80b, scale_b));
            sumi1a = _mm256_add_epi32(sumi1a, mrow_iq3xxs_dot32(q3a, q81a, scale_a));
            sumi1b = _mm256_add_epi32(sumi1b, mrow_iq3xxs_dot32(q3b, q81b, scale_b));
        }

        const __m256i isum0 = _mm256_add_epi32(sumi0a, sumi0b);
        const __m256i isum1 = _mm256_add_epi32(sumi1a, sumi1b);
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(dx * y0[i].d), _mm256_cvtepi32_ps(isum0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dx * y1[i].d), _mm256_cvtepi32_ps(isum1), acc1);
    }

    dst0[ir0] = 0.25f * mrow_hsum_ps(acc0);
    dst1[ir0] = 0.25f * mrow_hsum_ps(acc1);
}
#endif

static void mrow_iq3xxs_m2(
        int64_t n,
        const block_iq3_xxs * x,
        const block_q8_K * y0,
        const block_q8_K * y1,
        float * dst0,
        float * dst1,
        int64_t ir0) {
#if defined(__AVX2__)
    mrow_iq3xxs_m2_avx2(n, x, y0, y1, dst0, dst1, ir0);
#else
    mrow_iq3xxs_m2_generic(n, x, y0, y1, dst0, dst1, ir0);
#endif
}

static int mrow_kernel_accepts(enum ggml_type type, int M) {
    if (M < ggml_moe_mrow_vnni_min_m()) {
        return 0;
    }
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS ? M == 2 : 1;
}

static void mrow_stock(
        ggml_vec_dot_t vec_dot,
        int64_t ne00,
        const char * wrow,
        int M,
        const char * cols[],
        float * dst_cols[],
        int64_t ir0) {
    for (int m = 0; m < M; ++m) {
        float acc;
        vec_dot((int) ne00, &acc, 0, wrow, 0, cols[m], 0, 1);
        dst_cols[m][ir0] = acc;
    }
}

void ggml_moe_mrow_vnni_apply(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const char * src0_data,
        enum ggml_type src0_type,
        int64_t ne00,
        int64_t ne01,
        int64_t ne11,
        size_t nb01,
        size_t nb02,
        size_t nb1,
        size_t nb2,
        size_t nb11,
        size_t nb12,
        int n_as,
        int ids_ne0,
        int ids_ne1,
        const int64_t * matrix_row_counts,
        const struct mmid_row_mapping * matrix_rows,
        bool src1_cont,
        enum ggml_type vec_dot_type,
        const void * wdata,
        void * atomic_current_chunk_raw,
        size_t atomic_chunk_stride) {

    const int ith = params->ith;
    const int nth = params->nth;

    const struct ggml_type_traits_cpu * tt = ggml_get_type_traits_cpu(src0_type);
    ggml_vec_dot_t const vec_dot = tt ? tt->vec_dot : NULL;
    const size_t row_size = ggml_row_size(vec_dot_type, ne00);
    const int row_stride = ids_ne0 * ids_ne1;
    const int use_linear = src1_cont || dst->src[1]->type != vec_dot_type;

    // Persistent ggml pool: one contiguous output-row slab per thread so each
    // worker streams expert weights sequentially (no new pin/affinity here).
    const int64_t dr0 = (ne01 + nth - 1) / nth;
    const int64_t ir0_start = ith * dr0;
    const int64_t ir0_end = MIN(ir0_start + dr0, ne01);
    if (ir0_start >= ir0_end) {
        return;
    }

    if (ith == 0) {
        g_mrow_ct.ops_kernel++;
    }

    const char * cols[MROW_MAX_M];
    float * dst_cols[MROW_MAX_M];
    const float * fcols[MROW_MAX_M];
    const block_q8_K * qcols[MROW_MAX_M];

    // Arrow Lake mixes fast P cores and slower E cores. The original fixed
    // output slabs made the whole op wait for the slowest slab. This opt-in
    // path keeps the same M-row dot kernels and arithmetic, but exposes many
    // small output tiles through ggml's existing per-expert atomic queue.
    if (ggml_moe_mrow_vnni_dynamic_get() && atomic_current_chunk_raw && atomic_chunk_stride > 0) {
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            const int64_t Mfull = matrix_row_counts[cur_a];
            if (Mfull <= 0) {
                continue;
            }

            const char * src0_cur = src0_data + (size_t) cur_a * nb02;
            const int64_t n_mtiles = (Mfull + MROW_MAX_M - 1) / MROW_MAX_M;

            // Default four tiles per worker lets fast cores steal work from
            // slow cores. The research env can sweep this; never make empty
            // output tiles.
            int64_t n_out_tiles = (ne01 + 63) / 64;
            n_out_tiles = MAX(n_out_tiles, (int64_t) nth * ggml_moe_mrow_vnni_tiles_per_thread());
            n_out_tiles = MIN(n_out_tiles, ne01);
            const int64_t dr0 = (ne01 + n_out_tiles - 1) / n_out_tiles;
            const int64_t n_chunks = n_mtiles * n_out_tiles;

            if (ith == 0) {
                for (int64_t mtile = 0; mtile < n_mtiles; ++mtile) {
                    const int M = (int) MIN((int64_t) MROW_MAX_M, Mfull - mtile * MROW_MAX_M);
                    if (!mrow_kernel_accepts(src0_type, M)) {
                        mrow_count_bin(g_mrow_ct.fallback, M);
                    } else {
                        mrow_count_bin(g_mrow_ct.hit, M);
                    }
                }
            }

            mrow_atomic_int * current_chunk_ctr = (mrow_atomic_int *) ((char *) atomic_current_chunk_raw +
                                                                       (size_t) cur_a * atomic_chunk_stride);
            int current_chunk = (int) mrow_atomic_fetch_add(current_chunk_ctr, 1);

            while (current_chunk < n_chunks) {
                const int64_t mtile = current_chunk / n_out_tiles;
                const int64_t otile = current_chunk % n_out_tiles;
                const int64_t m0 = mtile * MROW_MAX_M;
                const int M = (int) MIN((int64_t) MROW_MAX_M, Mfull - m0);
                const int64_t tile_start = otile * dr0;
                const int64_t tile_end = MIN(tile_start + dr0, ne01);

                for (int m = 0; m < M; ++m) {
                    const struct mmid_row_mapping map = matrix_rows[cur_a * row_stride + (int) m0 + m];
                    const int id = map.i1;
                    const int64_t i11 = id % ne11;
                    const int64_t i12 = map.i2;
                    cols[m] = (const char *) wdata +
                        (use_linear ? (i11 + i12 * ne11) * row_size : (i11 * nb11 + i12 * nb12));
                    dst_cols[m] = (float *) ((char *) dst->data + (id * nb1 + i12 * nb2));
                    fcols[m] = (const float *) cols[m];
                    qcols[m] = (const block_q8_K *) cols[m];
                }

                if (!mrow_kernel_accepts(src0_type, M) && vec_dot) {
                    for (int64_t ir0 = tile_start; ir0 < tile_end; ++ir0) {
                        mrow_stock(vec_dot, ne00, src0_cur + ir0 * nb01, M, cols, dst_cols, ir0);
                    }
                } else if (src0_type == GGML_TYPE_F32) {
                    for (int64_t ir0 = tile_start; ir0 < tile_end; ++ir0) {
                        const float * wrow = (const float *) (src0_cur + ir0 * nb01);
                        mrow_f32(ne00, wrow, M, fcols, dst_cols, ir0);
                    }
                } else if (src0_type == GGML_TYPE_IQ1_M) {
                    for (int64_t ir0 = tile_start; ir0 < tile_end; ++ir0) {
                        const block_iq1_m * wrow = (const block_iq1_m *) (src0_cur + ir0 * nb01);
#if defined(__AVX2__)
                        if (ir0 + 1 < tile_end) {
                            _mm_prefetch((const char *) (src0_cur + (ir0 + 1) * nb01), _MM_HINT_T0);
                        }
#endif
                        mrow_iq1m(ne00, wrow, M, qcols, dst_cols, ir0);
                    }
                } else if (src0_type == GGML_TYPE_IQ2_XXS) {
#if defined(__AVX2__)
                    const block_q8_K * iq2_cols[2] = { qcols[0], qcols[1] };
                    float * iq2_dst[2] = { dst_cols[0], dst_cols[1] };
                    for (int64_t ir0 = tile_start; ir0 < tile_end; ++ir0) {
                        const block_iq2_xxs * wrow = (const block_iq2_xxs *) (src0_cur + ir0 * nb01);
                        mrow_iq2xxs_avx2_m2(ne00, wrow, iq2_cols, iq2_dst, ir0);
                    }
#endif
                } else {
                    assert(src0_type == GGML_TYPE_IQ3_XXS && M == 2);
                    for (int64_t ir0 = tile_start; ir0 < tile_end; ++ir0) {
                        const block_iq3_xxs * wrow = (const block_iq3_xxs *) (src0_cur + ir0 * nb01);
#if defined(__AVX2__)
                        if (ir0 + 1 < tile_end) {
                            _mm_prefetch((const char *) (src0_cur + (ir0 + 1) * nb01), _MM_HINT_T0);
                        }
#endif
                        mrow_iq3xxs_m2(ne00, wrow, qcols[0], qcols[1], dst_cols[0], dst_cols[1], ir0);
                    }
                }

                current_chunk = (int) mrow_atomic_fetch_add(current_chunk_ctr, 1);
            }
        }
        return;
    }

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t Mfull = matrix_row_counts[cur_a];
        if (Mfull <= 0) {
            continue;
        }

        const char * src0_cur = src0_data + (size_t) cur_a * nb02;

        int64_t m0 = 0;
        while (m0 < Mfull) {
            const int M = (int) MIN((int64_t) MROW_MAX_M, Mfull - m0);

            for (int m = 0; m < M; ++m) {
                const struct mmid_row_mapping map = matrix_rows[cur_a * row_stride + (int) m0 + m];
                const int id = map.i1;
                const int64_t i11 = id % ne11;
                const int64_t i12 = map.i2;
                cols[m] = (const char *) wdata +
                    (use_linear ? (i11 + i12 * ne11) * row_size : (i11 * nb11 + i12 * nb12));
                dst_cols[m] = (float *) ((char *) dst->data + (id * nb1 + i12 * nb2));
                fcols[m] = (const float *) cols[m];
                qcols[m] = (const block_q8_K *) cols[m];
            }

            // M=1: stock GEMV. The VNNI M=1 reimplementation lost N=8 verify
            // (no decode-once, extra scale shuffle, worse than vec_dot_iq1_m).
            if (!mrow_kernel_accepts(src0_type, M) && vec_dot) {
                if (ith == 0) {
                    mrow_count_bin(g_mrow_ct.fallback, M);
                }
                for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                    mrow_stock(vec_dot, ne00, src0_cur + ir0 * nb01, M, cols, dst_cols, ir0);
                }
                m0 += M;
                continue;
            }

            if (ith == 0) {
                mrow_count_bin(g_mrow_ct.hit, M);
            }

            if (src0_type == GGML_TYPE_F32) {
                for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                    const float * wrow = (const float *) (src0_cur + ir0 * nb01);
                    mrow_f32(ne00, wrow, M, fcols, dst_cols, ir0);
                }
            } else if (src0_type == GGML_TYPE_IQ1_M) {
                for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                    const block_iq1_m * wrow = (const block_iq1_m *) (src0_cur + ir0 * nb01);
#if defined(__AVX2__)
                    if (ir0 + 1 < ir0_end) {
                        _mm_prefetch((const char *) (src0_cur + (ir0 + 1) * nb01), _MM_HINT_T0);
                    }
#endif
                    mrow_iq1m(ne00, wrow, M, qcols, dst_cols, ir0);
                }
            } else if (src0_type == GGML_TYPE_IQ2_XXS) {
#if defined(__AVX2__)
                const block_q8_K * iq2_cols[2] = { qcols[0], qcols[1] };
                float * iq2_dst[2] = { dst_cols[0], dst_cols[1] };
                for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                    const block_iq2_xxs * wrow = (const block_iq2_xxs *) (src0_cur + ir0 * nb01);
                    mrow_iq2xxs_avx2_m2(ne00, wrow, iq2_cols, iq2_dst, ir0);
                }
#endif
            } else {
                assert(src0_type == GGML_TYPE_IQ3_XXS && M == 2);
                for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                    const block_iq3_xxs * wrow = (const block_iq3_xxs *) (src0_cur + ir0 * nb01);
#if defined(__AVX2__)
                    if (ir0 + 1 < ir0_end) {
                        _mm_prefetch((const char *) (src0_cur + (ir0 + 1) * nb01), _MM_HINT_T0);
                    }
#endif
                    mrow_iq3xxs_m2(ne00, wrow, qcols[0], qcols[1], dst_cols[0], dst_cols[1], ir0);
                }
            }

            m0 += M;
        }
    }
}
