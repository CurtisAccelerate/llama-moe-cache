#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fails = 0;

static void expect_true(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static void fill_f32(ggml_tensor * t, float seed) {
    const int64_t n = ggml_nelements(t);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < n; ++i) {
        p[i] = std::sin(seed + 0.017f * (float) i);
    }
}

static bool tensors_close(const float * a, const float * b, int64_t n, float eps, const char * tag) {
    for (int64_t i = 0; i < n; ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            fprintf(stderr, "mismatch %s i=%lld a=%g b=%g\n", tag, (long long) i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static void run_mul_mat_id_ab(
        enum ggml_type wtype,
        int n_expert, int n_used, int n_tokens, int n_embd, int n_out,
        const int32_t * expert_ids,
        float eps,
        bool exact = false) {
    const size_t mem = 64ull * 1024ull * 1024ull;
    struct ggml_init_params ip = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    expect_true(ctx != NULL, "ggml_init");

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, wtype, n_embd, n_out, n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_used, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    fill_f32(b, 1.1f);
    memcpy(ids->data, expert_ids, sizeof(int32_t) * (size_t) (n_used * n_tokens));

    if (wtype == GGML_TYPE_F32) {
        fill_f32(as, 0.3f);
    } else {
        expect_true(wtype == GGML_TYPE_IQ1_M || wtype == GGML_TYPE_IQ2_XXS || wtype == GGML_TYPE_IQ3_XXS, "quant fixture");
        std::vector<float> src((size_t) n_embd * (size_t) n_out * (size_t) n_expert);
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = std::sin(0.3f + 0.017f * (float) i);
        }
        std::vector<float> importance;
        const float * importance_ptr = nullptr;
        if (wtype == GGML_TYPE_IQ2_XXS) {
            importance.assign(src.size(), 1.0f);
            importance_ptr = importance.data();
        }
        ggml_quantize_init(wtype);
        const size_t wrote = ggml_quantize_chunk(
            wtype, src.data(), as->data,
            0, (int64_t) n_out * n_expert, n_embd, importance_ptr);
        expect_true(wrote > 0, "quantize fixture wrote");
    }

    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_moe_mrow_vnni_set(0);
    expect_true(ggml_moe_mrow_vnni_get() == 0, "mrow off");
    expect_true(ggml_graph_compute_with_ctx(ctx, gf, 4) == GGML_STATUS_SUCCESS, "compute off");

    const int64_t n = ggml_nelements(out);
    std::vector<float> off((size_t) n);
    memcpy(off.data(), out->data, sizeof(float) * (size_t) n);

    ggml_moe_mrow_vnni_set(1);
    expect_true(ggml_moe_mrow_vnni_get() == 1, "mrow on");
    ggml_moe_mrow_vnni_dynamic_set(0);
    expect_true(ggml_moe_mrow_vnni_dynamic_get() == 0, "mrow dynamic off");
    expect_true(ggml_graph_compute_with_ctx(ctx, gf, 4) == GGML_STATUS_SUCCESS, "compute on");
    expect_true(tensors_close(off.data(), (const float *) out->data, n, eps, "off==on"), "off==on greedy");
    if (exact) {
        expect_true(memcmp(off.data(), out->data, sizeof(float) * (size_t) n) == 0, "off==on bit exact");
    }

    ggml_moe_mrow_vnni_dynamic_set(1);
    expect_true(ggml_moe_mrow_vnni_dynamic_get() == 1, "mrow dynamic on");
    expect_true(ggml_graph_compute_with_ctx(ctx, gf, 4) == GGML_STATUS_SUCCESS, "compute dynamic");
    expect_true(tensors_close(off.data(), (const float *) out->data, n, eps, "off==dynamic"), "off==dynamic greedy");
    if (exact) {
        expect_true(memcmp(off.data(), out->data, sizeof(float) * (size_t) n) == 0, "off==dynamic bit exact");
    }

    ggml_moe_mrow_vnni_dynamic_set(0);
    ggml_moe_mrow_vnni_set(0);
    ggml_free(ctx);
}

static void fill_same_expert(int32_t * ids, int n_used, int n_tokens, int expert) {
    for (int t = 0; t < n_tokens; ++t) {
        for (int k = 0; k < n_used; ++k) {
            ids[t * n_used + k] = expert + k;
        }
    }
}

static void test_f32_m1_to_m8(void) {
    for (int M = 1; M <= 8; ++M) {
        int32_t ids[64];
        fill_same_expert(ids, /*n_used*/ 2, M, /*expert*/ 0);
        char tag[64];
        snprintf(tag, sizeof(tag), "f32 M=%d", M);
        run_mul_mat_id_ab(GGML_TYPE_F32, /*n_expert*/ 8, 2, M, 64, 16, ids, 1e-4f);
        if (g_fails) {
            fprintf(stderr, "failed at %s\n", tag);
            return;
        }
    }
}

static void test_iq1m_m1_to_m8(void) {
    // QK_K = 256. Tiny synthesized IQ1_M expert, no model file.
    const int n_embd = 256;
    const int n_out = 8;
    const int n_expert = 4;
    for (int M = 1; M <= 8; ++M) {
        int32_t ids[64];
        fill_same_expert(ids, /*n_used*/ 1, M, /*expert*/ 0);
        run_mul_mat_id_ab(GGML_TYPE_IQ1_M, n_expert, 1, M, n_embd, n_out, ids, 1e-3f);
        if (g_fails) {
            fprintf(stderr, "failed at iq1m M=%d\n", M);
            return;
        }
    }

    // Mixed reuse: 3 tokens share expert 0, 2 tokens share expert 1 (M=3 and M=2 tiles).
    int32_t mix[5] = { 0, 0, 0, 1, 1 };
    run_mul_mat_id_ab(GGML_TYPE_IQ1_M, n_expert, 1, 5, n_embd, n_out, mix, 1e-3f);
}

static void test_dispatch_supported(void) {
    expect_true(ggml_moe_mrow_vnni_supported(GGML_TYPE_IQ1_M) == 1, "iq1_m supported");
    expect_true(ggml_moe_mrow_vnni_supported(GGML_TYPE_IQ2_XXS) == 1, "iq2_xxs M=2 supported");
    expect_true(ggml_moe_mrow_vnni_supported(GGML_TYPE_IQ3_XXS) == 1, "iq3_xxs M=2 supported");
    expect_true(ggml_moe_mrow_vnni_supported(GGML_TYPE_F32) == 1, "f32 supported");
    expect_true(ggml_moe_mrow_vnni_supported(GGML_TYPE_Q4_K) == 0, "q4_k not this kernel");
    expect_true(ggml_moe_mrow_vnni_min_m() == 2, "break-even M=2");
}

static void test_iq2xxs_m1_m2_and_fallback(void) {
    const int n_embd = 256;
    const int n_out = 8;
    const int n_expert = 4;

    for (int M = 1; M <= 3; ++M) {
        int32_t ids[3];
        fill_same_expert(ids, 1, M, 0);
        run_mul_mat_id_ab(GGML_TYPE_IQ2_XXS, n_expert, 1, M, n_embd, n_out, ids, 0.0f, true);
        if (g_fails) {
            fprintf(stderr, "failed at iq2_xxs M=%d\n", M);
            return;
        }
    }

    // One M=2 expert takes decode-once; one M=3 expert remains stock row-wise.
    const int32_t mixed[5] = { 0, 0, 1, 1, 1 };
    run_mul_mat_id_ab(GGML_TYPE_IQ2_XXS, n_expert, 1, 5, n_embd, n_out, mixed, 0.0f, true);

    ggml_moe_mrow_vnni_counters_reset();
    const int32_t ids2[2] = { 0, 0 };
    run_mul_mat_id_ab(GGML_TYPE_IQ2_XXS, n_expert, 1, 2, n_embd, n_out, ids2, 0.0f, true);
    struct ggml_moe_mrow_vnni_counters c;
    ggml_moe_mrow_vnni_counters_get(&c);
    expect_true(c.hit[2] > 0, "iq2_xxs M=2 takes decode-once kernel");
}

static void test_iq3xxs_m1_m2_and_fallback(void) {
    const int n_embd = 256;
    const int n_out = 32;
    const int n_expert = 4;
    const int32_t ids1[1] = { 0 };
    const int32_t ids2[2] = { 0, 0 };
    const int32_t ids3[3] = { 0, 0, 0 };

    run_mul_mat_id_ab(GGML_TYPE_IQ3_XXS, n_expert, 1, 1, n_embd, n_out, ids1, 0.0f, true);

    ggml_moe_mrow_vnni_counters_reset();
    run_mul_mat_id_ab(GGML_TYPE_IQ3_XXS, n_expert, 1, 2, n_embd, n_out, ids2, 0.0f, true);
    struct ggml_moe_mrow_vnni_counters c2;
    ggml_moe_mrow_vnni_counters_get(&c2);
    expect_true(c2.hit[2] > 0, "iq3_xxs M=2 takes decode-once kernel");

    ggml_moe_mrow_vnni_counters_reset();
    run_mul_mat_id_ab(GGML_TYPE_IQ3_XXS, n_expert, 1, 3, n_embd, n_out, ids3, 0.0f, true);
    struct ggml_moe_mrow_vnni_counters c3;
    ggml_moe_mrow_vnni_counters_get(&c3);
    expect_true(c3.hit[3] == 0, "iq3_xxs M=3 stays on stock path");
    expect_true(c3.fallback[3] > 0, "iq3_xxs M=3 fallback recorded");
}

static void bench_iq2xxs_m2(void) {
    if (!std::getenv("LLAMA_MOE_MROW_BENCH")) {
        return;
    }

    // One GLM-sized gate/up expert. This is an optimistic ceiling for the
    // affected M=2 assignments, not an end-to-end token-speed claim.
    const int n_embd = 2048;
    const int n_out = 3456;
    const int n_expert = 4;
    const int n_tokens = 8;
    const size_t mem = 256ull * 1024ull * 1024ull;
    struct ggml_init_params ip = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    expect_true(ctx != NULL, "bench ggml_init");

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XXS, n_embd, n_out, n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_tokens);
    fill_f32(b, 0.7f);
    const int32_t expert_ids[n_tokens] = { 0, 0, 1, 1, 2, 2, 3, 3 };
    memcpy(ids->data, expert_ids, sizeof(expert_ids));

    std::vector<float> src((size_t) n_embd * n_out * n_expert);
    std::vector<float> importance(src.size(), 1.0f);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = std::sin(0.19f + 0.00017f * (float) i);
    }
    ggml_quantize_init(GGML_TYPE_IQ2_XXS);
    expect_true(ggml_quantize_chunk(
        GGML_TYPE_IQ2_XXS, src.data(), as->data, 0, n_out * n_expert, n_embd, importance.data()) > 0,
        "bench iq2 quantize");

    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const int threads = 20;
    const int warm = 4;
    const int iters = 100;
    const int rounds = 7;

    auto run = [&](bool enabled, bool dynamic) {
        ggml_moe_mrow_vnni_set(enabled ? 1 : 0);
        ggml_moe_mrow_vnni_dynamic_set(dynamic ? 1 : 0);
        for (int i = 0; i < warm; ++i) {
            expect_true(ggml_graph_compute_with_ctx(ctx, gf, threads) == GGML_STATUS_SUCCESS, "bench warm");
        }
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < iters; ++i) {
            expect_true(ggml_graph_compute_with_ctx(ctx, gf, threads) == GGML_STATUS_SUCCESS, "bench compute");
        }
        return (double) (ggml_time_us() - t0) / (double) iters;
    };

    double off_us[rounds];
    double dynamic_us[rounds];
    for (int r = 0; r < rounds; ++r) {
        if ((r & 1) == 0) {
            off_us[r] = run(false, false);
            dynamic_us[r] = run(true, true);
        } else {
            dynamic_us[r] = run(true, true);
            off_us[r] = run(false, false);
        }
    }
    auto median = [](double * v, int n) {
        for (int i = 1; i < n; ++i) {
            const double x = v[i];
            int j = i;
            while (j > 0 && v[j - 1] > x) {
                v[j] = v[j - 1];
                --j;
            }
            v[j] = x;
        }
        return v[n / 2];
    };
    const double off_median = median(off_us, rounds);
    const double dynamic_median = median(dynamic_us, rounds);

    run(false, false);
    std::vector<float> off((size_t) ggml_nelements(out));
    memcpy(off.data(), out->data, off.size() * sizeof(float));
    run(true, true);
    expect_true(memcmp(off.data(), out->data, off.size() * sizeof(float)) == 0, "bench bit exact");
    printf("iq2_xxs M=2 GLM-shape 4-expert microbench (7x100 alternating): off_p50=%.1f us dynamic_p50=%.1f us speedup=%.3fx\n",
           off_median, dynamic_median, off_median / dynamic_median);

    ggml_moe_mrow_vnni_dynamic_set(0);
    ggml_moe_mrow_vnni_set(0);
    ggml_free(ctx);
}

static void test_m1_bypasses_kernel(void) {
    // M=1 must stay on stock GEMV even with the flag on (N=8 verify is ~75% M=1).
    ggml_moe_mrow_vnni_counters_reset();
    int32_t ids1[1] = { 0 };
    run_mul_mat_id_ab(GGML_TYPE_IQ1_M, /*n_expert*/ 4, 1, 1, 256, 8, ids1, 1e-3f);

    struct ggml_moe_mrow_vnni_counters c1;
    ggml_moe_mrow_vnni_counters_get(&c1);
    expect_true(c1.hit[1] == 0, "M=1 does not take VNNI kernel");
    expect_true(c1.fallback[1] > 0, "M=1 counted as stock fallback");
    expect_true(c1.ops_stock > 0, "M=1-only op uses stock MUL_MAT_ID");
    expect_true(c1.ops_kernel == 0, "M=1-only op does not enter apply");

    ggml_moe_mrow_vnni_counters_reset();
    int32_t ids2[2] = { 0, 0 };
    run_mul_mat_id_ab(GGML_TYPE_IQ1_M, /*n_expert*/ 4, 1, 2, 256, 8, ids2, 1e-3f);

    struct ggml_moe_mrow_vnni_counters c2;
    ggml_moe_mrow_vnni_counters_get(&c2);
    expect_true(c2.hit[2] > 0, "M=2 takes decode-once kernel");
    expect_true(c2.ops_kernel > 0, "M=2 enters apply");
}

int main(void) {
    ggml_time_init();
    ggml_cpu_init();

    printf("ggml_moe_mrow_vnni default=%d kernel=%s vpdpbusd=%d\n",
           ggml_moe_mrow_vnni_get(),
           ggml_moe_mrow_vnni_kernel_name(),
           ggml_moe_mrow_vnni_uses_vpdpbusd());
    printf("cpu: avx2=%d avx_vnni=%d avx512=%d avx512_vnni=%d\n",
           ggml_cpu_has_avx2(),
           ggml_cpu_has_avx_vnni(),
           ggml_cpu_has_avx512(),
           ggml_cpu_has_avx512_vnni());

    expect_true(ggml_moe_mrow_vnni_get() == 0, "default off");
    test_dispatch_supported();
    test_f32_m1_to_m8();
    test_iq1m_m1_to_m8();
    test_iq2xxs_m1_m2_and_fallback();
    test_iq3xxs_m1_m2_and_fallback();
    test_m1_bypasses_kernel();
    bench_iq2xxs_m2();

    if (g_fails) {
        fprintf(stderr, "%d checks failed\n", g_fails);
        return 1;
    }
    printf("test-moe-mrow-vnni: ok\n");
    return 0;
}
