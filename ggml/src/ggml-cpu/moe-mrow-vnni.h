#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"

// One (expert-slot, token) assignment after grouping by expert id.
struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

#ifdef  __cplusplus
extern "C" {
#endif

// Default-off true M-row IQ1_M / IQ3_XXS / F32 expert-major kernel (LLAMA_MOE_MROW_VNNI=1).
// Off == current MUL_MAT_ID path (greedy exact). Does not extend the nrc=1
// grouped apply; that scheduler is only used to feed unique expert + row lists.
// M < min_m (2) uses stock vec_dot / stock MUL_MAT_ID — M=1 has no decode-once
// win and was the N=8 verify regression.
void ggml_moe_mrow_vnni_set(int enabled);
int  ggml_moe_mrow_vnni_get(void);
void ggml_moe_mrow_vnni_dynamic_set(int enabled);
int  ggml_moe_mrow_vnni_dynamic_get(void);

// IQ1_M, IQ3_XXS M=2, and F32. Other types fall back to stock.
int  ggml_moe_mrow_vnni_supported(enum ggml_type wtype);

// 1 if this translation unit emitted vpdpbusd (AVX2-VNNI), else 0 (maddubs).
int  ggml_moe_mrow_vnni_uses_vpdpbusd(void);

const char * ggml_moe_mrow_vnni_kernel_name(void);

int  ggml_moe_mrow_vnni_min_m(void);
void ggml_moe_mrow_vnni_counters_reset(void);
void ggml_moe_mrow_vnni_counters_get(struct ggml_moe_mrow_vnni_counters * out);
void ggml_moe_mrow_vnni_note_stock(const int64_t * counts, int n_as);

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
        void * atomic_current_chunk,
        size_t atomic_chunk_stride);

#ifdef  __cplusplus
}
#endif
