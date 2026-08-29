# GLM-5.3-Flash exact M-row CPU path

Branch: `glm53-mrow`

This branch is a focused, default-off extension of the experimental GLM5Next llama.cpp work at commit `8a8d0bcc4`. It contains the production subset we measured on a Core Ultra 9 285K, 64 GiB DDR5 host and RTX 5090 32 GiB. The RTX 4090 was not used for these results.

## What changed

GLM-5.3-Flash is larger than VRAM, so `-ncmoe 35` leaves part of routed-MoE execution on the CPU. Stock `MUL_MAT_ID` evaluates each selected activation row separately. When two rows select the same expert, this branch can decode that expert's `IQ2_XXS` or `IQ3_XXS` weights once and update both accumulators in one expert-major pass.

The implementation adds:

- exact M=2 decode-once kernels for `IQ2_XXS` and `IQ3_XXS` expert tensors;
- 256-bit AVX-VNNI (`vpdpbusd`) in the isolated kernel translation unit on supported Windows CPUs;
- dynamic output-tile scheduling so faster P-cores can steal work from slower E-cores;
- stock fallback for M=1, M>2, unsupported types and disabled mode;
- an optional exact-row KDA projection path for small speculative verifier shapes;
- bounded GLM recurrent rollback configuration for DFlash n=1;
- a synthetic exactness/microbenchmark test.

The IQ1_M target weights are unchanged. No experts are pruned, merged, substituted or re-quantized. The feature is off unless the environment flags below are set.

The weight-stationary idea is informed by ExLlamaV3's CPU-MoE work, but this is not a port: the kernels operate directly on GGML IQ2_XXS/IQ3_XXS blocks and integrate with llama.cpp's existing `MUL_MAT_ID` thread pool. GLM/Qwen architecture support remains attributable to its upstream authors.

## Measured result

Matched fully warm, immediate-repeat generation on two deterministic prompts, same IQ1_M target and DFlash n=1:

| Configuration | C64 tok/s | C tok/s | Mean |
| --- | ---: | ---: | ---: |
| Stock mixed-quant CPU path | 17.37 | 16.85 | 17.11 |
| M=2 IQ2_XXS + IQ3_XXS | 19.17 | 19.42 | 19.29 |

That is a **12.8% mean improvement**. Token arrays were identical:

- C64 SHA-256: `559a9070e4c6af6c81be9c83cba2ba3d15a9d908603d1879e5da857912e1bdb2`
- C SHA-256: `46dc1251d24f4c6f093f7b7c8cee4639b5777a716e4952c8b3e75d974d6f9dcd`

The selected executable measured 17.25 and 16.09 tok/s on 64-token immediate repeats. Longer cross-prompt trials ranged much more widely because Windows mmap page residency became dominant. This branch improves resident CPU expert arithmetic; it does not eliminate cold page faults. Do not present 19.29 tok/s as a cold-start guarantee.

DFlash n=1 was exact and cheap but approximately tied with no speculation once fully warm. It remains in the selected preset because removing the 2.23 GiB sidecar and moving one additional MoE layer to the GPU (`ncmoe34`) measured lower (17.94 tok/s mean). DFlash n>1 is not promoted here.

## Build (Windows, CUDA)

```powershell
git clone --branch glm53-mrow https://github.com/CurtisAccelerate/llama-moe-cache.git
cd llama-moe-cache
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release --target llama-server test-moe-mrow-vnni -j 12
$env:PATH = "$PWD\build\bin\Release;$env:PATH"
& .\build\bin\Release\test-moe-mrow-vnni.exe
```

Use CUDA architecture `89` for an RTX 4090, `120` for an RTX 5090, or `native` when building on the target machine.

## Selected serving preset

The convenience launcher exposes all paths as parameters:

```powershell
& .\scripts\start-glm53-mrow.ps1 `
  -ServerPath .\build\bin\Release\llama-server.exe `
  -ModelPath C:\models\GLM-5.3-Flash-UD-IQ1_M-00001-of-00003.gguf `
  -DraftPath C:\models\glm53-dflash2-bf16.gguf `
  -Port 8091
```

Equivalent environment and server settings:

```text
LLAMA_GLM53_EXACT_KDA_ROWS=1
LLAMA_MOE_MROW_VNNI=1
LLAMA_MOE_MROW_VNNI_DYNAMIC=1
LLAMA_MOE_MROW_VNNI_TILES_PER_THREAD=4
LLAMA_MOE_MROW_VNNI_MIN_M=2
LLAMA_GLM53_RS_ROLLBACK_CAP=1

-ngl all -ncmoe 35 --fit off
-c 32768 -b 512 -ub 512
-t 20 -tb 20 -np 1 -fa on
-ctk f16 -ctv f16
--spec-type draft-dflash -md <DFlash2 BF16 GGUF>
--spec-draft-device CUDA0 --spec-draft-ngl all
--spec-draft-n-max 1 --spec-draft-n-min 0 --spec-draft-p-min 0
--cache-ram 8192 --ctx-checkpoints 32
--host 127.0.0.1 --port 8091 --slots --metrics --jinja
```

Hardware placement is not universal. `-ncmoe 35`, 20 threads and the tile count were selected for the 285K/5090 machine. Sweep placement and threads on other systems, keep one server slot, and compare cold plus fully warm runs.

## Scope and limitations

- Experimental and Windows/AVX2-oriented; feature flags are default-off.
- Exactness was gated on the reported deterministic prompts and unit fixtures, not every model/quant/hardware combination.
- Cold performance is sensitive to available RAM and mmap residency.
- The branch does not include rejected page-prefetch, expanded IQ1 execution-image, streamed-miss, verifier-graph, pruning or distributed-target experiments.
- No weights or binaries are distributed here.
