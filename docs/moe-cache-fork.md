# CUDA MoE cache fork

## Scope and credit

This is a public experimental fork, not an upstream submission. The MIT license and upstream notices remain in place. Model weights have their own licenses and are not included.

- Base: ggml-org/llama.cpp `eb25b7263e1604b4382295563f5a924002d6f87c`.
- Cache foundation: [leloch/llama.cpp at 708c18c](https://github.com/leloch/llama.cpp/tree/708c18c), discussed in [#24524](https://github.com/ggml-org/llama.cpp/pull/24524). The initial local cache was substantially a direct port, not an independently invented cache. CPU/GPU hit-miss dispatch, insert workers, LRU, gate/up fusion, redirect and the original bitmap persistence belong to that work.
- Qwen loader/graph: source port from [ggml-org/llama.cpp #27742](https://github.com/ggml-org/llama.cpp/pull/27742), including the hybrid index memory implementation. This is a pinned experimental snapshot, not every later change to that PR.
- Local extensions: 512-expert support; hot-only restore without a sequential whole-bank sweep; frequency scores with decay; per-pool capacity-aware v3 hotset selection; Windows timer/save compatibility; placement and launch tooling.
- AI-assisted development and release preparation. No upstream PR is being opened from this release.

## What it does

Attention and resident expert layers use ordinary llama.cpp placement. For eligible CPU-resident expert tensors, a cache hit executes on the CUDA device, CPU threads execute misses, and the results are joined before the next operation. Background workers copy admitted expert weights from local host memory into VRAM. Cold file-backed pages may still cause SSD reads. This is not an all-GPU engine, a distributed RAM pool or a LAN expert server.

The model still selects its own experts. This fork does not prune or substitute them. However, CPU and CUDA kernels can produce different floating-point results. **Do not assume bit-exact greedy equivalence to cache-off**, and validate quality on your workload. Multi-position verifier experiments exposed shape-dependent mismatches; their experimental adapters and MTP integration are excluded here.

Hotset files store ranking metadata, not weights or KV/chat state. V3 files describe up to 512 experts per block. Use a different file for each model/quant/placement. The launcher derives its filename from shard paths, sizes, timestamps and placement; this is a convenience fingerprint, not cryptographic model authentication. Do not load untrusted hotset files. An absent or incompatible file falls back to demand learning. Restore begins after tensor/pool discovery, not instantly at process startup.

## Build

Use a recent CMake, C++17 compiler and CUDA toolkit supported by your GPU. The extracted source is checked with MSVC 19.44 and CUDA 13.3 for SM120. Linux and other GPU architectures need independent validation.

```sh
git clone --branch qwen-moe-cache https://github.com/CurtisAccelerate/llama-moe-cache.git
cd llama-moe-cache
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release --target llama-server -j 3
```

On Windows, run this in a Visual Studio developer shell with CUDA installed. The executable is normally `build/bin/Release/llama-server.exe`; on single-config generators it is `build/bin/llama-server`. Upstream build instructions below the fork banner remain relevant. No release binaries are bundled.

## Windows preset

Provide the first shard of your own Qwen3.8-Flash-Next UD-Q4_K_XL GGUF. The other shards must be beside it.

```powershell
powershell -NoProfile -File scripts/start-moe-cache.ps1 -ModelPath 'D:\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
```

The preset uses GPU layers all, first 35 MoE layers on CPU, 3600 MiB demand cache, 2048 MiB reserve, 32K context, batch/ubatch 4096, 12 CPU threads, Flash Attention, F16 KV, one server slot and no speculation. It binds **127.0.0.1:8080**. It does not stop existing servers, change power limits, download weights or modify your model. On a detected RTX 5090 it refuses a power limit above 450 W; set the desired safe limit yourself before starting.

Options: `-ServerPath`, `-Port`, `-ContextSize`, `-Threads`, `-CpuMoeLayers`, `-CacheMiB`, `-BatchSize`, `-UBatchSize`, `-StateDirectory`, `-CacheOff`, `-DryRun`. CacheOff retains the same placement for a matched cache ablation. This preset is not a fit guarantee for other models or VRAM sizes. Reduce context/cache or move more experts to CPU if allocation fails. Keep enough OS RAM headroom; 32 GiB VRAM plus 64 GiB RAM does not make a 104 GiB model fully resident.

## Equivalent environment and flags

```sh
export GGML_CUDA_MOE_CACHE=1
export GGML_CUDA_MOE_CACHE_BUDGET_MB=3600
export GGML_CUDA_MOE_CACHE_RESERVE_MB=2048
export GGML_CUDA_MOE_CACHE_MAX_BLOCK=34
export GGML_CUDA_MOE_CACHE_MAX_BATCH=16
export GGML_CUDA_MOE_CACHE_HOTSET=1
export GGML_CUDA_MOE_CACHE_HOTSET_PATH=/your/private/state/model-placement-v3.bin
export GGML_CUDA_MOE_CACHE_HOTSET_SAVE_SEC=15
export GGML_CUDA_MOE_CACHE_PREFETCH=1
export GGML_CUDA_MOE_CACHE_HOT_ONLY=1
export GGML_CUDA_MOE_CACHE_STATS=64
./build/bin/llama-server -m /your/model-00001-of-00004.gguf -ngl all -ncmoe 35 --fit off -c 32768 -b 4096 -ub 4096 -t 12 -tb 12 -fa on -ctk f16 -ctv f16 -np 1 --spec-type none --host 127.0.0.1 --port 8080 --jinja
```

Create the state directory first. Cache is disabled without `GGML_CUDA_MOE_CACHE=1`. The inherited warmup/calibration and bail-out remain active: early decode can use CPU while the cache establishes a baseline, and it can disable itself if measured slower. Cache budget is in addition to model/KV/compute allocations. Small prompt chunks can use the cache; large prefill batches use the normal path. MAX_BLOCK is the inclusive last CPU-expert block, not the number of layers.

## Results and release limits

Historical local observations, not one controlled end-to-end scaling table:

| Configuration | Decode tok/s | Interpretation |
| --- | ---: | --- |
| Early static placement, 19 GPU layers | 7.45 | Initial baseline, not the best stock configuration |
| Dense path on GPU, all experts CPU, cache off | 12.26 | Different placement from the final preset |
| Large demand cache after warming | 22.39-22.73 | Cache development run |
| 35 CPU-MoE layers + 3600 MiB cache, warmed short prompts | 26.90-29.66 | Three prompts: prose, C64, Python; mean 28.23 |

Single RTX 5090 at 450 W, 64 GiB host RAM, Windows, Qwen3.8-Flash-Next UD-Q4_K_XL (about 103.7 GiB across four shards). The last sample generated 128 tokens per prompt. Short repeated prompts, OS file cache and expert residency matter. These numbers do not establish a general speedup against best-tuned upstream, and the release extraction has not been rebenchmarked on the full model. Build/launcher checks are distinct from inference quality validation.

The host CPU was an Intel Core Ultra 9 285K. Benchmark requests used identical input token IDs, temperature 0, top-k 1, seed 1234, a 128-token output budget and prompt-cache reuse disabled. Each short prompt ran twice; the table's final mean is the warmed second pass across Python, C64 and prose. The 32,768-token setting is allocated context capacity, not the length of these prompts. Separate 10,000-token prompt tests on the production control decoded at 10.79 and 18.15 tok/s on first/repeat passes; those are not represented by the 28.23 short-prompt mean. Storage caches were not cleared. Repeated outputs were not consistently identical, so these runs do not establish bit-exactness or a quality gain.

Excluded: research MTP, compact/multi-row verifier adapters, shape flags, S1/N1 experiments, remote expert banks, cache-fallback networking, private paths, raw prompts and benchmark archives. Stock upstream speculative code still exists, but the preset explicitly selects none. The 57-59 tok/s speculative peak was not a validated general-chat configuration and is not a release claim.

To reproduce: record model revision, build SHA, GPU/CPU/RAM, power limit, all flags, prompt/output tokens, TTFT, prefill and decode separately. Compare identical prompts and generation budgets with CacheOff, first use, immediate repeat, and diverse prompts. A restart is not an OS-cold run. Check outputs as well as speed; no full perplexity/quality gate is claimed here. The running development service was not replaced during publication.
