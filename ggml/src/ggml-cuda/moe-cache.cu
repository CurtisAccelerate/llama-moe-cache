// MoE expert cache - dynamic VRAM cache for CPU-resident MoE expert weights.
//
// Cache core ported from leloch/llama.cpp (708c18c); see docs/moe-cache-fork.md.
// Design:
//  - Integration lives inside the CPU mul_mat_id kernel (see ggml-cpu.c):
//    thread 0 plans hits/misses and dispatches cached rows to the GPU in ONE
//    batched kernel launch while the remaining threadpool threads compute the
//    miss rows. Results are collected into dst before the node ends, so
//    correctness holds under any split topology and no shared tensors are
//    ever mutated.
//  - The cache fills from demand misses via dedicated insert worker threads.
//    Prompt batches stay on the stock path unless MAX_BATCH is set to 8, 16,
//    or 32.
//  - Slots live in per-(expert_size, type) pools whose slot stride equals the
//    source tensor's nb[2] exactly, so the batched mmvq kernel can index the
//    pool like a regular expert tensor (strides are in block units).
//  - Eviction: LRU per pool. Ranked hotsets seed the cache after restart.
//
// Keys are FNV-1a hashes of the weight tensor's name (stable across contexts
// and mmap remaps) mixed with the expert id.

#include "moe-cache.cuh"
#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "ggml-backend-impl.h"
#include "../ggml-backend-moe-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

#define MOE_CACHE_MAX_DEV    8
#define MOE_CACHE_MAX_POOLS  8
#define MOE_CACHE_MAX_HOT_EXPERTS 512
#define MOE_CACHE_HOT_WORDS (MOE_CACHE_MAX_HOT_EXPERTS / 64)
#define MOE_CACHE_MAX_BATCH_TOKENS 32
#define MOE_CACHE_MAX_ROUTED_ROWS  512
#define MOE_CACHE_HOTSET_VERSION   3

// key-space tag for paired (gate, up) entries: keeps them disjoint from the
// name-hash-keyed entries of unpaired pools that share the same shape
#define MOE_CACHE_PAIR_KEY_TAG 0xEC3000000000000ULL
#define MOE_CACHE_LOG(...)   fprintf(stderr, __VA_ARGS__)

struct moe_cache_slot {
    uint64_t key;
    int      prev;
    int      next;
    bool     valid;     // contents complete, lookups may hit
    bool     queued;    // insert copy queued or in flight
};

struct moe_cache_pool {
    size_t expert_size = 0;   // == slot stride == source tensor nb[2]
    int    wtype       = -1;
    char * slab        = nullptr;   // up weights (mmv x operand)
    char * slab2       = nullptr;   // gate weights (fusion operand), paired pools only
    bool   paired      = false;     // one entry covers the (gate, up) pair of an expert
    int    n_slots     = 0;
    int    n_used      = 0;

    std::vector<moe_cache_slot> slots;
    std::unordered_map<uint64_t, int> map;
    int lru_head = -1;
    int lru_tail = -1;
};

struct moe_cache_device {
    moe_cache_pool pools[MOE_CACHE_MAX_POOLS];
    int      n_pools = 0;
    bool     dead    = false;   // CUDA failure or trim: cache permanently off here

    cudaStream_t compute_stream = nullptr;

    // GPU-resident dst handoff machinery. The pinned image and its event are
    // double-buffered by parity: node N+1's D2H must not overwrite the image
    // node N's consumer-stream H2D still reads.
    char    * h_redir = nullptr; size_t h_redir_half = 0;    // pinned image x2
    cudaEvent_t redir_evt[2] = {};
    int       redir_par = 0;

    // staging for one node's batched dispatch
    int32_t * h_ids = nullptr;  int32_t * d_ids = nullptr;  size_t ids_cap = 0;   // count
    float   * h_act = nullptr;  float   * d_act = nullptr;  size_t act_cap = 0;   // bytes
    void    * d_act_q8 = nullptr;                           size_t act_q8_cap = 0;
    float   * d_out = nullptr;                              size_t d_out_cap = 0;
    float   * h_out = nullptr;                              size_t h_out_cap = 0; // pinned
    int       out_rows = 0;

    // activation reuse: gate and up of one layer share the same input row
    const float * q8_act_ptr = nullptr;   // host act already quantized on device
    int           q8_act_blk = -1;

    // fused gate+up+GLU: one layer in flight per device. Filled at the gate
    // node, matched+scattered by the CPU GLU kernel via glu_hits().
    struct {
        bool         active = false;
        const void * gate_dst = nullptr;  // gate MMID dst base (== glu src0)
        const void * up_dst   = nullptr;  // up MMID dst base   (== glu src1)
        unsigned long long mask = 0;      // dst row bits 0..63 (legacy glu_hits)
        uint64_t     mask_w[8] = {};      // bounded routed-row mask
        int          rows[512];           // dst row index per d_out row
        int          n = 0;
        int64_t      n_out = 0;
        bool         scattered = false;
        long long    serial = 0;          // gallocr reuses dst pointers every
                                          // layer: the GLU hook must match the
                                          // NEWEST entry, not any stale one
    } fused;
    long long fused_layers = 0;

    // stats
    long long hits = 0, misses = 0, inserts = 0, evictions = 0;
    long long insert_skips = 0, queued_misses = 0;
    // miss decomposition (counters only, no behavior change)
    long long pool_hits[MOE_CACHE_MAX_POOLS] = {}, pool_miss[MOE_CACHE_MAX_POOLS] = {};
    long long miss_compulsory = 0, miss_capacity = 0, miss_admission = 0;
    long long skip_throttle = 0, skip_budget = 0, skip_qfull = 0, skip_lrubusy = 0;
    std::unordered_set<uint64_t> ever_seen, ever_inserted;
    // per-phase wall time (thread-0 serial cost), microseconds
    long long t_plan_us = 0, t_disp_us = 0, t_coll_us = 0, n_nodes = 0;
    long long t_miss_wall_us = 0;
    long long prompt_nodes = 0, prompt_hits = 0, prompt_misses = 0;
    long long decode_nodes = 0, decode_hits = 0, decode_misses = 0;
    long long dispatch_h2d_bytes = 0;
    long long insert_stage_us = 0, insert_h2d_us = 0, insert_h2d_bytes = 0;
    long long redirect_claims = 0, redirect_misses_up = 0;
};

struct moe_cache_job {
    int        dev;
    int        pool;
    uint64_t   key;
    int        slot_idx;
    const void * src;        // up weights (paired) or sole tensor
    const void * src_gate;   // gate weights (paired pools), else NULL
    size_t     bytes;
    int        blk = -1;     // routing-bias bitmap maintenance
    int        eid = -1;
};

struct moe_cache_global {
    bool   enabled  = false;
    int    n_dev    = 0;
    size_t budget_mb = 0;        // 0 = auto (free VRAM at init minus reserve)
    size_t reserve_mb = 3072;    // VRAM left untouched per device: the CUDA pool
                                 // grows lazily AFTER our init; stealing it
                                 // crashes the model mid-decode (measured)
    int    inserts_per_plan = 8; // max inserts enqueued per plan() call
    int    throttle_mod     = 8; // at capacity admit 1-in-N misses (GGML_CUDA_MOE_CACHE_THROTTLE)
    int    queue_max        = 512;
    int    n_workers        = 4;
    size_t min_expert_bytes = 1u << 20; // skip models whose experts are too small
                                        // to amortize per-node dispatch (measured:
                                        // 0.45MB experts lose, 3MB+ win big)
    int    max_batch        = 1; // GGML_CUDA_MOE_CACHE_MAX_BATCH; 8/16/32 enables prompt chunks
    int    max_block        = 42; // GGML_CUDA_MOE_CACHE_MAX_BLOCK; inclusive, -1 = no cap
    int    stats_every      = 0; // log every N collect() calls (0 = off)
    int    io_stats_every   = 0; // GGML_CUDA_MOE_CACHE_IO_STATS; log every N engaged nodes

    moe_cache_device dev[MOE_CACHE_MAX_DEV];

    // prefetch backfill cursor (guarded by mu): walks (blk, eid) space and
    // enqueues nothing - workers pull directly when the demand queue is empty
    struct {
        bool enabled = true;       // GGML_CUDA_MOE_CACHE_PREFETCH=0 to disable
        bool hot_only = false;     // restore only the persisted working set; never sweep cold experts
        bool active  = false;      // pools exist somewhere
        int  phase   = 0;          // 0 = hot-set prior, 1 = sequential sweep
        int  blk     = 0;          // next block to backfill
        int  eid     = 0;          // next expert within the block
        bool done    = false;
        long long hot_jobs = 0;
        bool reported = false;
    } backfill;

    // insert queue + workers
    std::mutex              mu;  // guards pools/queue of all devices
    std::condition_variable cv;
    std::condition_variable cv_idle;          // signaled when a worker finishes a job
    std::deque<moe_cache_job>     queue;
    const void *            inflight_src[16] = {};   // per-worker current source
    size_t                  inflight_len[16] = {};
    bool                    workers_started = false;

    // current node context (begin..collect happen on one thread)
    uint64_t     cur_key_base = 0;
    const void * cur_host_base = nullptr;
    size_t       cur_expert_size = 0;
    int64_t      cur_n_expert = 0;
    int          cur_pool = -1;
    int          cur_blk  = -1;
    int          cur_role = -1;   // 0=gate 1=up 2=down -1=other
    int64_t      cur_n_tokens = 1;
    int32_t      cur_slot_idx[MOE_CACHE_MAX_ROUTED_ROWS] = {};
    int          cur_n_ids = 0;
    std::unordered_map<const void *, int> glu_learn;  // gate MMID dst base -> blk
    bool         reuse    = true; // GGML_CUDA_MOE_CACHE_REUSE=0 to disable act-quant reuse

    // fused gate+up+GLU path. Pair-fused dispatch engages per layer only after
    // the CPU GLU hook was OBSERVED matching that layer's gate/up dst pair
    // (learned on the first decode tokens) - a graph without the hook firing
    // would otherwise compute silu(garbage)*garbage for the skipped rows.
    // per-blk facts for prefetch backfill (written by begin, read by workers
    // under mu at job-pull time; plain arrays, blk-indexed)
    int8_t       blk_pair_pool[1024];    // paired-pool index on the owning dev, -1 unknown
    int8_t       blk_down_pool[1024];    // down-pool index, -1 unknown
    int          blk_n_expert[1024] = {};
    uint64_t     blk_down_kb[1024] = {}; // down key base (name-hash ^ ptr-hash)
    const void * blk_down_base[1024] = {};

    // Ranked hot-set persistence. Scores are decayed access counts for one
    // (block, expert, gate/up pair or down projection). Restore selection is
    // capped by the actual slot count of each pool.
    bool     hotset_enabled = false;     // GGML_CUDA_MOE_CACHE_HOTSET=1 to enable
    uint64_t hot_pair[1024][MOE_CACHE_HOT_WORDS] = {}; // ranked restore selection
    uint64_t hot_down[1024][MOE_CACHE_HOT_WORDS] = {};
    uint16_t hot_score_pair[1024][MOE_CACHE_MAX_HOT_EXPERTS] = {};
    uint16_t hot_score_down[1024][MOE_CACHE_MAX_HOT_EXPERTS] = {};
    uint64_t hot_score_plans = 0;
    uint64_t hot_score_decay_plans = 65536;
    bool     hot_loaded = false;
    bool     hot_restore_ready = false;
    int      hot_restore_last_blk = -1;
    int      hot_restore_wraps = 0;
    std::vector<uint32_t> hot_restore_order;
    size_t   hot_restore_pos = 0;
    int64_t  hotset_last_save = 0;
    int      hotset_save_sec = 90;
    char     hotset_path[512] = {};
    uint64_t resident_pair[1024][MOE_CACHE_HOT_WORDS] = {};
    uint64_t resident_down[1024][MOE_CACHE_HOT_WORDS] = {};

    bool fuse = true;                    // GGML_CUDA_MOE_CACHE_FUSE=0 to disable. Stale-entry
                                         // hazard (MOE_CACHE_READINESS.md B1) closed by
                                         // gate-begin epoch invalidation + up-node
                                         // mask reuse.
    long long fuse_serial = 0;
    bool safe_fuse_blk[1024] = {};
    const void * role_base[2][1024] = {}; // [role][blk] -> tensor host base
    // last gate/up MMID dst bases per blk (for GLU-hook learning)
    const void * learn_gate_dst[1024] = {};
    const void * learn_up_dst[1024] = {};

    // GPU-resident dst handoff: host dst base -> offered GPU copy. Entries are
    // one-shot: offered before the CPU split, optionally populated by collect,
    // resolved (claimed or dropped) at the consumer's input-copy site.
    struct redirect_entry {
        size_t  nb1 = 0;
        int64_t n_rows = 0;
        void *  gpu_ptr = nullptr;
        int     dev = -1;
        uint64_t hit_mask = 0;     // rows relayed by collect
        bool    populated = false; // collect engaged
        int     par = 0;           // pinned-image parity used by collect
    };
    std::unordered_map<const void *, redirect_entry> redirect;
    bool redirect_on = true;       // GGML_CUDA_MOE_CACHE_REDIRECT=0 to disable

    long long collect_calls = 0;

    // ---- baseline-sampled bail-out ----
    // Phase A (after pools build): eligible nodes run pure-CPU while their wall
    // time builds the baseline EWMA (begin returns -3 so the kernel reports the
    // sample). Phase B: cache engages; node walls feed the engaged EWMA. Once
    // enough engaged samples accumulate, sustained engaged > baseline * 1.05
    // disables the cache and frees its VRAM: the placement bet failed on this
    // workload and the CPU path is the better config.
    struct {
        long long eligible_seen = 0;       // counts begins after pools exist
        double    base_ewma = 0.0;         // pure-CPU node wall, us
        double    on_ewma   = 0.0;         // cache-engaged node wall, us
        long long base_n = 0, on_n = 0;
        int       strikes = 0;
        bool      tripped = false;
    } bail;
    static constexpr long long BAIL_WARM   = 500;   // ignored (first-touch effects)
    static constexpr long long BAIL_SAMPLE = 2750;  // baseline window end
};

// intentionally leaked: detached worker threads reference this state through
// process exit; running its destructor would tear a condition variable out
// from under a waiting thread (observed as a hang in atexit).
static moe_cache_global & g = *new moe_cache_global();

static bool moe_cache_decode_like(void) {
    if (g.cur_n_tokens == 1) {
        return true;
    }
    return false;
}

static int moe_cache_fuse_row_cap(void) {
    return 64;
}

// GGML_CUDA_MOE_CACHE_DEBUG=1: trace the first calls of each API to locate stalls
static int g_dbg = -1;
static long long g_dbg_n = 0;
#define MOE_CACHE_DBG(...) do { \
        if (g_dbg < 0) { const char * _e = getenv("GGML_CUDA_MOE_CACHE_DEBUG"); g_dbg = _e ? atoi(_e) : 0; } \
        if (g_dbg > 0 && g_dbg_n++ < (g_dbg >= 10 ? (long long)g_dbg : 400)) { MOE_CACHE_LOG(__VA_ARGS__); fflush(stderr); } \
    } while (0)

// checked CUDA call: on failure the device's cache is disabled (begin() will
// refuse) and the caller takes its degraded-but-finite path - a transient CUDA
// error must never abort the host process (the stock CPU path still works)
static bool moe_cache_ok(int di, cudaError_t e, const char * what) {
    if (e == cudaSuccess) return true;
    cudaGetLastError();
    if (di >= 0 && di < MOE_CACHE_MAX_DEV && !g.dev[di].dead) {
        g.dev[di].dead = true;
        MOE_CACHE_LOG("[moe-cache] dev=%d DISABLED: %s failed: %s (CPU path takes over)\n",
                di, what, cudaGetErrorString(e));
    }
    return false;
}

static uint64_t moe_cache_fnv1a(const char * s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 0x100000001b3ULL;
    }
    return h;
}

static inline uint64_t moe_cache_ptr_hash(const void * p) {
    uint64_t v = (uint64_t)(uintptr_t)p;
    v *= 0xFF51AFD7ED558CCDULL;
    v ^= v >> 33;
    return v;
}

static inline uint64_t moe_cache_key(uint64_t name_hash, int eid) {
    return name_hash ^ ((uint64_t)(uint32_t)eid * 0x9E3779B97F4A7C15ULL);
}

// ---- LRU helpers (caller holds g.mu) ---------------------------------------

static void moe_cache_lru_remove(moe_cache_pool & p, int idx) {
    moe_cache_slot & s = p.slots[idx];
    if (s.prev >= 0) p.slots[s.prev].next = s.next; else p.lru_head = s.next;
    if (s.next >= 0) p.slots[s.next].prev = s.prev; else p.lru_tail = s.prev;
    s.prev = s.next = -1;
}

static void moe_cache_lru_push_back(moe_cache_pool & p, int idx) {
    moe_cache_slot & s = p.slots[idx];
    s.prev = p.lru_tail;
    s.next = -1;
    if (p.lru_tail >= 0) p.slots[p.lru_tail].next = idx; else p.lru_head = idx;
    p.lru_tail = idx;
}

static inline bool moe_cache_slot_evict_blocked(const moe_cache_slot & s) {
    return s.queued;
}


struct moe_cache_hotset_header {
    char     magic[8];
    uint32_t version;
    uint32_t blocks;
    uint32_t experts;
    uint32_t score_bytes;
};

static_assert(sizeof(moe_cache_hotset_header) == 24, "unexpected hot-set header layout");

struct moe_cache_io_sample {
    uint64_t read_bytes = 0;
    uint64_t read_ops = 0;
    uint64_t page_faults = 0;
};

static moe_cache_io_sample moe_cache_sample_io() {
    moe_cache_io_sample s;
#ifdef _WIN32
    IO_COUNTERS io = {};
    if (GetProcessIoCounters(GetCurrentProcess(), &io)) {
        s.read_bytes = io.ReadTransferCount;
        s.read_ops = io.ReadOperationCount;
    }
    using get_memory_info_fn = BOOL (WINAPI *)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
    static get_memory_info_fn get_memory_info = (get_memory_info_fn) GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32GetProcessMemoryInfo");
    if (get_memory_info) {
        PROCESS_MEMORY_COUNTERS pmc = {};
        pmc.cb = sizeof(pmc);
        if (get_memory_info(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            s.page_faults = pmc.PageFaultCount;
        }
    }
#endif
    return s;
}

static moe_cache_io_sample g_io_base;
static bool g_io_started = false;

static void moe_cache_score_access(int blk, int eid, bool pair) {
    if (!g.hotset_enabled || blk < 0 || blk >= 1024 || eid < 0 || eid >= MOE_CACHE_MAX_HOT_EXPERTS) return;
    uint16_t & score = pair ? g.hot_score_pair[blk][eid] : g.hot_score_down[blk][eid];
    score = score > UINT16_MAX - 32 ? UINT16_MAX : (uint16_t)(score + 32);
}

static void moe_cache_score_decay() {
    if (!g.hotset_enabled) return;
    if (++g.hot_score_plans < g.hot_score_decay_plans) return;
    g.hot_score_plans = 0;
    for (int blk = 0; blk < 1024; blk++) {
        for (int eid = 0; eid < MOE_CACHE_MAX_HOT_EXPERTS; eid++) {
            g.hot_score_pair[blk][eid] = (uint16_t)((g.hot_score_pair[blk][eid] + 1) >> 1);
            g.hot_score_down[blk][eid] = (uint16_t)((g.hot_score_down[blk][eid] + 1) >> 1);
        }
    }
}

struct moe_cache_hot_candidate {
    uint16_t score;
    uint16_t blk;
    uint16_t eid;
    bool     pair;
};

static bool moe_cache_hot_sources_ready() {
    for (int blk = 0; blk < 1024; blk++) {
        for (int eid = 0; eid < MOE_CACHE_MAX_HOT_EXPERTS; eid++) {
            if (g.hot_score_pair[blk][eid] &&
                (g.blk_pair_pool[blk] < 0 || !g.role_base[0][blk] || !g.role_base[1][blk])) return false;
            if (g.hot_score_down[blk][eid] &&
                (g.blk_down_pool[blk] < 0 || !g.blk_down_base[blk])) return false;
        }
    }
    return true;
}

static void moe_cache_build_ranked_restore(bool force) {
    if (!g.hot_loaded || g.hot_restore_ready || (!force && !moe_cache_hot_sources_ready())) return;

    std::vector<moe_cache_hot_candidate> ranked[MOE_CACHE_MAX_DEV][MOE_CACHE_MAX_POOLS];
    for (int blk = 0; blk < 1024; blk++) {
        const int di = blk % g.n_dev;
        const int n_exp = std::min(g.blk_n_expert[blk], MOE_CACHE_MAX_HOT_EXPERTS);
        for (int eid = 0; eid < n_exp; eid++) {
            const int pair_pool = g.blk_pair_pool[blk];
            if (g.hot_score_pair[blk][eid] && pair_pool >= 0 && pair_pool < MOE_CACHE_MAX_POOLS) {
                ranked[di][pair_pool].push_back({g.hot_score_pair[blk][eid], (uint16_t)blk, (uint16_t)eid, true});
            }
            const int down_pool = g.blk_down_pool[blk];
            if (g.hot_score_down[blk][eid] && down_pool >= 0 && down_pool < MOE_CACHE_MAX_POOLS) {
                ranked[di][down_pool].push_back({g.hot_score_down[blk][eid], (uint16_t)blk, (uint16_t)eid, false});
            }
        }
    }

    memset(g.hot_pair, 0, sizeof(g.hot_pair));
    memset(g.hot_down, 0, sizeof(g.hot_down));
    long long selected = 0;
    std::vector<moe_cache_hot_candidate> selected_ranked;
    for (int di = 0; di < g.n_dev; di++) {
        moe_cache_device & d = g.dev[di];
        for (int pi = 0; pi < d.n_pools; pi++) {
            auto & v = ranked[di][pi];
            std::sort(v.begin(), v.end(), [](const moe_cache_hot_candidate & a, const moe_cache_hot_candidate & b) {
                if (a.score != b.score) return a.score > b.score;
                if (a.blk != b.blk) return a.blk < b.blk;
                if (a.eid != b.eid) return a.eid < b.eid;
                return a.pair && !b.pair;
            });
            const int take = std::min((int)v.size(), d.pools[pi].n_slots);
            for (int i = 0; i < take; i++) {
                const auto & c = v[i];
                uint64_t (*bits)[MOE_CACHE_HOT_WORDS] = c.pair ? g.hot_pair : g.hot_down;
                bits[c.blk][c.eid >> 6] |= 1ull << (c.eid & 63);
                selected_ranked.push_back(c);
            }
            selected += take;
            if (take > 0) {
                MOE_CACHE_LOG("[moe-cache] ranked restore dev=%d pool=%d selected=%d/%zu slots=%d\n",
                        di, pi, take, v.size(), d.pools[pi].n_slots);
            }
        }
    }
    std::sort(selected_ranked.begin(), selected_ranked.end(), [](const moe_cache_hot_candidate & a, const moe_cache_hot_candidate & b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.blk != b.blk) return a.blk < b.blk;
        if (a.eid != b.eid) return a.eid < b.eid;
        return a.pair && !b.pair;
    });
    g.hot_restore_order.clear();
    g.hot_restore_order.reserve(selected_ranked.size());
    for (const auto & c : selected_ranked) {
        g.hot_restore_order.push_back(((uint32_t)c.blk << 10) | ((uint32_t)c.eid << 1) | (c.pair ? 1u : 0u));
    }
    g.hot_restore_pos = 0;
    g.hot_restore_ready = true;
    g.backfill.active = true;
    g.backfill.done = false;
    g.backfill.phase = 0;
    g.backfill.blk = 0;
    g.backfill.eid = 0;
    g.backfill.hot_jobs = 0;
    g.backfill.reported = false;
    MOE_CACHE_LOG("[moe-cache] ranked hot-set ready: %lld entries fit actual pool budgets\n", selected);
    g.cv.notify_all();
}

// ---- insert workers ----------------------------------------------------------

// pick the next backfill insert (g.mu held). Walks blocks in order; for each
// block inserts the paired gate/up entry and the down entry for every expert
// until the owning pools fill. Returns false when the walk is exhausted.
static bool moe_cache_backfill_next(moe_cache_job & out) {
restart:
    if (g.backfill.phase == 0 && !g.hot_loaded) {
        if (g.backfill.hot_only) {
            g.backfill.done = true;
            if (!g.backfill.reported) {
                g.backfill.reported = true;
                MOE_CACHE_LOG("[moe-cache] hot-only restore complete: %lld entries queued\n", g.backfill.hot_jobs);
            }
            return false;
        }
        g.backfill.phase = 1;   // no prior: straight to the sweep
    }
    if (g.backfill.phase == 0 && g.hot_restore_ready) {
        while (g.hot_restore_pos < g.hot_restore_order.size()) {
            const uint32_t code = g.hot_restore_order[g.hot_restore_pos++];
            const int blk = (int)(code >> 10);
            const int eid = (int)((code >> 1) & 0x1ffu);
            const bool want_pair = (code & 1u) != 0;
            const int di = blk % g.n_dev;
            moe_cache_device & d = g.dev[di];
            if (d.dead) continue;
            const int pi = want_pair ? g.blk_pair_pool[blk] : g.blk_down_pool[blk];
            if (pi < 0 || pi >= d.n_pools) continue;
            moe_cache_pool & p = d.pools[pi];
            if (!p.slab || p.n_used >= p.n_slots) continue;
            if (want_pair && (!p.paired || !g.role_base[0][blk] || !g.role_base[1][blk])) continue;
            if (!want_pair && !g.blk_down_base[blk]) continue;
            const uint64_t key = want_pair
                ? moe_cache_key(MOE_CACHE_PAIR_KEY_TAG ^ ((uint64_t)blk << 32) ^ moe_cache_ptr_hash(g.role_base[0][blk]), eid)
                : moe_cache_key(g.blk_down_kb[blk], eid);
            if (p.map.count(key)) continue;
            const int si = p.n_used++;
            p.slots[si] = moe_cache_slot{key, -1, -1, false, true};
            moe_cache_lru_push_back(p, si);
            p.map[key] = si;
            d.inserts++;
            g.backfill.hot_jobs++;
            out = moe_cache_job{di, pi, key, si,
                    want_pair ? (const char *)g.role_base[1][blk] + (size_t)eid * p.expert_size
                              : (const char *)g.blk_down_base[blk] + (size_t)eid * p.expert_size,
                    want_pair ? (const char *)g.role_base[0][blk] + (size_t)eid * p.expert_size : nullptr,
                    p.expert_size, blk, eid};
            return true;
        }
        if (g.backfill.hot_only) {
            g.backfill.done = true;
            if (!g.backfill.reported) {
                g.backfill.reported = true;
                MOE_CACHE_LOG("[moe-cache] hot-only ranked restore complete: %lld entries queued\n", g.backfill.hot_jobs);
            }
            return false;
        }
        g.backfill.phase = 1;
        g.backfill.blk = 0;
        g.backfill.eid = 0;
        goto restart;
    }
    for (; g.backfill.blk < 1024; g.backfill.blk++, g.backfill.eid = 0) {
        const int blk = g.backfill.blk;
        const int di  = blk % g.n_dev;
        moe_cache_device & d = g.dev[di];
        if (d.dead || d.n_pools == 0) continue;
        if (g.blk_pair_pool[blk] < 0 && g.blk_down_pool[blk] < 0) continue;   // never visited

        // eid cursor covers the pair entry and the down entry for each expert:
        // unit = eid*2 (pair) and eid*2+1 (down)
        const int n_exp = g.blk_n_expert[blk];
        while (g.backfill.eid < 2 * n_exp) {
            const int unit = g.backfill.eid++;
            const int eid  = unit >> 1;
            const bool want_pair = (unit & 1) == 0;
            if (g.backfill.phase == 0) {
                if (eid >= MOE_CACHE_MAX_HOT_EXPERTS) continue;
                // hot-prior pass: only entries that were resident last session
                const uint64_t * hb = want_pair ? g.hot_pair[blk] : g.hot_down[blk];
                if (!((hb[eid >> 6] >> (eid & 63)) & 1)) continue;
            }

            const int pi = want_pair ? g.blk_pair_pool[blk] : g.blk_down_pool[blk];
            if (pi < 0 || pi >= d.n_pools) continue;
            moe_cache_pool & p = d.pools[pi];
            if (!p.slab || p.n_used >= p.n_slots) continue;
            if (want_pair && (!p.paired || !g.role_base[0][blk] || !g.role_base[1][blk])) continue;
            if (!want_pair && !g.blk_down_base[blk]) continue;

            const uint64_t key = want_pair
                ? moe_cache_key(MOE_CACHE_PAIR_KEY_TAG ^ ((uint64_t)blk << 32) ^ moe_cache_ptr_hash(g.role_base[0][blk]), eid)
                : moe_cache_key(g.blk_down_kb[blk], eid);
            if (p.map.count(key)) continue;

            const int si = p.n_used++;
            p.slots[si] = moe_cache_slot{key, -1, -1, false, true};
            moe_cache_lru_push_back(p, si);
            p.map[key] = si;
            d.inserts++;
            if (g.backfill.phase == 0) g.backfill.hot_jobs++;
            out = moe_cache_job{di, pi, key, si,
                          want_pair ? (const char *)g.role_base[1][blk] + (size_t)eid * p.expert_size
                                    : (const char *)g.blk_down_base[blk] + (size_t)eid * p.expert_size,
                          want_pair ? (const char *)g.role_base[0][blk] + (size_t)eid * p.expert_size
                                    : nullptr,
                          p.expert_size, blk, eid};
            return true;
        }
    }
    if (g.backfill.phase == 0) {
        if (g.backfill.hot_only) {
            g.backfill.done = true;
            if (!g.backfill.reported) {
                g.backfill.reported = true;
                MOE_CACHE_LOG("[moe-cache] hot-only restore complete: %lld entries queued\n", g.backfill.hot_jobs);
            }
            return false;
        }
        // hot prior exhausted: run the sequential sweep for the rest
        g.backfill.phase = 1;
        g.backfill.blk = 0;
        g.backfill.eid = 0;
        goto restart;
    }
    g.backfill.done = true;
    return false;
}

// periodic hot-set save (worker 0; runs from the idle tick so it fires even
// when no inserts are flowing - short sessions still persist their hot set)
static void moe_cache_hotset_save_tick(int wid) {
    if (wid != 0 || !g.hotset_enabled || !g.hotset_path[0]) return;
    const int64_t now = ggml_time_us();
    if (now - g.hotset_last_save < (int64_t)g.hotset_save_sec * 1000000ll) return;
    g.hotset_last_save = now;
    static uint16_t snap_pair[1024][MOE_CACHE_MAX_HOT_EXPERTS];
    static uint16_t snap_down[1024][MOE_CACHE_MAX_HOT_EXPERTS];
    {
        std::lock_guard<std::mutex> lk2(g.mu);
        memcpy(snap_pair, g.hot_score_pair, sizeof(snap_pair));
        memcpy(snap_down, g.hot_score_down, sizeof(snap_down));
    }
    char tmp[560];
    snprintf(tmp, sizeof(tmp), "%s.tmp", g.hotset_path);
    FILE * f = fopen(tmp, "wb");
    if (f) {
        const moe_cache_hotset_header h = {{'M','O','E','H','O','T','3','\0'}, MOE_CACHE_HOTSET_VERSION,
                1024, MOE_CACHE_MAX_HOT_EXPERTS, sizeof(uint16_t)};
        const bool ok = fwrite(&h, 1, sizeof(h), f) == sizeof(h)
                     && fwrite(snap_pair, 1, sizeof(snap_pair), f) == sizeof(snap_pair)
                     && fwrite(snap_down, 1, sizeof(snap_down), f) == sizeof(snap_down);
        fclose(f);
        // MSVCRT rename() does not replace an existing destination. Without
        // removing it, Windows persisted the first snapshot forever.
        if (ok) remove(g.hotset_path);
        if (ok && rename(tmp, g.hotset_path) == 0) {
            MOE_CACHE_DBG("[moe-cache-dbg] hot-set saved\n");
        } else {
            MOE_CACHE_LOG("[moe-cache] hot-set save failed: %s\n", g.hotset_path);
        }
    }
}

static void moe_cache_worker_main(int wid) {
    // per-worker pinned staging buffer + per-device copy streams: the host
    // memcpy runs at RAM speed on this thread, the H2D is a true async DMA on
    // a dedicated stream - no pageable-copy driver contention with the
    // dispatch path's kernel launches.
    char * stage = nullptr;
    size_t stage_cap = 0;
    cudaStream_t cstream[MOE_CACHE_MAX_DEV] = {};

    for (;;) {
        moe_cache_job job;
        {
            std::unique_lock<std::mutex> lk(g.mu);
            while (g.queue.empty()) {
                if (g.backfill.enabled && g.backfill.active && !g.backfill.done && moe_cache_backfill_next(job)) {
                    goto have_job;
                }
                lk.unlock();
                moe_cache_hotset_save_tick(wid);
                lk.lock();
                if (!g.queue.empty()) {
                    break;
                }
                g.cv.wait_for(lk, std::chrono::milliseconds(50));
            }
            job = g.queue.front();
            g.queue.pop_front();
        have_job:
            g.inflight_src[wid] = job.src;
            g.inflight_len[wid] = job.bytes;
        }

        moe_cache_pool & p = g.dev[job.dev].pools[job.pool];
        cudaSetDevice(job.dev);
        MOE_CACHE_DBG("[moe-cache-dbg] worker job dev=%d slot=%d bytes=%zu\n", job.dev, job.slot_idx, job.bytes);

        char * dst = p.slab + (size_t)job.slot_idx * p.expert_size;
        char * dst_gate = job.src_gate ? p.slab2 + (size_t)job.slot_idx * p.expert_size : nullptr;

        cudaError_t err = cudaSuccess;
        long long stage_us = 0;
        long long h2d_us = 0;
        if (stage_cap < job.bytes) {
            if (stage) cudaFreeHost(stage);
            err = cudaMallocHost((void **)&stage, job.bytes * 2);
            stage_cap = (err == cudaSuccess) ? job.bytes * 2 : 0;
            if (err != cudaSuccess) stage = nullptr;
        }
        if (!cstream[job.dev]) {
            cudaStreamCreateWithFlags(&cstream[job.dev], cudaStreamNonBlocking);
        }
        if (err == cudaSuccess && stage && cstream[job.dev]) {
            int64_t ts = ggml_time_us();
            memcpy(stage, job.src, job.bytes);
            stage_us += ggml_time_us() - ts;
            ts = ggml_time_us();
            err = cudaMemcpyAsync(dst, stage, job.bytes, cudaMemcpyHostToDevice, cstream[job.dev]);
            if (err == cudaSuccess) {
                err = cudaStreamSynchronize(cstream[job.dev]);
            }
            h2d_us += ggml_time_us() - ts;
            if (err == cudaSuccess && dst_gate) {
                ts = ggml_time_us();
                memcpy(stage, job.src_gate, job.bytes);
                stage_us += ggml_time_us() - ts;
                ts = ggml_time_us();
                err = cudaMemcpyAsync(dst_gate, stage, job.bytes, cudaMemcpyHostToDevice, cstream[job.dev]);
                if (err == cudaSuccess) {
                    err = cudaStreamSynchronize(cstream[job.dev]);
                }
                h2d_us += ggml_time_us() - ts;
            }
        } else if (err == cudaSuccess) {
            // pinned alloc failed: fall back to a direct pageable copy
            const int64_t ts = ggml_time_us();
            err = cudaMemcpy(dst, job.src, job.bytes, cudaMemcpyHostToDevice);
            if (err == cudaSuccess && dst_gate) {
                err = cudaMemcpy(dst_gate, job.src_gate, job.bytes, cudaMemcpyHostToDevice);
            }
            h2d_us += ggml_time_us() - ts;
        }

        {
            std::lock_guard<std::mutex> lk(g.mu);
            g.inflight_src[wid] = nullptr;
            g.inflight_len[wid] = 0;
            g.cv_idle.notify_all();
            moe_cache_device & d = g.dev[job.dev];
            d.insert_stage_us += stage_us;
            d.insert_h2d_us += h2d_us;
            d.insert_h2d_bytes += job.bytes * (job.src_gate ? 2 : 1);
            moe_cache_slot & s = p.slots[job.slot_idx];
            if (s.queued && s.key == job.key) {
                if (job.blk >= 0 && job.blk < 1024 && job.eid >= 0 && job.eid < MOE_CACHE_MAX_HOT_EXPERTS) {
                    (job.src_gate ? g.resident_pair : g.resident_down)[job.blk][job.eid >> 6]
                        |= 1ull << (job.eid & 63);
                }
                s.queued = false;
                if (err == cudaSuccess) {
                    s.valid = true;
                } else {
                    p.map.erase(s.key);
                    s.key = 0;
                }
            }
            if (err != cudaSuccess) {
                cudaGetLastError();
                static int warned = 0;
                if (warned++ < 3) {
                    MOE_CACHE_LOG("[moe-cache] insert copy failed: %s\n", cudaGetErrorString(err));
                }
            }
        }
    }
}

static void moe_cache_start_workers() {
    if (g.workers_started) return;
    g.workers_started = true;
    for (int i = 0; i < g.n_workers && i < 16; i++) {
        std::thread(moe_cache_worker_main, i).detach();
    }
}

// ---- init ---------------------------------------------------------------------

// Shape discovery. Pools are created per device ON DEMAND once any tensor name
// has repeated globally (i.e. the steady decode loop has begun - allocating on
// the very first sighting would mis-budget before the model placement and KV
// allocations settle). There is no "warmup complete" latch: a shape first seen
// late (odd per-layer quants, partial GPU placement, bench context churn)
// simply gets its pool late. No visit order can lock a device out.
struct moe_cache_discovery {
    std::unordered_set<uint64_t> seen;
    struct shape { size_t size; int wtype; int n_tensors; int roles; int64_t n_expert; };
    std::vector<shape> pending[MOE_CACHE_MAX_DEV];   // shapes seen, pool not yet built
    bool any_repeat = false;
    // stable-census guard: pools are built only after the shape census has not
    // changed for a window of eligible visits AND a tensor has repeated. A
    // partially-discovered census mis-sizes pools permanently (measured: the
    // 754B server lost its down pools to visit-order luck).
    int  stable_count = 0;
};
static moe_cache_discovery g_disc;

// build one pool for (size, wtype) on device di; caller ensures no duplicate
static bool moe_cache_pool_alloc(int di, size_t expert_size, int wtype, size_t budget, bool paired,
                           int64_t max_entries) {
    moe_cache_device & d = g.dev[di];
    if (d.n_pools >= MOE_CACHE_MAX_POOLS) return false;

    ggml_cuda_set_device(di);

    // a paired slot stores the (gate, up) tensors of one expert in two
    // parallel slabs: same byte budget, half the slot count, same number of
    // cached EXPERTS per byte as two independent entries - but joint by
    // construction, which the fused kernel requires
    int ns = (int)(budget / (paired ? 2 * expert_size : expert_size));
    // cap slots at the number of distinct cacheable entries (safe now: pools
    // are only built after the stable-census window, so n_tensors is real)
    if (max_entries > 0 && ns > max_entries) ns = (int)max_entries;
    if (ns < 64) {
        static int warned = 0;
        if (warned++ < 2) {
            MOE_CACHE_LOG("[moe-cache] dev=%d pool for %zu KB slots skipped (budget %zu MB too small) - cache stays off for this shape\n",
                    di, expert_size >> 10, budget >> 20);
        }
        // dead marker: prevents endless re-discovery + re-trigger + log spam
        moe_cache_pool & p = d.pools[d.n_pools];
        p.expert_size = expert_size;
        p.wtype       = wtype;
        p.slab        = nullptr;
        p.n_slots     = 0;
        d.n_pools++;
        return false;
    }

    char * slab = nullptr;
    cudaError_t err = cudaMalloc((void **)&slab, (size_t)ns * expert_size);
    if (err != cudaSuccess) {
        cudaGetLastError();
        MOE_CACHE_LOG("[moe-cache] dev=%d pool alloc failed: %s\n", di, cudaGetErrorString(err));
        return false;
    }
    char * slab2 = nullptr;
    if (paired) {
        err = cudaMalloc((void **)&slab2, (size_t)ns * expert_size);
        if (err != cudaSuccess) {
            cudaGetLastError();
            cudaFree(slab);
            MOE_CACHE_LOG("[moe-cache] dev=%d paired pool alloc failed: %s\n", di, cudaGetErrorString(err));
            return false;
        }
    }

    moe_cache_pool & p = d.pools[d.n_pools];
    p.expert_size = expert_size;
    p.wtype       = wtype;
    p.slab        = slab;
    p.slab2       = slab2;
    p.paired      = paired;
    p.n_slots     = ns;
    p.n_used      = 0;
    p.map.clear();
    p.lru_head = p.lru_tail = -1;
    p.slots.assign(ns, moe_cache_slot{0, -1, -1, false, false});
    d.n_pools++;
    MOE_CACHE_LOG("[moe-cache] dev=%d pool[%d]: type=%d slot=%zu KB slots=%d total=%zu MB%s\n",
            di, d.n_pools - 1, wtype, expert_size >> 10, ns,
            ((size_t)(paired ? 2 : 1) * ns * expert_size) >> 20, paired ? " (paired)" : "");

    if (!d.compute_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&d.compute_stream, cudaStreamNonBlocking));
    }

    moe_cache_start_workers();
    return true;
}

// ---- API: begin -----------------------------------------------------------------

static int moe_cache_begin(const char * name, const void * host_base, size_t expert_size,
                     int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens) {
    GGML_UNUSED(n_in); GGML_UNUSED(n_out);

    if (!g.enabled || n_tokens < 1) return -1;
    if (expert_size < g.min_expert_bytes) return -1;
    const bool pp_phase = n_tokens > g.max_batch;   // discovery-only visit

    // single-owner engagement: begin..collect carries per-node state in g.cur_*;
    // a second thread (concurrent llama_context) gets a clean refusal instead
    // of corrupted planning state
    {
        const uint64_t self = (uint64_t)(uintptr_t)&self;   // stack addr as cheap thread tag
        GGML_UNUSED(self);
        static std::atomic<int64_t> owner{-1};
        // note: ggml threadpools reuse thread 0 across graphs; identify by
        // thread id, claimed lazily and never released (single decode thread
        // is the rule; a true second decoder simply never engages)
        static thread_local bool is_owner_thread = false;
        if (!is_owner_thread) {
            int64_t expect = -1;
            static std::atomic<int64_t> next_id{1};
            static thread_local int64_t my_id = next_id.fetch_add(1);
            if (owner.compare_exchange_strong(expect, my_id)) {
                is_owner_thread = true;
            } else if (owner.load() == my_id) {
                is_owner_thread = true;
            } else {
                return -1;
            }
        }
    }
    if (g.io_stats_every > 0 && !g_io_started) {
        g_io_base = moe_cache_sample_io();
        g_io_started = true;
    }

    // only types with a kernel case in mul_mat_vec_q_switch_type - anything else
    // would GGML_ABORT on the first cached row (e.g. F16/BF16/TQ expert tensors)
    switch ((ggml_type)wtype) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:  case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:   case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
            break;
        default:
            return -1;
    }

    const char * p = strstr(name, "blk.");
    if (!p || !strstr(name, "_exps")) return -1;
    const int blk = atoi(p + 4);

    // DSV4's merged GGUF uses block 43 for MTP, while Qwen3.8 has 48 ordinary
    // trunk blocks. Make the ceiling an explicit model-placement boundary:
    // hybrid Qwen launchers set it to the final CPU-MoE block so this cache can
    // never compete with the natively resident GPU expert layers.
    if (g.max_block >= 0 && blk > g.max_block) return -1;

    int role = -1;
    if      (strstr(name, "_gate_exps")) role = 0;
    else if (strstr(name, "_up_exps"))   role = 1;
    else if (strstr(name, "_down_exps")) role = 2;

    const int di = blk % g.n_dev;

    const uint64_t kb = moe_cache_fnv1a(name);
    moe_cache_device & d = g.dev[di];
    if (d.dead) return -1;
    const bool first_sight = g_disc.seen.count(kb) == 0;

    // shape discovery + on-demand pool construction (see moe_cache_discovery)
    int pi = -1;
    for (int i = 0; i < d.n_pools; i++) {
        if (d.pools[i].expert_size == expert_size && d.pools[i].wtype == wtype) { pi = i; break; }
    }
    if (pi < 0) {
        moe_cache_discovery::shape * shp = nullptr;
        for (auto & sh : g_disc.pending[di]) {
            if (sh.size == expert_size && sh.wtype == wtype) { shp = &sh; break; }
        }
        if (!shp) {
            g_disc.pending[di].push_back({expert_size, wtype, 0, 0, n_expert});
            shp = &g_disc.pending[di].back();
            g_disc.stable_count = 0;   // census changed: restart the stability window
            MOE_CACHE_DBG("[moe-cache-dbg] new shape %s blk=%d dev=%d size=%zu type=%d\n",
                    name, blk, di, expert_size, wtype);
        } else {
            g_disc.stable_count++;
        }
        if (first_sight) shp->n_tensors++;   // distinct tensors using this shape
        if (role >= 0) shp->roles |= 1 << role;
        if (!g_disc.any_repeat) {
            if (g_disc.seen.count(kb)) {
                g_disc.any_repeat = true;
            } else {
                g_disc.seen.insert(kb);
                return -1;
            }
        }
        // stable-census window: 64 eligible visits without a new shape
        if (g_disc.stable_count < 64) {
            return -1;
        }
        static bool announced = false;
        if (!announced) {
            announced = true;
            MOE_CACHE_LOG("[moe-cache] decode loop detected, shape census stable - building pools\n");
            if (g.hotset_enabled && !g.hotset_path[0]) {
                // fingerprint: model-distinguishing facts known at this point
                const char * explicit_path = getenv("GGML_CUDA_MOE_CACHE_HOTSET_PATH");
                const char * base = getenv("HOME");           // POSIX
                if (!base) base = getenv("LOCALAPPDATA");     // Windows
                if (explicit_path && explicit_path[0]) {
                    snprintf(g.hotset_path, sizeof(g.hotset_path), "%s", explicit_path);
                } else if (base) {
                    snprintf(g.hotset_path, sizeof(g.hotset_path),
                             "%s/.cache/llama.cpp/moe-cache-hotset-%lldx%dx%zu-d%d.bin",
                             base, (long long)n_expert, blk >= 0 ? 1 : 0,
                             expert_size, g.n_dev);
                } else {
                    g.hotset_enabled = false;                 // no cache dir: skip persistence
                }
                if (g.hotset_enabled) {
                    FILE * f = fopen(g.hotset_path, "rb");
                    if (f) {
                        moe_cache_hotset_header h = {};
                        const bool header_ok = fread(&h, 1, sizeof(h), f) == sizeof(h)
                            && memcmp(h.magic, "MOEHOT3", 7) == 0
                            && h.version == MOE_CACHE_HOTSET_VERSION
                            && h.blocks == 1024
                            && h.experts == MOE_CACHE_MAX_HOT_EXPERTS
                            && h.score_bytes == sizeof(uint16_t);
                        const bool ok = header_ok
                            && fread(g.hot_score_pair, 1, sizeof(g.hot_score_pair), f) == sizeof(g.hot_score_pair)
                            && fread(g.hot_score_down, 1, sizeof(g.hot_score_down), f) == sizeof(g.hot_score_down);
                        fclose(f);
                        if (ok) {
                            long ranked = 0;
                            for (int b2 = 0; b2 < 1024; b2++) {
                                for (int e2 = 0; e2 < MOE_CACHE_MAX_HOT_EXPERTS; e2++) {
                                    uint16_t & ps = g.hot_score_pair[b2][e2];
                                    uint16_t & ds = g.hot_score_down[b2][e2];
                                    ps = (uint16_t)((ps + 1) >> 1);
                                    ds = (uint16_t)((ds + 1) >> 1);
                                    ranked += ps != 0;
                                    ranked += ds != 0;
                                }
                            }
                            g.hot_loaded = ranked > 0;
                            MOE_CACHE_LOG("[moe-cache] ranked hot-set loaded: %ld candidates; waiting for pool census\n", ranked);
                        } else {
                            MOE_CACHE_LOG("[moe-cache] hot-set ignored: incompatible or incomplete v3 file\n");
                        }
                    }
                }
            }
        }
        // steady state reached: build this device's pending pools in ONE
        // proportional pass. Budgets are weighted by referenced bytes per
        // token (shape size x number of tensors using the shape - gate+up
        // share a shape, so theirs weighs ~2x per layer vs down's 1x); the
        // previous sequential-halving scheme left ~25% of the budget
        // unallocated and starved the down pool (measured).
        auto & pend = g_disc.pending[di];
        if (!pend.empty()) {
            const size_t reserve = g.reserve_mb << 20;
            size_t free_mem = 0, total_mem = 0;
            ggml_cuda_set_device(di);
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            size_t avail = free_mem > reserve ? free_mem - reserve : 0;
            if (g.budget_mb > 0 && (g.budget_mb << 20) < avail) {
                avail = g.budget_mb << 20;
            }
            // Role-group budgeting. Mixed-quant models (UD-*_XL) fragment one
            // role into many (size, type) shapes; naive per-shape weighting
            // hands each fragment a sliver below the slot floor and the whole
            // role dies (measured: every down projection ran CPU-only on a
            // 24 GB device while gate/up hit 77%). Budget by role group first,
            // then allocate within the group largest-first, folding the share
            // of dead fragments into the survivors.
            auto shape_w = [](const moe_cache_discovery::shape & sh) {
                return (double)sh.size * (sh.n_tensors > 0 ? sh.n_tensors : 1);
            };
            auto is_paired_sh = [&](const moe_cache_discovery::shape & sh) {
                return g.fuse && (sh.roles & 0b11) == 0b11;
            };
            double w_pair = 0.0, w_rest = 0.0;
            for (auto & sh : pend) {
                (is_paired_sh(sh) ? w_pair : w_rest) += shape_w(sh);
            }
            const double w_all = w_pair + w_rest;
            std::vector<const moe_cache_discovery::shape *> order;
            for (auto & sh : pend) order.push_back(&sh);
            std::sort(order.begin(), order.end(),
                      [&](const moe_cache_discovery::shape * a, const moe_cache_discovery::shape * b) {
                          return shape_w(*a) > shape_w(*b);
                      });
            size_t group_left[2] = {                       // [0]=paired, [1]=rest
                w_all > 0 ? (size_t)(avail * (w_pair / w_all)) : 0,
                w_all > 0 ? (size_t)(avail * (w_rest / w_all)) : 0,
            };
            double group_w_left[2] = { w_pair, w_rest };
            for (const auto * shp_p : order) {
                const auto & sh = *shp_p;
                const bool paired = is_paired_sh(sh);
                const int  gidx   = paired ? 0 : 1;
                if (group_w_left[gidx] <= 0.0) continue;
                size_t budget = (size_t)(group_left[gidx] * (shape_w(sh) / group_w_left[gidx]));
                const int64_t max_entries = (paired ? sh.n_tensors / 2 : sh.n_tensors) * (int64_t)sh.n_expert;
                const size_t need = (size_t)max_entries * sh.size * (paired ? 2 : 1);
                if (budget > need) budget = need;          // never strand bytes in caps
                const size_t before = budget;
                if (moe_cache_pool_alloc(di, sh.size, sh.wtype, budget, paired, max_entries)) {
                    group_left[gidx] -= before;            // consumed (approx; slack folds forward)
                }
                // dead fragments consume nothing: their share flows to the
                // next shapes of the same group automatically
                group_w_left[gidx] -= shape_w(sh);
            }
        }
        pend.clear();
        for (int i = 0; i < d.n_pools; i++) {
            if (d.pools[i].expert_size == expert_size && d.pools[i].wtype == wtype) { pi = i; break; }
        }
        if (pi < 0) return -1;
    }
    if (d.pools[pi].slab == nullptr) return -1;   // dead marker (alloc failed)
    if (!g_disc.any_repeat) {
        if (g_disc.seen.count(kb)) {
            g_disc.any_repeat = true;
        } else {
            g_disc.seen.insert(kb);
            return -1;
        }
    }

    // fused-state epoch: every gate node on a device invalidates any leftover
    // fused entry (zero-hit gate nodes never reach collect, and gallocr reuses
    // dst pointers across layers - a stale entry must never survive into the
    // next layer's GLU; see MOE_CACHE_READINESS.md B1)
    if (role == 0) {
        g.dev[di].fused.active = false;
    }

    g.cur_blk  = blk;
    g.cur_role = role;

    MOE_CACHE_DBG("[moe-cache-dbg] begin %s dev=%d pool=%d role=%d\n", name, di, pi, role);

    bool ranked_restore_pending = false;
    if (blk >= 0 && blk < 1024) {
        std::lock_guard<std::mutex> lk(g.mu);
        g.blk_n_expert[blk] = (int)n_expert;
        if (d.pools[pi].paired && (role == 0 || role == 1)) {
            g.blk_pair_pool[blk] = (int8_t)pi;
            g.role_base[role][blk] = host_base;
        } else if (role == 2) {
            g.blk_down_pool[blk] = (int8_t)pi;
            g.blk_down_kb[blk]   = kb ^ moe_cache_ptr_hash(host_base);
            g.blk_down_base[blk] = host_base;
        }
        if (g.hot_loaded && g.hot_restore_last_blk >= 0 && blk < g.hot_restore_last_blk) {
            g.hot_restore_wraps++;
            moe_cache_build_ranked_restore(g.hot_restore_wraps >= 2);
        }
        if (blk != g.hot_restore_last_blk) g.hot_restore_last_blk = blk;
        if (!g.hot_loaded || g.hot_restore_ready) g.backfill.active = true;
        ranked_restore_pending = g.hot_loaded && g.backfill.hot_only && !g.backfill.done;
    }

    if (pp_phase) {
        // prompt processing: pools may now exist and the backfill workers warm
        // them in parallel with the prompt; the decode path stays untouched
        return -1;
    }
    // A hot-only ranked restore reserves the small pool for its persisted
    // winners. Until that restore has drained, run the node on the exact CPU
    // path: ordinary demand admission would otherwise fill all ~1.2 GiB while
    // the source census is completing, leaving zero slots for the ranking.
    // Demand admission resumes automatically once the ranked queue is drained.
    if (ranked_restore_pending) return -1;

    // bail-out phases (decode visits on a working pool only)
    if (n_tokens == 1 && !g.bail.tripped) {
        const long long vis = g.bail.eligible_seen++;
        if (vis < moe_cache_global::BAIL_WARM) return -1;        // warmup: no sample
        if (vis < moe_cache_global::BAIL_SAMPLE) return -3;      // pure CPU + timing sample
    }

    // paired pools share ONE entry per (blk, expert): key by blk + the GATE
    // tensor's host base (two models in one process must never alias - names
    // and blk indices collide across models, data pointers do not)
    if (d.pools[pi].paired && (role == 0 || role == 1)) {
        if (blk >= 0 && blk < 1024) g.role_base[role][blk] = host_base;
        const void * anchor = (blk >= 0 && blk < 1024 && g.role_base[0][blk])
                              ? g.role_base[0][blk] : host_base;
        g.cur_key_base = MOE_CACHE_PAIR_KEY_TAG ^ ((uint64_t)blk << 32) ^ moe_cache_ptr_hash(anchor);
    } else {
        g.cur_key_base = kb ^ moe_cache_ptr_hash(host_base);
    }
    g.cur_host_base   = host_base;
    g.cur_expert_size = expert_size;
    g.cur_n_expert    = n_expert;
    g.cur_n_tokens    = n_tokens;
    g.cur_pool        = pi;
    return di;
}

// ---- API: plan --------------------------------------------------------------------

static int moe_cache_plan(int di, const int32_t * ids, int n_ids, int32_t * slot_idx) {
    moe_cache_device & d = g.dev[di];
    moe_cache_pool   & p = d.pools[g.cur_pool];

    const int64_t t0 = ggml_time_us();

    // fused up node: reuse the gate node's recorded hit mask verbatim. The
    // rows are computed already (fused dispatch at the gate node); a fresh
    // lookup could diverge (eviction between the plans) and skip a row that
    // nobody computed.
    if (moe_cache_decode_like() && p.paired && g.cur_role == 1 &&
        g.cur_blk >= 0 && g.cur_blk < 1024 && g.safe_fuse_blk[g.cur_blk] &&
        d.fused.active && d.fused.gate_dst != nullptr) {
        int nh = 0;
        const int cap = moe_cache_fuse_row_cap();
        for (int k = 0; k < n_ids && k < cap; k++) {
            const bool hit = (k < 512) && ((d.fused.mask_w[k / 64] >> (k % 64)) & 1ull);
            slot_idx[k] = hit ? 0 : -1;   // value unused (no dispatch); sign is the skip signal
            if (hit) nh++;
        }
        g.cur_n_ids = n_ids;
        for (int k = 0; k < n_ids && k < cap && k < MOE_CACHE_MAX_ROUTED_ROWS; k++) {
            g.cur_slot_idx[k] = slot_idx[k];
        }
        d.t_plan_us += ggml_time_us() - t0;
        d.n_nodes++;
        return nh;
    }

    int n_hits = 0;
    int n_valid = 0;
    int inserts_left = g.inserts_per_plan;

    std::lock_guard<std::mutex> lk(g.mu);
    moe_cache_score_decay();

    for (int k = 0; k < n_ids; k++) {
        slot_idx[k] = -1;
        const int eid = ids[k];
        if (eid < 0 || eid >= g.cur_n_expert) continue;
        n_valid++;
        {
            if (g.cur_role == 2) {
                moe_cache_score_access(g.cur_blk, eid, false);
            } else if (p.paired && g.cur_role == 0) {
                moe_cache_score_access(g.cur_blk, eid, true);
            }
        }
        const uint64_t key = moe_cache_key(g.cur_key_base, eid);

        auto it = p.map.find(key);
        if (it != p.map.end()) {
            const int si = it->second;
            moe_cache_slot & s = p.slots[si];
            if (s.valid) {
                moe_cache_lru_remove(p, si);
                moe_cache_lru_push_back(p, si);
                slot_idx[k] = si;
                d.hits++;
                d.pool_hits[g.cur_pool]++;
                n_hits++;
            } else {
                // insert still queued/in-flight: CPU computes the row this time
                d.queued_misses++;
                d.misses++;
                d.pool_miss[g.cur_pool]++;
            }
            continue;
        }

        d.misses++;
        d.pool_miss[g.cur_pool]++;
        if (d.ever_seen.insert(key).second) {
            d.miss_compulsory++;
        } else if (d.ever_inserted.count(key)) {
            d.miss_capacity++;   // was in cache once, evicted, needed again
        } else {
            d.miss_admission++;  // seen before but never admitted
        }

        // ---- enqueue async insert (budgeted) ----
        if (inserts_left <= 0) {
            d.insert_skips++;
            d.skip_budget++;
            continue;
        }
        if ((int)g.queue.size() >= g.queue_max) {
            d.insert_skips++;
            d.skip_qfull++;
            continue;
        }
        // admission throttle at capacity: when the pool is full, churn (evict +
        // re-copy on every miss) steals host RAM bandwidth from the CPU matmuls.
        // Admit only a fraction of misses so the content still adapts but the
        // copy traffic stays bounded.
        if (p.n_used >= p.n_slots && (d.misses % g.throttle_mod) != 0) {
            d.insert_skips++;
            d.skip_throttle++;
            continue;
        }
        // paired-entry inserts (gate/up roles only - pools can be SHARED with
        // other roles whose tensors merely have the same shape; those use the
        // plain name-keyed path below and never collide in key space)
        const bool pair_entry = p.paired && (g.cur_role == 0 || g.cur_role == 1);
        if (pair_entry && (g.cur_blk < 0 || g.cur_blk >= 1024 ||
                           !g.role_base[0][g.cur_blk] || !g.role_base[1][g.cur_blk])) {
            d.insert_skips++;
            continue;
        }

        int si = -1;
        if (p.n_used < p.n_slots) {
            si = p.n_used++;
        } else {
            int cand = p.lru_head;
            int guard = 0;
            while (cand >= 0 && moe_cache_slot_evict_blocked(p.slots[cand]) && guard++ < p.n_slots) {
                cand = p.slots[cand].next;
            }
            if (cand < 0 || moe_cache_slot_evict_blocked(p.slots[cand])) {
                d.insert_skips++;
                d.skip_lrubusy++;
                continue;
            }
            si = cand;
            moe_cache_slot & old = p.slots[si];
            if (old.valid || old.queued) {
                p.map.erase(old.key);
                d.evictions++;
                // residency bitmap (hot-set persistence): this expert is leaving.
                // The key does not encode (blk,eid) reversibly, so clear lazily:
                // a stale bit merely makes the next session's warm backfill load
                // one expert that is no longer hot (harmless).
            }
            moe_cache_lru_remove(p, si);
        }

        const void * src_up   = nullptr;
        const void * src_gate = nullptr;
        if (pair_entry) {
            src_up   = (const char *)g.role_base[1][g.cur_blk] + (size_t)eid * g.cur_expert_size;
            src_gate = (const char *)g.role_base[0][g.cur_blk] + (size_t)eid * g.cur_expert_size;
        } else {
            src_up = (const char *)g.cur_host_base + (size_t)eid * g.cur_expert_size;
        }

        p.slots[si] = moe_cache_slot{key, -1, -1, false, true};
        moe_cache_lru_push_back(p, si);
        p.map[key] = si;
        d.inserts++;
        d.ever_inserted.insert(key);
        inserts_left--;

        {
            // role 2 fills the down bitmap; roles 0/1 (pair_entry) the pair bitmap
            const int bblk = (g.cur_role == 2 || pair_entry) ? g.cur_blk : -1;
            g.queue.push_back(moe_cache_job{di, g.cur_pool, key, si, src_up,
                                      pair_entry ? src_gate : nullptr,
                                      g.cur_expert_size, bblk, eid});
        }
        g.cv.notify_one();
    }

    // stash the per-position result so collect can reconstruct dst row indices
    g.cur_n_ids = n_ids;
    for (int k = 0; k < n_ids && k < MOE_CACHE_MAX_ROUTED_ROWS; k++) g.cur_slot_idx[k] = slot_idx[k];

    if (g.cur_n_tokens > 1) {
        d.prompt_nodes++;
        d.prompt_hits += n_hits;
        d.prompt_misses += n_valid - n_hits;
    } else {
        d.decode_nodes++;
        d.decode_hits += n_hits;
        d.decode_misses += n_valid - n_hits;
    }

    d.t_plan_us += ggml_time_us() - t0;
    d.n_nodes++;
    MOE_CACHE_DBG("[moe-cache-dbg] plan dev=%d hits=%d q=%zu\n", di, n_hits, g.queue.size());
    return n_hits;
}

// ---- API: dispatch ------------------------------------------------------------------

static void moe_cache_dispatch(int di, int wtype_int, int64_t n_in, int64_t n_out, int n_hits,
                         const int32_t * slot_idx_compact, const float * const * act_rows) {
    if (n_hits <= 0) return;
    const int64_t t0 = ggml_time_us();
    moe_cache_device & d = g.dev[di];
    moe_cache_pool   & p = d.pools[g.cur_pool];

    const bool blk_ok     = g.cur_blk >= 0 && g.cur_blk < 1024;
    const bool fuse_layer = moe_cache_decode_like() && p.paired && blk_ok && g.safe_fuse_blk[g.cur_blk];
    if (fuse_layer && g.cur_role == 1) {
        // fused rows were computed at the gate node; nothing to launch here
        d.t_disp_us += ggml_time_us() - t0;
        return;
    }
    ggml_cuda_set_device(di);
    cudaStream_t st = d.compute_stream;

    const ggml_type wtype = (ggml_type)wtype_int;
    const int64_t n_in_padded = ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;

    // distinct activation rows: all-same (gate/up) -> 1 row; else one per hit
    bool shared_act = true;
    for (int i = 1; i < n_hits; i++) {
        if (act_rows[i] != act_rows[0]) { shared_act = false; break; }
    }
    const int act_n = shared_act ? 1 : n_hits;

    // grow staging (pinned host + device). The stream is synchronized before
    // the next dispatch reuses these (collect syncs, or the GLU hook syncs on
    // the fused path), so single-buffered staging is safe.
    if (d.ids_cap < (size_t)n_hits) {
        const size_t cap = n_hits * 2 + 8;
        if (d.h_ids) cudaFreeHost(d.h_ids);
        if (d.d_ids) cudaFree(d.d_ids);
        d.h_ids = nullptr; d.d_ids = nullptr;
        if (!moe_cache_ok(di, cudaMallocHost((void **)&d.h_ids, cap * sizeof(int32_t)), "ids host alloc") ||
            !moe_cache_ok(di, cudaMalloc((void **)&d.d_ids, cap * sizeof(int32_t)), "ids dev alloc")) {
            d.out_rows += n_hits;
            return;
        }
        d.ids_cap = cap;
    }
    const size_t need_act = (size_t)act_n * n_in * sizeof(float);
    if (d.act_cap < need_act) {
        const size_t cap = need_act * 2;
        if (d.h_act) cudaFreeHost(d.h_act);
        if (d.d_act) cudaFree(d.d_act);
        d.h_act = nullptr; d.d_act = nullptr;
        if (!moe_cache_ok(di, cudaMallocHost((void **)&d.h_act, cap), "act host alloc") ||
            !moe_cache_ok(di, cudaMalloc((void **)&d.d_act, cap), "act dev alloc")) {
            d.out_rows += n_hits;
            return;
        }
        d.act_cap = cap;
    }
    int32_t * const h_ids_h = d.h_ids;
    int32_t * const d_ids_h = d.d_ids;
    float   * const h_act_h = d.h_act;
    float   * const d_act_h = d.d_act;
    const size_t need_q8 = (size_t)act_n * (n_in_padded / QK8_1) * sizeof(block_q8_1);
    if (d.act_q8_cap < need_q8) {
        const size_t cap = need_q8 * 2;
        if (d.d_act_q8) cudaFree(d.d_act_q8);
        d.d_act_q8 = nullptr;
        if (!moe_cache_ok(di, cudaMalloc(&d.d_act_q8, cap), "q8 alloc")) {
            d.out_rows += n_hits;
            return;
        }
        d.act_q8_cap = cap;
    }
    const size_t need_out = (size_t)(d.out_rows + n_hits) * n_out * sizeof(float);
    if (d.d_out_cap < need_out) {
        const size_t cap = ((size_t)(d.out_rows + n_hits) * n_out * sizeof(float)) * 2 + 65536;
        if (d.d_out) cudaFree(d.d_out);
        d.d_out = nullptr;
        if (!moe_cache_ok(di, cudaMalloc((void **)&d.d_out, cap), "out dev alloc")) {
            d.out_rows += n_hits;
            return;
        }
        d.d_out_cap = cap;
    }
    if (d.h_out_cap < need_out) {
        const size_t cap = need_out * 2 + 65536;
        if (d.h_out) cudaFreeHost(d.h_out);
        d.h_out = nullptr;
        if (!moe_cache_ok(di, cudaMallocHost((void **)&d.h_out, cap), "out host alloc")) {
            d.out_rows += n_hits;
            return;
        }
        d.h_out_cap = cap;
    }

    // fill pinned staging on the host (the async copies read it at execution)
    for (int i = 0; i < n_hits; i++) h_ids_h[i] = slot_idx_compact[i];

    // gate and up of one layer read the same activation row: reuse the
    // quantized copy already on the device when possible
    const bool reuse_q8 = g.reuse && act_n == 1 && g.cur_role == 1 &&
                          d.q8_act_ptr == act_rows[0] && d.q8_act_blk == g.cur_blk;
    const char * act_q8 = (const char *)d.d_act_q8;
    if (!reuse_q8) {
        for (int i = 0; i < act_n; i++) {
            memcpy(h_act_h + (size_t)i * n_in, act_rows[i], n_in * sizeof(float));
        }
        d.q8_act_ptr = act_n == 1 ? act_rows[0] : nullptr;
        d.q8_act_blk = g.cur_blk;
    }

    // the GPU chain: ids H2D [+ act H2D + quantize] + batched mmv. All buffer
    // addresses and sizes are fixed for a given shape key, so the chain can be
    // captured once into a CUDA graph and replayed as a single launch - the
    // chain's per-op launch latency is the dominant per-node cost.
    auto emit_chain = [&](cudaStream_t s) {
        if (!moe_cache_ok(di, cudaMemcpyAsync(d_ids_h, h_ids_h, n_hits * sizeof(int32_t), cudaMemcpyHostToDevice, s), "ids H2D")) return;
        if (!reuse_q8) {
            if (!moe_cache_ok(di, cudaMemcpyAsync(d_act_h, h_act_h, need_act, cudaMemcpyHostToDevice, s), "act H2D")) return;
            quantize_row_q8_1_cuda(d_act_h, /*ids=*/nullptr, (void *)act_q8, wtype,
                                   n_in, /*s01=*/n_in, /*s02=*/(int64_t)act_n * n_in, /*s03=*/(int64_t)act_n * n_in,
                                   n_in_padded, /*ne1=*/act_n, /*ne2=*/1, /*ne3=*/1, s);
        }
        const void * mmv_x   = p.slab;
        const void * mmv_gate = nullptr;
        int          mmv_glu  = -1;
        if (p.paired) {
            if (fuse_layer && g.cur_role == 0) {
                // fused: x = up slab (result operand), gate slab silu'd on-chip
                mmv_gate = p.slab2;
                mmv_glu  = (int)GGML_GLU_OP_SWIGLU;
            } else if (g.cur_role == 0) {
                mmv_x = p.slab2;   // separate gate matvec reads the gate slab
            }
        }
        ggml_cuda_moe_cache_mmv(mmv_x, wtype, act_q8, d_ids_h, d.d_out + (size_t)d.out_rows * n_out,
                          n_in, n_out, p.n_slots, (int64_t)p.expert_size,
                          n_hits, /*act_rows=*/act_n, s, mmv_gate, mmv_glu);
    };

    // note: CUDA-graph capture of this chain was tried and measured to be a
    // net loss - the chain is GPU-exec-bound, not launch-bound, and pools can
    // hold mixed (n_in, n_out) shapes which makes graph keying hazardous.
    emit_chain(st);
    d.dispatch_h2d_bytes += n_hits * sizeof(int32_t) + (reuse_q8 ? 0 : need_act);

    d.out_rows += n_hits;
    d.t_disp_us += ggml_time_us() - t0;
    MOE_CACHE_DBG("[moe-cache-dbg] dispatch dev=%d hits=%d act_n=%d n_in=%lld n_out=%lld\n",
            di, n_hits, act_n, (long long)n_in, (long long)n_out);
}

// ---- API: collect --------------------------------------------------------------------

static void moe_cache_stats(void);

static void moe_cache_collect(int di, int n_hits, float * const * dst_rows, int64_t n_out) {
    const int64_t t0 = ggml_time_us();
    moe_cache_device & d = g.dev[di];

    // fused-GLU learning must happen before any early return: record this
    // node's dst base so the GLU hook can prove the pair wiring per layer
    if (moe_cache_decode_like() && g.cur_pool >= 0 && d.pools[g.cur_pool].paired && n_hits > 0 &&
        g.cur_blk >= 0 && g.cur_blk < 1024 && (g.cur_role == 0 || g.cur_role == 1)) {
        int k0l = -1;
        for (int k = 0; k < g.cur_n_ids; k++) {
            if (g.cur_slot_idx[k] >= 0) { k0l = k; break; }
        }
        if (k0l >= 0) {
            const char * dbase = (const char *)dst_rows[0] - (size_t)k0l * n_out * sizeof(float);
            if (g.cur_role == 0) {
                g.learn_gate_dst[g.cur_blk] = dbase;
                g.glu_learn[dbase] = g.cur_blk;
                MOE_CACHE_DBG("[moe-cache-dbg] gate-dst blk=%d %p\n", g.cur_blk, (const void *)dbase);
            } else {
                g.learn_up_dst[g.cur_blk] = dbase;
            }
        }
    }

    ggml_cuda_set_device(di);

    // ---- fused gate+up+GLU path (paired pools) ----
    if (moe_cache_decode_like() && g.cur_pool >= 0 && d.pools[g.cur_pool].paired && n_hits > 0 &&
        g.cur_blk >= 0 && g.cur_blk < 1024 && (g.cur_role == 0 || g.cur_role == 1)) {
        // reconstruct this node's dst base from the first hit's ids position
        int k0 = -1;
        for (int k = 0; k < g.cur_n_ids; k++) {
            if (g.cur_slot_idx[k] >= 0) { k0 = k; break; }
        }
        const char * dst_base = k0 >= 0
            ? (const char *)dst_rows[0] - (size_t)k0 * n_out * sizeof(float) : nullptr;

        if (g.safe_fuse_blk[g.cur_blk]) {
            if (g.cur_role == 0) {
                // gate node of a fused layer: d_out holds the fused swiglu rows.
                // D2H them (async) and hand off to the GLU hook; nothing is
                // written to the gate/up MMID dsts (the GLU kernel skips these
                // rows and nothing else reads them).
                auto & f = d.fused;
                f.active   = true;
                f.scattered = false;
                f.serial   = ++g.fuse_serial;
                f.gate_dst = dst_base;
                f.up_dst   = nullptr;   // filled at the up node
                f.mask     = 0;
                memset(f.mask_w, 0, sizeof(f.mask_w));
                f.n        = 0;
                f.n_out    = n_out;
                {
                    const int cap = moe_cache_fuse_row_cap();
                    for (int k = 0; k < g.cur_n_ids && f.n < cap; k++) {
                        if (g.cur_slot_idx[k] < 0) continue;
                        f.rows[f.n++] = k;
                        if (k < 64) {
                            f.mask |= 1ull << k;
                        }
                        if (k < 512) {
                            f.mask_w[k / 64] |= 1ull << (k % 64);
                        }
                    }
                }
                const size_t bytes = (size_t)d.out_rows * n_out * sizeof(float);
                moe_cache_ok(di, cudaMemcpyAsync(d.h_out, d.d_out, bytes, cudaMemcpyDeviceToHost, d.compute_stream), "fused D2H");
                d.out_rows = 0;
                d.q8_act_ptr = nullptr;
                d.fused_layers++;
                d.t_coll_us += ggml_time_us() - t0;
                if (g.stats_every > 0 && ++g.collect_calls % g.stats_every == 0) moe_cache_stats();
                return;
            }
            if (g.cur_role == 1) {
                // up node of a fused layer: nothing was dispatched; just record
                // the up dst base so the GLU hook can match the pair
                if (d.fused.active && dst_base) d.fused.up_dst = dst_base;
                d.out_rows = 0;
                d.t_coll_us += ggml_time_us() - t0;
                return;
            }
        }
    }

    // ---- GPU-resident dst handoff (down nodes) ----
    // If the scheduler offered the consumer's GPU copy of this dst, scatter the
    // hit rows straight into it (peer write, async) and skip the D2H + host
    // scatter + thread-0 sync entirely. CPU miss rows are uploaded later in
    // redirect_finalize (after the node barrier, when they are complete).
    if (g.redirect_on && g.cur_n_tokens == 1 && g.cur_role == 2 && n_hits > 0 && n_hits <= 64) {
        moe_cache_global::redirect_entry * re = nullptr;
        const void * base = nullptr;
        for (auto & kv : g.redirect) {
            const char * b = (const char *)kv.first;
            if ((const char *)dst_rows[0] >= b &&
                (const char *)dst_rows[0] <  b + (size_t)kv.second.n_rows * kv.second.nb1) {
                re = &kv.second;
                base = kv.first;
                break;
            }
        }
        if (re && re->n_rows <= 64) {
            // P2P-free relay: async-D2H each hit row into a pinned full-tensor
            // image at its TARGET row offset, record an event. No host sync.
            // redirect_finalize fills the miss rows into the same image and
            // issues one H2D on the consumer's stream, ordered by the event.
            const size_t img_cap = 64 * re->nb1;
            bool rok = true;
            if (d.h_redir_half < img_cap) {
                if (d.h_redir) cudaFreeHost(d.h_redir);
                d.h_redir = nullptr;
                rok = moe_cache_ok(di, cudaMallocHost((void **)&d.h_redir, 2 * img_cap), "redirect image alloc");
                d.h_redir_half = rok ? img_cap : 0;
            }
            d.redir_par ^= 1;
            const int par = d.redir_par;
            char * img = d.h_redir + (size_t)par * d.h_redir_half;
            if (rok && !d.redir_evt[par]) {
                rok = moe_cache_ok(di, cudaEventCreateWithFlags(&d.redir_evt[par], cudaEventDisableTiming), "redirect event create");
            }
            uint64_t mask = 0;
            for (int i = 0; rok && i < n_hits; i++) {
                const int ridx = (int)(((const char *)dst_rows[i] - (const char *)base) / re->nb1);
                mask |= 1ull << ridx;
                rok = moe_cache_ok(di, cudaMemcpyAsync(img + (size_t)ridx * re->nb1,
                                           d.d_out + (size_t)i * n_out,
                                           n_out * sizeof(float),
                                           cudaMemcpyDeviceToHost, d.compute_stream), "redirect D2H");
            }
            if (rok) {
                rok = moe_cache_ok(di, cudaEventRecord(d.redir_evt[par], d.compute_stream), "redirect event record");
            }
            if (!rok) {
                // degraded: ship finite zeros via the scheduler's normal copy
                for (int i = 0; i < n_hits; i++) memset(dst_rows[i], 0, n_out * sizeof(float));
            }
            re->hit_mask  = mask;
            re->populated = rok;
            re->dev       = di;
            re->par       = par;
            d.out_rows = 0;
            d.t_coll_us += ggml_time_us() - t0;
            if (g.stats_every > 0 && ++g.collect_calls % g.stats_every == 0) moe_cache_stats();
            return;
        }
    }

    MOE_CACHE_DBG("[moe-cache-dbg] collect dev=%d rows=%d pre-sync\n", di, d.out_rows);
    const size_t bytes = (size_t)d.out_rows * n_out * sizeof(float);
    const bool cok = !d.dead && d.h_out && d.d_out
        && moe_cache_ok(di, cudaMemcpyAsync(d.h_out, d.d_out, bytes, cudaMemcpyDeviceToHost, d.compute_stream), "collect D2H")
        && moe_cache_ok(di, cudaStreamSynchronize(d.compute_stream), "collect sync");
    MOE_CACHE_DBG("[moe-cache-dbg] collect dev=%d post-sync\n", di);
    for (int i = 0; i < n_hits; i++) {
        if (cok) memcpy(dst_rows[i], d.h_out + (size_t)i * n_out, n_out * sizeof(float));
        else     memset(dst_rows[i], 0, n_out * sizeof(float));
    }
    d.out_rows   = 0;
    d.q8_act_ptr = nullptr;
    d.t_coll_us += ggml_time_us() - t0;

    if (g.stats_every > 0 && ++g.collect_calls % g.stats_every == 0) {
        moe_cache_stats();
    }
}

// ---- API: GPU-resident dst handoff ----------------------------------------------------

static void moe_cache_redirect_offer(const void * host_dst_data, size_t nb1, int64_t n_rows,
                               void * gpu_copy_data, void * consumer_backend) {
    GGML_UNUSED(consumer_backend);
    if (!g.redirect_on || n_rows > 64) return;
    moe_cache_global::redirect_entry re;
    re.nb1     = nb1;
    re.n_rows  = n_rows;
    re.gpu_ptr = gpu_copy_data;
    g.redirect[host_dst_data] = re;
}

static int moe_cache_redirect_finalize(const void * host_dst_data, void * consumer_backend) {
    auto it = g.redirect.find(host_dst_data);
    if (it == g.redirect.end()) return 0;
    moe_cache_global::redirect_entry re = it->second;
    g.redirect.erase(it);
    if (!re.populated || re.dev < 0) return 0;

    moe_cache_device & d = g.dev[re.dev];
    char * img = d.h_redir + (size_t)re.par * d.h_redir_half;

    // fill the miss rows into the pinned image (the node barrier has passed:
    // the CPU-computed rows are complete in host dst memory)
    for (int r = 0; r < (int)re.n_rows; r++) {
        if (re.hit_mask & (1ull << r)) continue;
        memcpy(img + (size_t)r * re.nb1,
               (const char *)host_dst_data + (size_t)r * re.nb1, re.nb1);
        d.redirect_misses_up++;
    }

    // one H2D of the full image on the CONSUMER's own stream, ordered behind
    // the hit-row D2H copies via the event. No host sync anywhere, no P2P.
    ggml_backend_t be = (ggml_backend_t)consumer_backend;
    ggml_backend_cuda_context * cc = (ggml_backend_cuda_context *)be->context;
    ggml_cuda_set_device(cc->device);
    if (!moe_cache_ok(re.dev, cudaStreamWaitEvent(cc->stream(), d.redir_evt[re.par], 0), "redirect wait") ||
        !moe_cache_ok(re.dev, cudaMemcpyAsync(re.gpu_ptr, img, (size_t)re.n_rows * re.nb1,
                               cudaMemcpyHostToDevice, cc->stream()), "redirect H2D")) {
        return 0;   // scheduler performs its normal copy from the host dst
    }

    d.redirect_claims++;
    return 1;
}

// ---- API: fused GLU hook ---------------------------------------------------------------

static unsigned long long moe_cache_glu_hits(const void * src0_data, const void * src1_data,
                                       void * dst_data, size_t dst_nb1, int ith) {
    // learning: observing the GLU node whose inputs are a layer's gate/up MMID
    // dsts proves the fused dispatch is safe for that layer
    if (!g.fuse) return 0;
    if (ith == 0) {
        MOE_CACHE_DBG("[moe-cache-dbg] glu call src0=%p src1=%p\n", src0_data, src1_data);
    }
    auto lit = g.glu_learn.find(src0_data);
    if (lit == g.glu_learn.end()) return 0;
    const int blk = lit->second;
    if (blk >= 0 && blk < 1024 && g.learn_up_dst[blk] == src1_data && !g.safe_fuse_blk[blk]) {
        g.safe_fuse_blk[blk] = true;
        MOE_CACHE_DBG("[moe-cache-dbg] fuse-safe blk=%d\n", blk);
    }

    // active fused rows for this pair? dst buffers are reused across layers,
    // so several devices can hold matching (stale) entries - take the newest
    int best = -1;
    long long best_serial = -1;
    for (int di = 0; di < g.n_dev; di++) {
        auto & f = g.dev[di].fused;
        if (!f.active || f.gate_dst != src0_data || f.up_dst != src1_data) continue;
        if (f.serial > best_serial) { best_serial = f.serial; best = di; }
    }
    {
        const int di = best;
        if (di < 0) return 0;
        auto & f = g.dev[di].fused;
        if (ith == 0 && !f.scattered) {
            moe_cache_device & d = g.dev[di];
            ggml_cuda_set_device(di);
            const bool gok = !d.dead &&
                moe_cache_ok(di, cudaStreamSynchronize(d.compute_stream), "glu sync");   // D2H of fused rows
            for (int i = 0; i < f.n; i++) {
                char * row = (char *)dst_data + (size_t)f.rows[i] * dst_nb1;
                if (gok) memcpy(row, d.h_out + (size_t)i * f.n_out, f.n_out * sizeof(float));
                else     memset(row, 0, f.n_out * sizeof(float));
            }
            f.scattered = true;
        }
        return f.mask;
    }
}

static int moe_cache_glu_hits_w(const void * src0_data, const void * src1_data,
                                void * dst_data, size_t dst_nb1, int ith,
                                uint64_t * mask, int n_words) {
    const unsigned long long m0 = moe_cache_glu_hits(src0_data, src1_data, dst_data, dst_nb1, ith);
    if (mask && n_words > 0) {
        memset(mask, 0, (size_t)n_words * sizeof(uint64_t));
        int best = -1;
        long long best_serial = -1;
        for (int di = 0; di < g.n_dev; di++) {
            auto & f = g.dev[di].fused;
            if (!f.active || f.gate_dst != src0_data || f.up_dst != src1_data) continue;
            if (f.serial > best_serial) { best_serial = f.serial; best = di; }
        }
        if (best >= 0) {
            auto & f = g.dev[best].fused;
            const int nw = n_words < 8 ? n_words : 8;
            for (int w = 0; w < nw; w++) {
                mask[w] = f.mask_w[w];
            }
            return f.n > 0;
        }
        mask[0] = (uint64_t) m0;
    }
    return m0 != 0;
}

// ---- trim: surrender the device's cache VRAM under allocator pressure ------------------
//
// Called from the CUDA backend's pool-alloc OOM retry. Frees every slab and
// scratch buffer on the device and marks its cache dead (conservative: the
// budget decision was clearly wrong for this workload). Returns bytes freed.

extern "C" size_t ggml_moe_cache_trim(int device) {
    if (!g.enabled || device < 0 || device >= g.n_dev) return 0;
    moe_cache_device & d = g.dev[device];
    if (d.n_pools == 0 && !d.d_out) return 0;

    std::unique_lock<std::mutex> lk(g.mu);
    for (auto it = g.queue.begin(); it != g.queue.end(); ) {
        if (it->dev == device) it = g.queue.erase(it); else ++it;
    }
    g.cv_idle.wait(lk, [&]{
        for (int w = 0; w < 16; w++) if (g.inflight_src[w]) return false;
        return true;
    });

    cudaSetDevice(device);
    size_t freed = 0;
    for (int i = 0; i < d.n_pools; i++) {
        moe_cache_pool & p = d.pools[i];
        if (p.slab)  { freed += (size_t)p.n_slots * p.expert_size; cudaFree(p.slab);  p.slab  = nullptr; }
        if (p.slab2) { freed += (size_t)p.n_slots * p.expert_size; cudaFree(p.slab2); p.slab2 = nullptr; }
        p.map.clear();
        p.n_slots = 0;
        p.n_used  = 0;
    }
    if (d.d_ids)    { cudaFree(d.d_ids);          d.d_ids = nullptr;    d.ids_cap = 0; }
    if (d.d_act)    { cudaFree(d.d_act);          d.d_act = nullptr;    d.act_cap = 0; }
    if (d.d_act_q8) { cudaFree(d.d_act_q8);       d.d_act_q8 = nullptr; d.act_q8_cap = 0; }
    if (d.d_out)    { freed += d.d_out_cap; cudaFree(d.d_out); d.d_out = nullptr; d.d_out_cap = 0; }
    cudaGetLastError();
    d.out_rows     = 0;
    d.fused.active = false;
    memset(g.resident_pair, 0, sizeof(g.resident_pair));
    memset(g.resident_down, 0, sizeof(g.resident_down));
    d.dead = true;
    MOE_CACHE_LOG("[moe-cache] dev=%d TRIMMED %zu MB under VRAM pressure - cache off on this device\n",
            device, freed >> 20);
    return freed;
}

// ---- API: invalidate (host weight buffer teardown) -------------------------------------
//
// Called when a host buffer is freed (model unload). Queued insert jobs whose
// source lies in the range are dropped; a worker mid-copy from the range is
// waited out; per-blk tensor-base learning that points into the range is reset.
// Cached slots become unreachable automatically (keys mix the host base) and
// are reclaimed by LRU.

static void moe_cache_invalidate(const void * base, size_t size) {
    if (!g.enabled) return;
    const char * lo = (const char *)base;
    const char * hi = lo + size;
    auto in_range = [&](const void * p) {
        return p && (const char *)p >= lo && (const char *)p < hi;
    };

    std::unique_lock<std::mutex> lk(g.mu);
    for (auto it = g.queue.begin(); it != g.queue.end(); ) {
        if (in_range(it->src) || in_range(it->src_gate)) {
            // orphan the slot bookkeeping (entry stays queued=true and is
            // skipped by eviction until overwritten; safe and rare)
            it = g.queue.erase(it);
        } else {
            ++it;
        }
    }
    g.cv_idle.wait(lk, [&]{
        for (int w = 0; w < 16; w++) {
            const char * s = (const char *)g.inflight_src[w];
            if (s && s + g.inflight_len[w] > lo && s < hi) return false;
        }
        return true;
    });
    for (int b = 0; b < 1024; b++) {
        if (in_range(g.role_base[0][b])) g.role_base[0][b] = nullptr;
        if (in_range(g.role_base[1][b])) g.role_base[1][b] = nullptr;
    }
}

// ---- API: node wall-time samples (bail-out) ---------------------------------------------

static void moe_cache_node_time(int code, int64_t us) {
    if (g.bail.tripped) return;
    auto & b = g.bail;
    if (code == -3) {
        b.base_ewma = b.base_n == 0 ? (double)us : b.base_ewma + ((double)us - b.base_ewma) / 256.0;
        b.base_n++;
        return;
    }
    if (code < 0) return;
    b.on_ewma = b.on_n == 0 ? (double)us : b.on_ewma + ((double)us - b.on_ewma) / 256.0;
    b.on_n++;
    if (b.base_n < 1000 || b.on_n < 5000) return;
    if (b.on_n % 256 != 0) return;
    if (b.on_ewma > b.base_ewma * 1.05) {
        if (++b.strikes >= 4) {
            b.tripped = true;
            MOE_CACHE_LOG("[moe-cache] bail-out: cache-engaged nodes average %.0fus vs %.0fus pure-CPU - "
                    "disabling the cache and freeing its VRAM for this run\n",
                    b.on_ewma, b.base_ewma);
            for (int di = 0; di < g.n_dev; di++) {
                ggml_moe_cache_trim(di);
            }
            g.enabled = false;
        }
    } else {
        b.strikes = 0;
    }
}


static void moe_cache_miss_time(int dev, int64_t us) {
    if (dev < 0 || dev >= g.n_dev) return;
    moe_cache_device & d = g.dev[dev];
    d.t_miss_wall_us += us;
    if (g.io_stats_every <= 0 || d.n_nodes == 0 || d.n_nodes % g.io_stats_every != 0) return;

    const moe_cache_io_sample io_now = moe_cache_sample_io();
    std::lock_guard<std::mutex> lk(g.mu);
    const long long prompt_rows = d.prompt_hits + d.prompt_misses;
    const long long decode_rows = d.decode_hits + d.decode_misses;
    MOE_CACHE_LOG("[moe-cache-io] dev=%d prompt_nodes=%lld prompt_hits=%lld/%lld(%.1f%%) decode_nodes=%lld decode_hits=%lld/%lld(%.1f%%) cpu_miss_wall_ms=%.1f dispatch_h2d_mb=%.1f insert_stage_ms=%.1f insert_h2d_ms=%.1f insert_h2d_mb=%.1f process_read_mb=%.1f read_ops=%llu page_faults=%llu\n",
            dev,
            d.prompt_nodes, d.prompt_hits, prompt_rows, prompt_rows ? 100.0 * d.prompt_hits / prompt_rows : 0.0,
            d.decode_nodes, d.decode_hits, decode_rows, decode_rows ? 100.0 * d.decode_hits / decode_rows : 0.0,
            d.t_miss_wall_us / 1000.0, d.dispatch_h2d_bytes / 1048576.0,
            d.insert_stage_us / 1000.0, d.insert_h2d_us / 1000.0, d.insert_h2d_bytes / 1048576.0,
            (io_now.read_bytes - g_io_base.read_bytes) / 1048576.0,
            (unsigned long long)(io_now.read_ops - g_io_base.read_ops),
            (unsigned long long)(io_now.page_faults - g_io_base.page_faults));
}

// ---- API: stats ----------------------------------------------------------------------

static void moe_cache_stats(void) {
    for (int i = 0; i < g.n_dev; i++) {
        moe_cache_device & d = g.dev[i];
        if (!d.compute_stream) continue;
        const long long tot = d.hits + d.misses;
        int used = 0, slots = 0;
        for (int pi = 0; pi < d.n_pools; pi++) { used += d.pools[pi].n_used; slots += d.pools[pi].n_slots; }
        MOE_CACHE_LOG("[moe-cache] dev=%d hits=%lld/%lld (%.1f%%) inserts=%lld evict=%lld skip=%lld queued-miss=%lld used=%d/%d q=%zu\n",
                i, d.hits, tot, tot ? 100.0 * d.hits / tot : 0.0,
                d.inserts, d.evictions, d.insert_skips, d.queued_misses,
                used, slots, g.queue.size());
        MOE_CACHE_LOG("[moe-cache] dev=%d decomp: compulsory=%lld capacity=%lld admission=%lld inflight=%lld uniq-seen=%zu uniq-inserted=%zu | skips: throttle=%lld budget=%lld qfull=%lld lru=%lld\n",
                i, d.miss_compulsory, d.miss_capacity, d.miss_admission, d.queued_misses,
                d.ever_seen.size(), d.ever_inserted.size(),
                d.skip_throttle, d.skip_budget, d.skip_qfull, d.skip_lrubusy);
        for (int pi = 0; pi < d.n_pools; pi++) {
            const long long ptot = d.pool_hits[pi] + d.pool_miss[pi];
            MOE_CACHE_LOG("[moe-cache] dev=%d pool[%d]: hits=%lld/%lld (%.1f%%) slots=%d slot=%zuKB\n",
                    i, pi, d.pool_hits[pi], ptot, ptot ? 100.0 * d.pool_hits[pi] / ptot : 0.0,
                    d.pools[pi].n_slots, d.pools[pi].expert_size >> 10);
        }
        if (d.n_nodes > 0) {
            MOE_CACHE_LOG("[moe-cache] dev=%d timing: nodes=%lld plan=%.1fus disp=%.1fus coll=%.1fus per-node total=%.1fus\n",
                    i, d.n_nodes,
                    (double)d.t_plan_us / d.n_nodes, (double)d.t_disp_us / d.n_nodes,
                    (double)d.t_coll_us / d.n_nodes,
                    (double)(d.t_plan_us + d.t_disp_us + d.t_coll_us) / d.n_nodes);
            MOE_CACHE_LOG("[moe-cache] dev=%d redirect: claims=%lld miss-rows-up=%lld fused-layers=%lld\n",
                    i, d.redirect_claims, d.redirect_misses_up, d.fused_layers);
            if (i == 0) {
                MOE_CACHE_LOG("[moe-cache] bail-ewma: base=%.1fus(n=%lld) on=%.1fus(n=%lld)\n",
                        g.bail.base_ewma, g.bail.base_n, g.bail.on_ewma, g.bail.on_n);
            }
        }
    }
}

// ---- self-test ---------------------------------------------------------------------------
//
// GGML_CUDA_MOE_CACHE_SELFTEST=1 runs at registration: builds a synthetic quantized pool,
// runs the full plan-free dispatch+collect path, and compares against a host
// reference matvec on dequantized weights. No model required - this validates
// the batched mmvq stride mapping and the staging logic in seconds.

static bool moe_cache_selftest_one(int di, ggml_type wtype, int64_t n_in, int64_t n_out,
                             int n_hits, bool shared_act) {
    const int n_slots = 16;
    const size_t row_bytes  = ggml_row_size(wtype, n_in);
    const size_t slot_bytes = (size_t)n_out * row_bytes;

    // build random fp32 weights, quantize per slot
    std::vector<float> wf((size_t)n_slots * n_out * n_in);
    for (size_t i = 0; i < wf.size(); i++) wf[i] = 0.02f * (float)((int)(i * 2654435761u % 1000) - 500) / 500.0f;
    std::vector<char> wq((size_t)n_slots * slot_bytes);
    for (int s = 0; s < n_slots; s++) {
        ggml_quantize_chunk(wtype, wf.data() + (size_t)s * n_out * n_in,
                            wq.data() + (size_t)s * slot_bytes, 0, n_out, n_in, nullptr);
    }

    // upload pool
    ggml_cuda_set_device(di);
    char * d_pool = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_pool, wq.size()));
    CUDA_CHECK(cudaMemcpy(d_pool, wq.data(), wq.size(), cudaMemcpyHostToDevice));

    // fabricate device + pool state
    moe_cache_device & d = g.dev[di];
    if (!d.compute_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&d.compute_stream, cudaStreamNonBlocking));
    }
    moe_cache_pool & p = d.pools[0];
    p.expert_size = slot_bytes;
    p.wtype = wtype;
    p.slab = d_pool;
    p.n_slots = n_slots;
    g.cur_pool = 0;

    // activations
    std::vector<float> act((size_t)n_hits * n_in);
    for (size_t i = 0; i < act.size(); i++) act[i] = 0.05f * (float)((int)(i * 40503u % 997) - 498) / 498.0f;

    int32_t slot_ids[64];
    const float * act_rows[64];
    for (int i = 0; i < n_hits; i++) {
        slot_ids[i] = (i * 5 + 3) % n_slots;
        act_rows[i] = shared_act ? act.data() : act.data() + (size_t)i * n_in;
    }

    moe_cache_dispatch(di, (int)wtype, n_in, n_out, n_hits, slot_ids, act_rows);

    std::vector<float> out((size_t)n_hits * n_out, -12345.0f);
    float * out_rows[64];
    for (int i = 0; i < n_hits; i++) out_rows[i] = out.data() + (size_t)i * n_out;
    moe_cache_collect(di, n_hits, out_rows, n_out);

    // host reference: dequantize slot rows, dot with fp32 act (mmvq quantizes the
    // activation to q8_1, so allow a tolerance)
    const ggml_type_traits * tr = ggml_get_type_traits(wtype);
    std::vector<float> wrow(n_in);
    double max_rel = 0.0;
    for (int i = 0; i < n_hits; i++) {
        const char * slot = wq.data() + (size_t)slot_ids[i] * slot_bytes;
        const float * a = act_rows[i];
        for (int r = 0; r < n_out; r += 37) {   // sample rows
            tr->to_float(slot + (size_t)r * row_bytes, wrow.data(), n_in);
            double ref = 0.0;
            for (int64_t c = 0; c < n_in; c++) ref += (double)wrow[c] * a[c];
            const double got = out[(size_t)i * n_out + r];
            const double rel = fabs(got - ref) / (fabs(ref) + 1e-3);
            if (rel > max_rel) max_rel = rel;
        }
    }

    // latency micro-benchmark: 200 dispatch+collect cycles
    cudaStreamSynchronize(d.compute_stream);
    const int reps = 200;
    int64_t t0 = ggml_time_us();
    for (int r = 0; r < reps; r++) {
        moe_cache_dispatch(di, (int)wtype, n_in, n_out, n_hits, slot_ids, act_rows);
        moe_cache_collect(di, n_hits, out_rows, n_out);
    }
    const double us_per_node = (double)(ggml_time_us() - t0) / reps;

    // tolerance: mmvq quantizes the activation to q8_1 while the reference uses
    // fp32, so a few percent of relative error on near-zero outputs is expected
    const bool ok = max_rel < 0.10;
    MOE_CACHE_LOG("[moe-cache-selftest] dev=%d type=%s n_in=%lld n_out=%lld hits=%d %s: max_rel=%.4f %s | %.1f us/node\n",
            di, ggml_type_name(wtype), (long long)n_in, (long long)n_out, n_hits,
            shared_act ? "shared-act" : "multi-act", max_rel, ok ? "OK" : "FAIL", us_per_node);

    cudaFree(d_pool);
    p = moe_cache_pool{};
    return ok;
}

static void moe_cache_selftest(void) {
    bool all = true;
    all &= moe_cache_selftest_one(0, GGML_TYPE_Q4_K, 2048, 768,  8, true);
    all &= moe_cache_selftest_one(0, GGML_TYPE_Q4_K, 768,  2048, 8, false);
    all &= moe_cache_selftest_one(0, GGML_TYPE_Q4_K, 2048, 768,  1, true);
    all &= moe_cache_selftest_one(0, GGML_TYPE_Q6_K, 2048, 768,  5, true);
    all &= moe_cache_selftest_one(0, GGML_TYPE_Q6_K, 512,  2048, 8, false);
    MOE_CACHE_LOG("[moe-cache-selftest] %s\n", all ? "ALL PASS" : "FAILURES PRESENT");
}

// ---- registration ----------------------------------------------------------------------

void ggml_moe_cache_register(void) {
    // always-on auto mode: active unless explicitly disabled. Engagement is
    // still gated per model (expert size, type allowlist, census) and a
    // baseline-sampled bail-out disables the cache if it ever measures itself
    // losing on this workload.
    const char * v = getenv("GGML_CUDA_MOE_CACHE");
    // ported into the hot-resident fork as OPT-IN (upstream default was on):
    // enable only with GGML_CUDA_MOE_CACHE=1 so existing benchmarks/workflows
    // (hot-resident, dual-chain, spec runs) are bit-identical by default
    g.enabled = (v && atoi(v) > 0);
    if (!g.enabled) return;

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count <= 0) { g.enabled = false; return; }

    g.n_dev = dev_count > MOE_CACHE_MAX_DEV ? MOE_CACHE_MAX_DEV : dev_count;
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_NDEV"))      { int n = atoi(e); if (n > 0 && n < g.n_dev) g.n_dev = n; }
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_BUDGET_MB")) g.budget_mb = (size_t)atoll(e);
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_INSERTS"))   g.inserts_per_plan = atoi(e);
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_THROTTLE"))  { g.throttle_mod = atoi(e); if (g.throttle_mod < 1) g.throttle_mod = 1; }
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_WORKERS"))   { int n = atoi(e); if (n > 0 && n <= 16) g.n_workers = n; }
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_STATS"))     g.stats_every = atoi(e);
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB")) g.min_expert_bytes = (size_t)atoll(e) << 10;
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_RESERVE_MB"))    g.reserve_mb = (size_t)atoll(e);
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_REUSE"))         g.reuse = atoi(e) > 0;
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_FUSE"))          g.fuse = atoi(e) > 0;
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_MAX_BATCH")) {
        const int n = atoi(e);
        if (n == 1 || n == 8 || n == 16 || n == MOE_CACHE_MAX_BATCH_TOKENS) g.max_batch = n;
    }
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_MAX_BLOCK")) {
        const int n = atoi(e);
        if (n >= -1 && n < 1024) g.max_block = n;
    }
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_PREFETCH"))      g.backfill.enabled = atoi(e) > 0;
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_HOT_ONLY"))      g.backfill.hot_only = atoi(e) > 0;
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_HOTSET"))        g.hotset_enabled = atoi(e) > 0;
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_HOTSET_SAVE_SEC")) {
        const int n = atoi(e);
        if (n > 0) g.hotset_save_sec = n;
    }
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_HOTSET_DECAY_PLANS")) {
        const long long n = atoll(e);
        if (n > 0) g.hot_score_decay_plans = (uint64_t)n;
    }
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_IO_STATS")) {
        const int n = atoi(e);
        if (n > 0) g.io_stats_every = n;
    }
    // Registration runs while CUDA backends are enumerated, before ggml_time_init()
    // on Windows. Calling ggml_time_us() here divides by an uninitialized QPC
    // frequency. The zero default preserves the intended delay because the first
    // worker tick happens after timer initialization and sees < 90 seconds.
    g.hotset_last_save = 0;
    memset(g.blk_pair_pool, -1, sizeof(g.blk_pair_pool));
    memset(g.blk_down_pool, -1, sizeof(g.blk_down_pool));

    ggml_moe_cache.begin    = moe_cache_begin;
    ggml_moe_cache.plan     = moe_cache_plan;
    ggml_moe_cache.dispatch = moe_cache_dispatch;
    ggml_moe_cache.collect  = moe_cache_collect;
    ggml_moe_cache.stats    = moe_cache_stats;
    ggml_moe_cache.redirect_offer    = moe_cache_redirect_offer;
    ggml_moe_cache.redirect_finalize = moe_cache_redirect_finalize;
    ggml_moe_cache.glu_hits          = moe_cache_glu_hits;
    ggml_moe_cache.invalidate        = moe_cache_invalidate;
    ggml_moe_cache.node_time         = moe_cache_node_time;
    ggml_moe_cache.miss_time         = moe_cache_miss_time;
    ggml_moe_cache.glu_hits_w        = moe_cache_glu_hits_w;
    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_REDIRECT")) g.redirect_on = atoi(e) > 0;

    MOE_CACHE_LOG("[moe-cache] enabled: n_dev=%d budget=%s inserts/plan=%d workers=%d stats_every=%d max_batch=%d max_block=%d\n",
            g.n_dev, g.budget_mb ? "env" : "auto-70%-free", g.inserts_per_plan,
            g.n_workers, g.stats_every, g.max_batch, g.max_block);

    if (const char * e = getenv("GGML_CUDA_MOE_CACHE_SELFTEST"); e && atoi(e) > 0) {
        moe_cache_selftest();
    }
}
