// GPU-resident harness wavefront stages (spec 06; review F2). Mirrors
// agcad_harness::wavefront CPU references exactly (AG-06-006).
#include "ag_common.cuh"

#define CONFIG_ALL 0xFFFFFFFFu

struct CommandRecord { unsigned short opcode, flags; u32 configuration, dst, args[6], expected_revision, configuration_mask_slot, reserved; double imm[2]; };
struct StatusRecord { unsigned short status, flags; u32 dst, budget_used, aux; };

struct ValidateParams { u64 cmds, status, allow, budget, revisions, keys; u32 n, configs; };
struct BinBoundsParams { u64 sorted_keys, bin_starts, bin_count; u32 n, max_bins; };
struct DecideParams { u64 trial_flags, current, trial, accepted, status; u32 configs, required_flags; };
struct MeterParams { u64 used, remaining, active, status; u32 configs, _pad; };
struct MaskScoresParams { u64 scores, active, flags, masked; u32 configs, required_flags; };

AG_KERNEL void ag_hv_validate(const ValidateParams* p) {
    u32 i = AG_TID;
    if (i >= p->n) return;
    const CommandRecord c = ag_ptr<const CommandRecord>(p->cmds)[i];
    const u32* allow = ag_ptr<const u32>(p->allow);
    const u32* budget = ag_ptr<const u32>(p->budget);
    const u32* rev = ag_ptr<const u32>(p->revisions);
    unsigned short st = AG_ACCEPTED;
    if (((allow[c.opcode >> 5] >> (c.opcode & 31)) & 1u) == 0u) st = AG_INVALID_OPCODE;
    else if (c.configuration != CONFIG_ALL && c.configuration >= p->configs) st = AG_INVALID_PARAMETER;
    else {
        u32 cfg = (c.configuration == CONFIG_ALL) ? 0u : c.configuration;
        if (rev[cfg] != c.expected_revision) st = AG_STALE_REVISION;
        else if (budget[cfg] == 0u) st = AG_BUDGET_EXHAUSTED;
        else if (!isfinite(c.imm[0]) || !isfinite(c.imm[1])) st = AG_NONFINITE_PARAMETER;
    }
    StatusRecord s; s.status = st; s.flags = 0; s.dst = c.dst; s.budget_used = (st == AG_ACCEPTED) ? 1u : 0u; s.aux = 0;
    ag_ptr<StatusRecord>(p->status)[i] = s;
    ag_ptr<u64>(p->keys)[i] = (st == AG_ACCEPTED) ? (((u64)c.opcode << 32) | (u64)i) : 0xFFFFFFFFFFFFFFFFull;
}

// One thread per sorted key: a bin starts where the opcode changes. Bins are written in
// order by a deterministic scan over the "is start" flags (second pass on the CPU reference).
AG_KERNEL void ag_hv_bin_bounds(const BinBoundsParams* p) {
    u32 i = AG_TID;
    if (i >= p->n) return;
    const u64* k = ag_ptr<const u64>(p->sorted_keys);
    if (k[i] == 0xFFFFFFFFFFFFFFFFull) return;
    bool start = (i == 0) || ((k[i] >> 32) != (k[i - 1] >> 32));
    if (!start) return;
    // count starts before i (deterministic; O(n) per start is fine for ≤ 64k records/round)
    u32 rank = 0;
    for (u32 j = 1; j <= i; ++j) if (k[j] != 0xFFFFFFFFFFFFFFFFull && (k[j] >> 32) != (k[j - 1] >> 32)) ++rank;
    if (rank < p->max_bins) ag_ptr<u32>(p->bin_starts)[rank] = i;
    {
        u32 total = 0;
        for (u32 j = 0; j < p->n; ++j) if (k[j] != 0xFFFFFFFFFFFFFFFFull && (j == 0 || (k[j] >> 32) != (k[j - 1] >> 32))) ++total;
        if (rank == 0) ag_ptr<u32>(p->bin_count)[0] = total;
    }
}

AG_KERNEL void ag_hv_decide(const DecideParams* p) {
    u32 c = AG_TID;
    if (c >= p->configs) return;
    u32 f = ag_ptr<const u32>(p->trial_flags)[c];
    u32* cur = ag_ptr<u32>(p->current);
    u32* tri = ag_ptr<u32>(p->trial);
    bool ok = (f & p->required_flags) == p->required_flags;
    ag_ptr<u32>(p->accepted)[c] = ok ? 1u : 0u;
    if (ok) { u32 t = cur[c]; cur[c] = tri[c]; tri[c] = t; }
    ag_ptr<unsigned short>(p->status)[c] = ok ? AG_ACCEPTED : ((f & 1u) == 0u ? AG_NUMERICAL_CHECK_FAILED : AG_REQUIREMENT_FAILED);
}

AG_KERNEL void ag_hv_meter(const MeterParams* p) {
    u32 c = AG_TID;
    if (c >= p->configs) return;
    u32 used = ag_ptr<const u32>(p->used)[c];
    u32* rem = ag_ptr<u32>(p->remaining);
    u32 r = rem[c];
    r = (used >= r) ? 0u : r - used;
    rem[c] = r;
    if (r == 0u) { ag_ptr<u32>(p->active)[c] = 0u; ag_ptr<unsigned short>(p->status)[c] = AG_BUDGET_EXHAUSTED; }
}

// Inactive or failing configurations rank last (+inf) without changing order among ties.
AG_KERNEL void ag_hv_mask_scores(const MaskScoresParams* p) {
    u32 c = AG_TID;
    if (c >= p->configs) return;
    bool ok = ag_ptr<const u32>(p->active)[c] != 0u && ((ag_ptr<const u32>(p->flags)[c] & p->required_flags) == p->required_flags);
    ag_ptr<double>(p->masked)[c] = ok ? ag_ptr<const double>(p->scores)[c] : __longlong_as_double(0x7FF0000000000000LL);
}
