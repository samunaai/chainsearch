import os
import sys
import re
import json
import shutil
import torch
import multiprocessing as mp
from collections import Counter
from tqdm.auto import tqdm

# Line-buffered logs when redirected with:
#   python generate_pretrain_sample_list.py &> out_setminus.txt
if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except AttributeError:
        pass

ASTAR_PATH = "/workspace/sam/minireason_archives/data_pts/astar/astar_data_NALL_S6-6_T{6:100-999}_exp2500_trace10000_sw8_stick0_compact0_useh20_h2all0_h2k1.0_seed42_fmtv2.pt"
HR3_PATH   = "/workspace/sam/minireason_archives/data_pts/colton_h3/countdown_colton_h3_full"
OUT_DIR    = "/workspace/sam/minireason_archives/data_pts/colton_h3/hr3_samples_setminus_astar_sharded"
SHARD_SIZE = 100_000

# Use more than one CPU, but keep it gentle. This is mostly I/O + filtering,
# not the heavy HR3 solver itself.
WORKERS = int(os.environ.get("SETMINUS_WORKERS", "16"))

# Each task processes this many input HR3 shards. Larger batches reduce
# scheduling overhead; smaller batches give smoother progress updates.
BATCH_SHARDS = int(os.environ.get("SETMINUS_BATCH_SHARDS", "50"))

# Clean OUT_DIR before writing so canceled/restarted runs cannot leave stale
# shards around.
OVERWRITE = True

# Filled in by the parent before forking. On Linux, fork shares this large set
# copy-on-write with workers, avoiding a huge per-task pickle.
ASTAR_KEYS = None

PROMPT_RE = re.compile(r"IN:\s*\[\s*(.*?)\s*\]\s*TGT:\s*(\d+)")

def prompt_to_key(prompt: str):
    m = PROMPT_RE.fullmatch(prompt.strip())
    if not m:
        raise ValueError(f"Bad prompt: {prompt}")
    nums = tuple(int(x.strip()) for x in m.group(1).split(","))
    tgt = int(m.group(2))
    return nums, tgt

def hr3_record_key(r):
    return tuple(r["inputs"]), int(r["target"])

def shard_paths(path):
    if os.path.isdir(path):
        manifest = os.path.join(path, "manifest.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                names = json.load(f)["shards"]
            return [os.path.join(path, name) for name in names]
        return sorted(
            os.path.join(path, fn)
            for fn in os.listdir(path)
            if fn.startswith("shard_") and fn.endswith(".pt")
        )
    return [path]

def iter_hr3(path):
    for p in shard_paths(path):
        data = torch.load(p, weights_only=False, map_location="cpu")
        for record in data:
            yield record

def flush_shard(buffer, out_dir, shard_name):
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, shard_name)
    tmp_path = final_path + ".tmp"
    torch.save(buffer, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield i // n, xs[i:i + n]

def process_hr3_path_batch(task):
    """
    Worker task.

    Reads a batch of HR3 shard paths, removes records whose (inputs, target)
    key appears in ASTAR_KEYS, and writes kept records directly to OUT_DIR.

    Returns only stats + output shard paths, not the kept records themselves.
    """
    task_id, paths = task
    if ASTAR_KEYS is None:
        raise RuntimeError("Worker ASTAR_KEYS was not initialized.")

    buffer = []
    out_shards = []
    local_out_idx = 0

    total_hr3 = 0
    removed = 0
    kept = 0
    removed_scores = Counter()
    kept_scores = Counter()
    removed_key_counts = Counter()

    def flush_local():
        nonlocal buffer, local_out_idx
        if not buffer:
            return
        shard_name = f"batch_{task_id:05d}_shard_{local_out_idx:04d}.pt"
        out_shards.append(flush_shard(buffer, OUT_DIR, shard_name))
        local_out_idx += 1
        buffer = []

    for p in paths:
        data = torch.load(p, weights_only=False, map_location="cpu")
        for r in data:
            total_hr3 += 1
            key = hr3_record_key(r)

            if key in ASTAR_KEYS:
                removed += 1
                removed_scores[r["score"]] += 1
                removed_key_counts[key] += 1
                continue

            kept += 1
            kept_scores[r["score"]] += 1
            buffer.append(r)

            if len(buffer) >= SHARD_SIZE:
                flush_local()

    flush_local()

    return {
        "task_id": task_id,
        "n_input_shards": len(paths),
        "out_shards": out_shards,
        "total_hr3": total_hr3,
        "removed": removed,
        "kept": kept,
        "removed_scores": dict(removed_scores),
        "kept_scores": dict(kept_scores),
        "removed_key_counts": dict(removed_key_counts),
    }

def counter_from_dict(d):
    c = Counter()
    c.update(d)
    return c


def print_run_header() -> None:
    width = 72
    print("=" * width)
    print("HR3 \\ A* set-difference builder".center(width))
    print("=" * width)
    print("Purpose:")
    print("  Build a sharded HR3 dataset that excludes any sample whose")
    print("  (inputs, target) key already appears in the A* dataset.")
    print()
    print("Overlap definition:")
    print("  key = (tuple(inputs), int(target))")
    print("  Traces/expressions are ignored; only the underlying problem identity matters.")
    print()
    print("Configured paths:")
    print(f"  ASTAR_PATH : {ASTAR_PATH}")
    print(f"  HR3_PATH   : {HR3_PATH}")
    print(f"  OUT_DIR    : {OUT_DIR}")
    print(f"  SHARD_SIZE : {SHARD_SIZE:,}")
    print(f"  WORKERS    : {WORKERS:,}")
    print(f"  BATCH_SHARDS: {BATCH_SHARDS:,}")
    print()
    print("Sanity checks that will run:")
    print("  1) bookkeeping: kept + removed == total_hr3")
    print("  2) overlap count: removed == number of unique A* keys")
    print("  3) duplicate overlap keys in HR3: warn if any removed key appears >1 time")
    print("  4) removed-score check: warn if any removed HR3 samples have score != 10")
    print("=" * width)
    print()

# -------------------------
# 1) Load A* prompt keys
# -------------------------
print_run_header()
print("Loading A* dataset and building A* key set...", flush=True)
astar = torch.load(ASTAR_PATH, map_location="cpu", weights_only=False)
prompts = astar["prompts"]
astar_keys = {prompt_to_key(p) for p in prompts}
ASTAR_KEYS = astar_keys

print(f"A* total prompts : {len(prompts):,}")
print(f"A* unique keys   : {len(astar_keys):,}")
print(f"A* duplicates    : {len(prompts) - len(astar_keys):,}")
print(flush=True)

# -------------------------
# 2) Stream HR3 and filter
# -------------------------
if OVERWRITE and os.path.exists(OUT_DIR):
    print(f"Removing existing output directory: {OUT_DIR}", flush=True)
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

hr3_paths = shard_paths(HR3_PATH)
tasks = list(chunked(hr3_paths, BATCH_SHARDS))
print(f"HR3 input shards : {len(hr3_paths):,}")
print(f"Task batches     : {len(tasks):,}")
print(f"Starting HR3 filtering with {WORKERS} worker(s)...", flush=True)

out_shards = []

total_hr3 = 0
removed = 0
kept = 0
removed_key_counts = Counter()

removed_scores = Counter()
kept_scores = Counter()

def absorb_result(res):
    global total_hr3, removed, kept
    total_hr3 += int(res["total_hr3"])
    removed += int(res["removed"])
    kept += int(res["kept"])
    out_shards.extend(res["out_shards"])
    removed_scores.update(counter_from_dict(res["removed_scores"]))
    kept_scores.update(counter_from_dict(res["kept_scores"]))
    removed_key_counts.update(counter_from_dict(res["removed_key_counts"]))

if WORKERS <= 1:
    iterator = map(process_hr3_path_batch, tasks)
    pbar = tqdm(iterator, total=len(tasks), desc="HR3 shard batches", dynamic_ncols=True)
    for res in pbar:
        absorb_result(res)
        pbar.set_postfix(
            kept=f"{kept:,}",
            removed=f"{removed:,}",
            total=f"{total_hr3:,}",
        )
else:
    # Linux/container path: fork shares ASTAR_KEYS copy-on-write with workers.
    # This avoids pickling a 1M-key set into every task.
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("[warning] fork context unavailable; falling back to single-process mode.", flush=True)
        iterator = map(process_hr3_path_batch, tasks)
        pbar = tqdm(iterator, total=len(tasks), desc="HR3 shard batches", dynamic_ncols=True)
        for res in pbar:
            absorb_result(res)
            pbar.set_postfix(kept=f"{kept:,}", removed=f"{removed:,}", total=f"{total_hr3:,}")
    else:
        with ctx.Pool(processes=WORKERS, maxtasksperchild=25) as pool:
            iterator = pool.imap_unordered(process_hr3_path_batch, tasks, chunksize=1)
            pbar = tqdm(iterator, total=len(tasks), desc="HR3 shard batches", dynamic_ncols=True)
            for res in pbar:
                absorb_result(res)
                pbar.set_postfix(
                    kept=f"{kept:,}",
                    removed=f"{removed:,}",
                    total=f"{total_hr3:,}",
                )

manifest_path = os.path.join(OUT_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump({"shards": sorted(os.path.basename(p) for p in out_shards)}, f, indent=2)

# -------------------------
# 3) Report
# -------------------------
print(f"\nHR3 total              : {total_hr3:,}")
print(f"Removed by A* overlap  : {removed:,}")
print(f"Kept in setminus       : {kept:,}")

if kept + removed != total_hr3:
    raise RuntimeError(
        f"Bookkeeping error: kept({kept:,}) + removed({removed:,}) != total_hr3({total_hr3:,})"
    )
print(f"[sanity] kept + removed = total_hr3 = {total_hr3:,}")

expected_removed = len(astar_keys)
if removed == expected_removed:
    print(f"[sanity] Removed count matches unique A* keys: {expected_removed:,}")
else:
    print(
        f"[warning] Removed {removed:,} records, but A* has "
        f"{expected_removed:,} unique (inputs,target) keys."
    )
    print("[warning] This suggests incomplete overlap, duplicated HR3 keys, or a key-format mismatch.")

print("\nRemoved score distribution:")
for sc in [10, 7, 5, 0]:
    print(f"  score {sc:>2}: {removed_scores[sc]:>10,}")

duplicate_removed_keys = sum(1 for c in removed_key_counts.values() if c > 1)
max_removed_multiplicity = max(removed_key_counts.values(), default=0)
if duplicate_removed_keys == 0:
    print("[sanity] No duplicated overlap keys were seen in HR3.")
else:
    print(
        f"[warning] {duplicate_removed_keys:,} overlap keys appeared multiple times in HR3 "
        f"(max multiplicity={max_removed_multiplicity})."
    )

removed_non10 = removed_scores[7] + removed_scores[5] + removed_scores[0]
if removed_non10 == 0:
    print("[sanity] All removed overlaps were score-10 HR3 samples.")
else:
    print(
        f"[warning] Removed {removed_non10:,} non-10-point HR3 samples "
        f"(7:{removed_scores[7]:,}, 5:{removed_scores[5]:,}, 0:{removed_scores[0]:,})."
    )
    print("[warning] That is unexpected for exact-solution A* prompts; inspect a few examples.")


print("\nKept score distribution:")
for sc in [10, 7, 5, 0]:
    print(f"  score {sc:>2}: {kept_scores[sc]:>10,}")

print(f"\nSaved sharded output to: {OUT_DIR}")
print(f"Manifest: {manifest_path}")