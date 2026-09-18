#!/usr/bin/env python3
"""
quick_check_hr3_astar_no_overlap.py

Check that the HR3 \\ A* set-minus bundle contains no examples whose
(inputs, target) key appears in the A* dataset.

This belongs after:
    python generate_pretrain_sample_list.py

and before:
    python generate_final_pretrain_traces.py
"""

import os
import re
import json
import torch
from collections import Counter

ASTAR_PATH = (
    "/workspace/sam/minireason_archives/data_pts/astar/"
    "astar_data_NALL_S6-6_T{6:100-999}_exp2500_trace10000_sw8_stick0_"
    "compact0_useh20_h2all0_h2k1.0_seed42_fmtv2.pt"
)

HR3_SETMINUS_PATH = (
    "/workspace/sam/minireason_archives/data_pts/colton_h3/"
    "hr3_samples_setminus_astar_sharded"
)

PROMPT_RE = re.compile(r"IN:\s*\[\s*(.*?)\s*\]\s*TGT:\s*(\d+)")

def prompt_to_key(prompt: str):
    m = PROMPT_RE.fullmatch(prompt.strip())
    if not m:
        raise ValueError(f"Bad prompt: {prompt!r}")
    nums = tuple(int(x.strip()) for x in m.group(1).split(",") if x.strip())
    tgt = int(m.group(2))
    return nums, tgt

def hr3_record_key(record: dict):
    return tuple(int(x) for x in record["inputs"]), int(record["target"])

def shard_paths(path: str):
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

def iter_hr3(path: str):
    for p in shard_paths(path):
        data = torch.load(p, weights_only=False, map_location="cpu")
        if not isinstance(data, list):
            raise ValueError(f"Expected list[dict] in {p}, got {type(data)}")
        for record in data:
            yield record

print("=" * 80)
print("HR3 \\ A* SET-MINUS OVERLAP CHECK")
print("=" * 80)
print(f"A* path:          {ASTAR_PATH}")
print(f"HR3 setminus path:{HR3_SETMINUS_PATH}")
print()

astar = torch.load(ASTAR_PATH, map_location="cpu", weights_only=False)
astar_prompts = astar["prompts"]
astar_keys = {prompt_to_key(p) for p in astar_prompts}

print(f"A* prompts:       {len(astar_prompts):,}")
print(f"A* unique keys:   {len(astar_keys):,}")
print(f"A* duplicates:    {len(astar_prompts) - len(astar_keys):,}")

total = 0
overlap = 0
score_counts = Counter()
overlap_score_counts = Counter()
examples = []

for record in iter_hr3(HR3_SETMINUS_PATH):
    total += 1
    score_counts[int(record.get("score", -1))] += 1
    key = hr3_record_key(record)
    if key in astar_keys:
        overlap += 1
        overlap_score_counts[int(record.get("score", -1))] += 1
        if len(examples) < 10:
            examples.append(record)

print()
print(f"HR3 setminus records: {total:,}")
print(f"Overlap with A*:      {overlap:,}")
print(f"No overlap?           {overlap == 0}")
print()

print("HR3 setminus score distribution:")
for sc in sorted(score_counts):
    print(f"  score {sc:>2}: {score_counts[sc]:>12,}")

if overlap:
    print()
    print("Overlap score distribution:")
    for sc in sorted(overlap_score_counts):
        print(f"  score {sc:>2}: {overlap_score_counts[sc]:>12,}")

    print()
    print("Example leaked records:")
    for r in examples:
        print(f"  inputs={tuple(r['inputs'])} target={r['target']} score={r.get('score')} expr={r.get('expr')}")
    raise SystemExit("FAILED: HR3 setminus still overlaps with A*.")

print()
print("PASS: HR3 setminus has no exact (inputs, target) overlap with A*.")