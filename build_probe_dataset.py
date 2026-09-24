#!/usr/bin/env python3
"""
build_probe_dataset.py

Collect every probe-eligible checkpoint from saved A* traces, keep related
six-number input families in the same train/validation/test split, replay each
required trace once, and save labels plus model activations in one artifact.
"""

from __future__ import annotations

import os

os.environ["COUNTDOWN_STAGE"] = "astar"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from tqdm.auto import tqdm

import pretrain_and_sft_model as pm


# =========================
# CONFIG
# =========================

RUN_TAG = "exp34"
RANDOM_SEED = 42

ASTAR_DIR = "/workspace/user/chainsearch_archives/data_pts/astar"
MODEL_DIR = "/workspace/user/chainsearch_archives/model_pts/astar"
PROBING_DIR = os.path.join(ASTAR_DIR, "probing")

TOKEN_CACHE_PATH = os.path.join(
    ASTAR_DIR,
    "tokenisation_cache",
    f"{RUN_TAG}_astar_max2048_drop.pt",
)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"{RUN_TAG}_astar_best.pt")
EVALUATED_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_astar_unused_samples_evaluated_for_probing.pt",
)
LABEL_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_generated_state_solvability_for_probing_samples.pt",
)
DATASET_PATH = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_astar_probe_dataset.pt",
)

ACTIVATION_SHARD_ROOT = os.path.join(
    PROBING_DIR,
    f"{RUN_TAG}_astar_probe_dataset_shards",
)

SPLIT_NAMES = ["train", "val", "test"]
SPLIT_TO_ID = {name: i for i, name in enumerate(SPLIT_NAMES)}
SPLIT_WEIGHTS = {"train": 5, "val": 1, "test": 1}

TASK_NAMES = [
    "operation_reachability",
    "back_reachability",
    "trace_outcome_trajectory",
    "terminal_outcome_controls",
]
TASK_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}

CHECKPOINT_NAMES = [
    "pre_first_operand",
    "first_operand_end",
    "operator",
    "second_operand_end",
    "result_end",
    "resulting_state_semicolon",
    "back_token",
    "restored_state_semicolon",
    "pre_back",
    "success_token",
    "expr_final_parenthesis",
]
CHECKPOINT_TO_ID = {name: i for i, name in enumerate(CHECKPOINT_NAMES)}

TASK_LABEL_NAMES = {
    "operation_reachability": "resulting_state_reachability",
    "back_reachability": "restored_state_reachability",
    "trace_outcome_trajectory": "expr_target_correct",
    "terminal_outcome_controls": "expr_target_correct",
}

SOURCE_TABLE_NAMES = ["operations", "back_transitions", "generated_traces"]
SOURCE_TABLE_TO_ID = {name: i for i, name in enumerate(SOURCE_TABLE_NAMES)}
   
REQUIRE_PARSEABLE_EXPR = True

ACTIVATION_TRACE_BATCH_SIZE = 32
ACTIVATION_DTYPE = torch.float16
ACTIVATION_SHARD_SIZE = 200_000
ACTIVATION_AUDIT_CHUNK_SIZE = 25_000

POSITION_KEY_BASE = 65_536

# =========================
# BASIC HELPERS
# =========================


def safe_torch_save(obj: Any, path: str) -> None:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    try:
        with open(tmp_path, "wb") as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    except BaseException as exc:
        print(
            f"\nSAVE FAILED for {final_path}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        print(
            f"Partial output may remain at {tmp_path}",
            flush=True,
        )
        raise


def save_activation_shards(
    attention: torch.Tensor,
    mlp: torch.Tensor,
    residual: torch.Tensor,
    final: torch.Tensor,
    shard_dir: str,
    shard_size: int,
) -> list[dict[str, Any]]:
    if shard_size <= 0:
        raise ValueError("Activation shard size must be positive.")

    n_positions = int(attention.shape[0])
    if (
        int(mlp.shape[0]) != n_positions
        or int(residual.shape[0]) != n_positions
        or int(final.shape[0]) != n_positions
    ):
        raise RuntimeError("Activation tensors have different lengths.")

    output_dir = Path(shard_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    n_shards = (n_positions + shard_size - 1) // shard_size
    manifest: list[dict[str, Any]] = []

    try:
        for shard_idx, start in enumerate(
            tqdm(
                range(0, n_positions, shard_size),
                total=n_shards,
                desc="Saving activation shards",
                dynamic_ncols=True,
            )
        ):
            end = min(start + shard_size, n_positions)
            filename = f"shard_{shard_idx:05d}.pt"
            shard_path = output_dir / filename

            # clone() prevents a slice from serializing the full backing tensor.
            shard_payload = {
                "format_version": 2,
                "start": start,
                "end": end,
                "attention_head_output_pre_projection": (
                    attention[start:end].clone()
                ),
                "mlp_output_before_residual": (
                    mlp[start:end].clone()
                ),
                "residual_stream_after_block": (
                    residual[start:end].clone()
                ),
                "final_layer_normalized_hidden": (
                    final[start:end].clone()
                ),
            }
            safe_torch_save(shard_payload, str(shard_path))
            manifest.append(
                {
                    "file": filename,
                    "start": start,
                    "end": end,
                    "bytes": int(shard_path.stat().st_size),
                }
            )
            del shard_payload

        safe_torch_save(
            {
                "format_version": 2,
                "complete": True,
                "n_positions": n_positions,
                "shard_size": shard_size,
                "attention_shape": tuple(attention.shape),
                "mlp_shape": tuple(mlp.shape),
                "residual_shape": tuple(residual.shape),
                "final_shape": tuple(final.shape),
                "shards": manifest,
            },
            str(output_dir / "manifest.pt"),
        )
    except BaseException as exc:
        print(
            f"\nACTIVATION SHARD SAVE FAILED: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        print(
            f"Partial shard directory retained at {output_dir}",
            flush=True,
        )
        raise

    return manifest

def require_fields(table: dict[str, Any], fields: Iterable[str], name: str) -> None:
    missing = set(fields) - set(table)
    if missing:
        raise KeyError(f"{name} is missing required fields: {sorted(missing)}")


def generated_ids_at(payload: dict[str, Any], record_pos: int) -> list[int]:
    start = int(payload["generated_offsets"][record_pos].item())
    length = int(payload["generated_lengths"][record_pos].item())
    return payload["generated_flat_ids"][start:start + length].to(torch.int64).tolist()


def cache_row_ids(blob: dict[str, Any], idx: int) -> list[int]:
    start = int(blob["offsets"][idx].item())
    length = int(blob["lengths"][idx].item())
    return blob["flat_ids"][start:start + length].to(torch.int64).tolist()


def prompt_ids_from_cache(blob: dict[str, Any], idx: int, tok: pm.Tokenizer) -> list[int]:
    ids = cache_row_ids(blob, idx)
    sep = [i for i, token_id in enumerate(ids) if token_id == tok.sep_id]
    if len(sep) != 1:
        raise ValueError(f"Cache row {idx} has {len(sep)} SEP tokens; expected 1.")
    return ids[:sep[0] + 1]


class TokenContract:
    def __init__(self, tok: pm.Tokenizer):
        self.digits = torch.tensor(
            [tok.tok2id[str(d)] for d in range(10)],
            dtype=torch.int64,
        )
        self.operators = torch.tensor(
            [
                tok.tok2id["+"],
                tok.tok2id["-"],
                tok.tok2id["*"],
                tok.tok2id["/"],
            ],
            dtype=torch.int64,
        )
        self.eq = tok.tok2id["="]
        self.arrow = tok.tok2id["->"]
        self.semi = tok.tok2id[";"]
        self.back = tok.tok2id["BACK"]
        self.success = tok.tok2id["SUCCESS"]
        self.expr = tok.tok2id["EXPR"]
        self.rparen = tok.tok2id[")"]
        self.rbr = tok.tok2id["]"]

def packed_tokens_at(
    evaluated: dict[str, Any],
    record_pos: torch.Tensor,
    generated_idx: torch.Tensor,
) -> torch.Tensor:
    record_pos = record_pos.to(torch.int64)
    generated_idx = generated_idx.to(torch.int64)
    lengths = evaluated["generated_lengths"].to(torch.int64)

    if bool((record_pos < 0).any()) or bool(
        (record_pos >= lengths.numel()).any()
    ):
        raise RuntimeError("A token lookup has an invalid record_pos.")
    if bool((generated_idx < 0).any()) or bool(
        (generated_idx >= lengths[record_pos]).any()
    ):
        raise RuntimeError("A token lookup has an invalid token index.")

    offsets = evaluated["generated_offsets"].to(torch.int64)
    flat_idx = offsets[record_pos] + generated_idx
    return evaluated["generated_flat_ids"][flat_idx].to(torch.int64)



# =========================
# MODEL HELPERS
# =========================


def load_model(
    tok: pm.Tokenizer,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], int]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}.")

    state = checkpoint.get("model_state") or checkpoint.get("state")
    if state is None:
        raise KeyError("Checkpoint has neither 'model_state' nor 'state'.")

    config = dict(pm.HPARAMS)
    config.update(checkpoint.get("config") or {})
    model_max_len = pm.infer_model_max_len_from_checkpoint(CHECKPOINT_PATH)

    model = pm._build_model(
        config,
        tok,
        model_max_len,
        use_grad_ckpt=False,
        dropout=0.0,
    )
    pm._load_model_state(model, state)
    model.to(device)
    model.eval()
    return model, config, model_max_len


def infer_model_dimensions(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> tuple[int, int, int, int]:
    n_layers = len(model.blocks)
    if n_layers <= 0:
        raise RuntimeError("Model has no Transformer blocks.")

    first = model.blocks[0]
    n_heads = int(first.n_heads)
    head_dim = int(first.head_dim)
    d_model = n_heads * head_dim

    for layer_idx, block in enumerate(model.blocks):
        if int(block.n_heads) != n_heads or int(block.head_dim) != head_dim:
            raise RuntimeError(f"Attention dimensions differ in block {layer_idx}.")

    if config.get("d_model") is not None and int(config["d_model"]) != d_model:
        raise RuntimeError("Checkpoint d_model disagrees with block dimensions.")
    if config.get("n_heads") is not None and int(config["n_heads"]) != n_heads:
        raise RuntimeError("Checkpoint n_heads disagrees with block dimensions.")
    return n_layers, d_model, n_heads, head_dim


class ActivationCapture:
    def __init__(self, model: torch.nn.Module):
        self.heads: dict[int, torch.Tensor] = {}
        self.mlps: dict[int, torch.Tensor] = {}
        self.residuals: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        for layer_idx, block in enumerate(model.blocks):
            if not hasattr(block, "out_proj"):
                raise AttributeError(
                    f"Transformer block {layer_idx} has no out_proj module."
                )
            self.handles.append(
                block.out_proj.register_forward_pre_hook(
                    self._head_hook(layer_idx)
                )
            )
            if not hasattr(block, "mlp"):
                raise AttributeError(
                    f"Transformer block {layer_idx} has no mlp module."
                )

            self.handles.append(
                block.mlp.register_forward_hook(
                    self._mlp_hook(layer_idx)
                )
            )
            self.handles.append(
                block.register_forward_hook(self._block_hook(layer_idx))
            )

    def _head_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                raise RuntimeError("Attention output projection received no input.")
            self.heads[layer_idx] = inputs[0]
        return hook

    def _mlp_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            self.mlps[layer_idx] = output

        return hook

    def _block_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            self.residuals[layer_idx] = output
        return hook

    def clear(self) -> None:
        self.heads.clear()
        self.mlps.clear()
        self.residuals.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def gather_positions(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length = tensor.shape[:2]
    clamped = positions.clamp(min=0, max=sequence_length - 1)
    batch_index = torch.arange(batch_size, device=tensor.device).unsqueeze(1)
    gathered = tensor[batch_index, clamped]
    mask_shape = list(valid_mask.shape) + [1] * (gathered.ndim - valid_mask.ndim)
    return gathered.masked_fill(~valid_mask.reshape(mask_shape), 0)


# =========================
# SPLITS AND VALIDATION
# =========================


def canonical_family(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in values))


def split_name_for_family(family: tuple[int, ...]) -> str:
    total = sum(SPLIT_WEIGHTS.values())
    digest = hashlib.blake2b(
        (f"{RANDOM_SEED}|" + ",".join(str(x) for x in family)).encode("utf-8"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, "big") / float(2 ** 64)

    cumulative = 0.0
    for name in SPLIT_NAMES:
        cumulative += SPLIT_WEIGHTS[name] / total
        if value < cumulative:
            return name
    return SPLIT_NAMES[-1]


def build_family_splits(
    input_numbers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    family_to_id: dict[tuple[int, ...], int] = {}
    families: list[tuple[int, ...]] = []
    family_splits: list[int] = []
    trace_family = torch.empty(input_numbers.shape[0], dtype=torch.int32)
    trace_split = torch.empty(input_numbers.shape[0], dtype=torch.int8)

    for record_pos, row in enumerate(input_numbers.tolist()):
        family = canonical_family(row)
        family_id = family_to_id.get(family)
        if family_id is None:
            family_id = len(families)
            family_to_id[family] = family_id
            families.append(family)
            family_splits.append(SPLIT_TO_ID[split_name_for_family(family)])
        trace_family[record_pos] = family_id
        trace_split[record_pos] = family_splits[family_id]

    family_values = torch.tensor(families, dtype=torch.int16)
    family_split = torch.tensor(family_splits, dtype=torch.int8)
    if not torch.equal(trace_split, family_split[trace_family.to(torch.long)]):
        raise RuntimeError("An input-number family crossed probe splits.")
    return trace_split, trace_family, family_values, family_split


def validate_sources(
    evaluated: dict[str, Any],
    labels: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if not bool(evaluated.get("complete", False)):
        raise RuntimeError("Evaluated-generation artifact is not complete.")
    if evaluated.get("index_space") != "astar_tokenisation_cache_rows":
        raise RuntimeError("Unexpected evaluated-generation index space.")
    if not bool(labels.get("complete", False)):
        raise RuntimeError("Reachability-label artifact is not complete.")
    if int(labels.get("format_version", -1)) < 4:
        raise RuntimeError("Reachability-label format version must be at least 4.")

    for value, expected, name in (
        (labels.get("source_evaluated_path"), EVALUATED_PATH, "evaluated artifact"),
        (labels.get("source_token_cache_path"), TOKEN_CACHE_PATH, "token cache"),
        (labels.get("source_checkpoint_path"), CHECKPOINT_PATH, "checkpoint"),
    ):
        if os.path.realpath(str(value)) != os.path.realpath(expected):
            raise RuntimeError(f"Reachability labels point to a different {name}.")

    generated = labels.get("generated_traces")
    operations = labels.get("operations")
    backs = labels.get("back_transitions")
    for name, table in (
        ("generated_traces", generated),
        ("operations", operations),
        ("back_transitions", backs),
    ):
        if not isinstance(table, dict):
            raise TypeError(f"Reachability artifact has no {name} table.")

    require_fields(
        generated,
        {
            "cache_idx", "target", "prompt_length", "generated_length",
            "n_backtracks", "input_numbers",
        },
        "generated_traces",
    )
    require_fields(
        operations,
        {
            "record_pos", "cache_idx", "target", "operation_idx",
            "operator_generated_idx", "equals_generated_idx",
            "result_end_generated_idx", "semicolon_generated_idx",
            "after_binary_reachability_label_mask",
            "after_binary_reachability_label",
        },
        "operations",
    )
    require_fields(
        backs,
        {
            "record_pos", "cache_idx", "target", "backtrack_idx",
            "pre_back_generated_idx", "back_generated_idx",
            "semicolon_generated_idx", "post_binary_reachability_label_mask",
            "post_binary_reachability_label",
        },
        "back_transitions",
    )
    require_fields(
        evaluated,
        {
            "cache_idx", "target", "input_numbers", "prompt_length",
            "generated_flat_ids", "generated_offsets", "generated_lengths",
            "expr_target_correct", "expr_parseable",
        },
        "evaluated",
    )

    n_records = int(evaluated["cache_idx"].numel())
    for table_name, table in (
        ("operations", operations),
        ("back_transitions", backs),
    ):
        record_pos = table["record_pos"].to(torch.int64)
        if bool((record_pos < 0).any()) or bool((record_pos >= n_records).any()):
            raise RuntimeError(f"{table_name} contains an invalid record_pos.")

    for field in ("cache_idx", "target", "prompt_length", "input_numbers"):
        if not torch.equal(generated[field].to(torch.int64), evaluated[field].to(torch.int64)):
            raise RuntimeError(f"Field {field!r} differs across source artifacts.")
    if not torch.equal(
        generated["generated_length"].to(torch.int64),
        evaluated["generated_lengths"].to(torch.int64),
    ):
        raise RuntimeError("Generated lengths differ across source artifacts.")

    back_counts = torch.bincount(
        backs["record_pos"].to(torch.int64),
        minlength=n_records,
    ).to(torch.int64)
    if not torch.equal(back_counts, generated["n_backtracks"].to(torch.int64)):
        raise RuntimeError("BACK-transition rows do not match trace BACK counts.")

    return generated


# =========================
# STRUCTURAL FILTERS
# =========================
  
 
def valid_pre_back_row_mask(
    evaluated: dict[str, Any],
    backs: dict[str, torch.Tensor],
    contract: TokenContract,
) -> torch.Tensor:
    """Return BACK rows whose immediately preceding token is a semicolon."""
    record_pos = backs["record_pos"].to(torch.int64)
    pre_idx = backs["pre_back_generated_idx"].to(torch.int64)
    back_idx = backs["back_generated_idx"].to(torch.int64)
    lengths = evaluated["generated_lengths"].to(torch.int64)
    offsets = evaluated["generated_offsets"].to(torch.int64)

    record_in_range = (record_pos >= 0) & (record_pos < lengths.numel())
    trace_lengths = torch.zeros_like(back_idx)
    trace_lengths[record_in_range] = lengths[record_pos[record_in_range]]
    in_range = (
        record_in_range
        & (pre_idx >= 0)
        & (back_idx == pre_idx + 1)
        & (back_idx < trace_lengths)
    )
    valid = torch.zeros(record_pos.numel(), dtype=torch.bool)
    rows = torch.nonzero(in_range, as_tuple=False).flatten()
    if not rows.numel():
        return valid

    flat_ids = evaluated["generated_flat_ids"]
    row_offsets = offsets[record_pos[rows]]
    pre_flat = row_offsets + pre_idx[rows]
    back_flat = row_offsets + back_idx[rows]
    valid[rows] = (
        (flat_ids[pre_flat] == contract.semi)
        & (flat_ids[back_flat] == contract.back)
    )
    return valid


def scan_terminal_positions(
    evaluated: dict[str, Any],
    contract: TokenContract,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    records: list[int] = []
    successes: list[int] = []
    expr_ends: list[int] = []

    parseable = evaluated["expr_parseable"].to(torch.bool)
    for record_pos in range(int(parseable.numel())):
        if REQUIRE_PARSEABLE_EXPR and not bool(parseable[record_pos].item()):
            continue
        ids = generated_ids_at(evaluated, record_pos)
        success = [i for i, token_id in enumerate(ids) if token_id == contract.success]
        expr = [i for i, token_id in enumerate(ids) if token_id == contract.expr]
        if len(success) != 1 or len(expr) != 1 or not ids:
            continue

        success_idx = success[0]
        expr_idx = expr[0]
        if (
            success_idx + 2 >= len(ids)
            or ids[success_idx + 1] != contract.semi
            or expr_idx != success_idx + 2
            or ids[-1] != contract.rparen
        ):
            continue
        records.append(record_pos)
        successes.append(success_idx)
        expr_ends.append(len(ids) - 1)

    return (
        torch.tensor(records, dtype=torch.int64),
        torch.tensor(successes, dtype=torch.int64),
        torch.tensor(expr_ends, dtype=torch.int64),
    )


def find_pre_first_operand_indices(
    evaluated: dict[str, Any],
    record_pos: torch.Tensor,
    operator_idx: torch.Tensor,
    contract: TokenContract,
) -> torch.Tensor:
    """Return the arrow token immediately before each first operand."""

    # operator_idx - 1 is the final digit of the first operand.
    first_start = operator_idx.to(torch.int64) - 1
    if bool((first_start < 0).any()):
        raise RuntimeError("An operation has no first-operand token.")

    # Walk backward across any earlier digits of a multidigit operand.
    while True:
        previous = first_start - 1
        movable = previous >= 0
        move = torch.zeros_like(movable)
        rows = torch.nonzero(movable, as_tuple=False).flatten()

        if rows.numel():
            previous_tokens = packed_tokens_at(
                evaluated,
                record_pos[rows],
                previous[rows],
            )
            move[rows] = torch.isin(previous_tokens, contract.digits)

        if not bool(move.any()):
            break

        first_start[move] -= 1

    pre_first_operand = first_start - 1
    if bool((pre_first_operand < 0).any()):
        raise RuntimeError("An operation has no token before its first operand.")

    preceding_tokens = packed_tokens_at(
        evaluated,
        record_pos,
        pre_first_operand,
    )
    if not bool((preceding_tokens == contract.arrow).all()):
        raise RuntimeError(
            "A pre-first-operand position does not point to an arrow token."
        )

    return pre_first_operand

 
# =========================
# CHECKPOINT COLLECTION
# =========================


EXAMPLE_DTYPES: dict[str, torch.dtype] = {
    "task_id": torch.int8,
    "checkpoint_id": torch.int8,
    "source_table_id": torch.int8,
    "source_row": torch.int64,
    "record_pos": torch.int32,
    "cache_idx": torch.int64,
    "target": torch.int16,
    "label": torch.bool,
    "generated_token_idx": torch.int16,
    "absolute_token_idx": torch.int16,
    "split_id": torch.int8,
    "family_id": torch.int32,
    "operation_idx": torch.int16,
    "backtrack_idx": torch.int16,
    "trace_generated_length": torch.int16,
    "trace_total_back_count": torch.int16,
}


def make_examples(
    *,
    task_name: str,
    checkpoint_ids: torch.Tensor,
    source_table_name: str,
    source_rows: torch.Tensor,
    record_pos: torch.Tensor,
    cache_idx: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    generated_idx: torch.Tensor,
    operation_idx: torch.Tensor,
    backtrack_idx: torch.Tensor,
    generated: dict[str, torch.Tensor],
    trace_split: torch.Tensor,
    trace_family: torch.Tensor,
) -> dict[str, torch.Tensor]:
    record_pos = record_pos.to(torch.int64)
    generated_idx = generated_idx.to(torch.int64)
    n = int(record_pos.numel())

    table = {
        "task_id": torch.full(
            (n,),
            TASK_TO_ID[task_name],
            dtype=torch.int8,
        ),
        "checkpoint_id": checkpoint_ids.to(torch.int8),
        "source_table_id": torch.full(
            (n,),
            SOURCE_TABLE_TO_ID[source_table_name],
            dtype=torch.int8,
        ),
        "source_row": source_rows.to(torch.int64),
        "record_pos": record_pos.to(torch.int32),
        "cache_idx": cache_idx.to(torch.int64),
        "target": target.to(torch.int16),
        "label": label.to(torch.bool),
        "generated_token_idx": generated_idx.to(torch.int16),
        "absolute_token_idx": (
            generated["prompt_length"][record_pos].to(torch.int64)
            + generated_idx
        ).to(torch.int16),
        "split_id": trace_split[record_pos].to(torch.int8),
        "family_id": trace_family[record_pos].to(torch.int32),
        "operation_idx": operation_idx.to(torch.int16),
        "backtrack_idx": backtrack_idx.to(torch.int16),
        "trace_generated_length": generated[
            "generated_length"
        ][record_pos].to(torch.int16),
        "trace_total_back_count": generated[
            "n_backtracks"
        ][record_pos].to(torch.int16),
    }

    if {int(value.numel()) for value in table.values()} != {n}:
        raise RuntimeError("Example-table fields are misaligned.")

    return table


def collect_operation_examples(
    evaluated: dict[str, Any],
    labels: dict[str, Any],
    generated: dict[str, torch.Tensor],
    trace_split: torch.Tensor,
    trace_family: torch.Tensor,
    contract: TokenContract,
) -> dict[str, torch.Tensor]:
    operations = labels["operations"]
    rows = torch.nonzero(
        operations["after_binary_reachability_label_mask"].to(torch.bool),
        as_tuple=False,
    ).flatten().to(torch.int64)

    record_pos = operations["record_pos"][rows].to(torch.int64)
    operator_idx = operations[
        "operator_generated_idx"
    ][rows].to(torch.int64)
    equals_idx = operations[
        "equals_generated_idx"
    ][rows].to(torch.int64)
    result_idx = operations[
        "result_end_generated_idx"
    ][rows].to(torch.int64)
    semicolon_idx = operations[
        "semicolon_generated_idx"
    ][rows].to(torch.int64)

    pre_first_operand_idx = find_pre_first_operand_indices(
        evaluated,
        record_pos,
        operator_idx,
        contract,
    )

    first_operand_idx = operator_idx - 1
    second_operand_idx = equals_idx - 1
    lengths = generated["generated_length"][record_pos].to(torch.int64)

    valid_order = (
        (pre_first_operand_idx >= 0)
        & (pre_first_operand_idx < first_operand_idx)
        & (first_operand_idx >= 0)
        & (first_operand_idx < operator_idx)
        & (operator_idx < second_operand_idx)
        & (second_operand_idx < equals_idx)
        & (equals_idx < result_idx)
        & (result_idx < semicolon_idx)
        & (semicolon_idx < lengths)
    )
    if not bool(valid_order.all()):
        raise RuntimeError(
            "An operation has malformed checkpoint ordering."
        )

    first_token = packed_tokens_at(
        evaluated,
        record_pos,
        first_operand_idx,
    )
    operator_token = packed_tokens_at(
        evaluated,
        record_pos,
        operator_idx,
    )
    second_token = packed_tokens_at(
        evaluated,
        record_pos,
        second_operand_idx,
    )
    equals_token = packed_tokens_at(
        evaluated,
        record_pos,
        equals_idx,
    )
    result_token = packed_tokens_at(
        evaluated,
        record_pos,
        result_idx,
    )
    right_bracket = packed_tokens_at(
        evaluated,
        record_pos,
        semicolon_idx - 1,
    )
    semicolon_token = packed_tokens_at(
        evaluated,
        record_pos,
        semicolon_idx,
    )

    if not bool(torch.isin(first_token, contract.digits).all()):
        raise RuntimeError(
            "A first-operand checkpoint is not a digit."
        )
    if not bool(torch.isin(operator_token, contract.operators).all()):
        raise RuntimeError(
            "An operator checkpoint is not an operator."
        )
    if not bool(torch.isin(second_token, contract.digits).all()):
        raise RuntimeError(
            "A second-operand checkpoint is not a digit."
        )
    if not bool((equals_token == contract.eq).all()):
        raise RuntimeError("An equals checkpoint is malformed.")
    if not bool(torch.isin(result_token, contract.digits).all()):
        raise RuntimeError("A result checkpoint is not a digit.")
    if not bool((right_bracket == contract.rbr).all()) or not bool(
        (semicolon_token == contract.semi).all()
    ):
        raise RuntimeError(
            "An operation state boundary is malformed."
        )

    positions = torch.stack(
        [
            pre_first_operand_idx,
            first_operand_idx,
            operator_idx,
            second_operand_idx,
            result_idx,
            semicolon_idx,
        ],
        dim=1,
    )

    n_units = int(rows.numel())
    repeats = 6

    return make_examples(
        task_name="operation_reachability",
        checkpoint_ids=torch.tensor(
            [      
                CHECKPOINT_TO_ID["pre_first_operand"],
                CHECKPOINT_TO_ID["first_operand_end"],
                CHECKPOINT_TO_ID["operator"],
                CHECKPOINT_TO_ID["second_operand_end"],
                CHECKPOINT_TO_ID["result_end"],
                CHECKPOINT_TO_ID["resulting_state_semicolon"],
            ],
            dtype=torch.int8,
        ).repeat(n_units),
        source_table_name="operations",
        source_rows=rows.repeat_interleave(repeats),
        record_pos=record_pos.repeat_interleave(repeats),
        cache_idx=operations[
            "cache_idx"
        ][rows].repeat_interleave(repeats),
        target=operations[
            "target"
        ][rows].repeat_interleave(repeats),
        label=operations[
            "after_binary_reachability_label"
        ][rows].repeat_interleave(repeats),
        generated_idx=positions.reshape(-1),
        operation_idx=operations[
            "operation_idx"
        ][rows].repeat_interleave(repeats),
        backtrack_idx=torch.full(
            (n_units * repeats,),
            -1,
            dtype=torch.int16,
        ),
        generated=generated,
        trace_split=trace_split,
        trace_family=trace_family,
    )


def collect_back_examples(
    evaluated: dict[str, Any],
    labels: dict[str, Any],
    generated: dict[str, torch.Tensor],
    trace_split: torch.Tensor,
    trace_family: torch.Tensor,
    contract: TokenContract,
) -> dict[str, torch.Tensor]:
    backs = labels["back_transitions"]
    rows = torch.nonzero(
        backs["post_binary_reachability_label_mask"].to(torch.bool),
        as_tuple=False,
    ).flatten().to(torch.int64)

    record_pos = backs["record_pos"][rows].to(torch.int64)
    back_idx = backs[
        "back_generated_idx"
    ][rows].to(torch.int64)
    semicolon_idx = backs[
        "semicolon_generated_idx"
    ][rows].to(torch.int64)
    lengths = generated["generated_length"][record_pos].to(torch.int64)

    valid_order = (
        (back_idx >= 0)
        & (back_idx < semicolon_idx)
        & (semicolon_idx < lengths)
    )
    if not bool(valid_order.all()):
        raise RuntimeError(
            "A BACK transition has malformed checkpoint ordering."
        )

    back_token = packed_tokens_at(
        evaluated,
        record_pos,
        back_idx,
    )
    right_bracket = packed_tokens_at(
        evaluated,
        record_pos,
        semicolon_idx - 1,
    )
    semicolon_token = packed_tokens_at(
        evaluated,
        record_pos,
        semicolon_idx,
    )

    if not bool((back_token == contract.back).all()):
        raise RuntimeError(
            "A BACK checkpoint does not point to BACK."
        )
    if not bool((right_bracket == contract.rbr).all()) or not bool(
        (semicolon_token == contract.semi).all()
    ):
        raise RuntimeError(
            "A restored-state boundary is malformed."
        )

    positions = torch.stack(
        [back_idx, semicolon_idx],
        dim=1,
    )

    n_units = int(rows.numel())
    repeats = 2

    return make_examples(
        task_name="back_reachability",
        checkpoint_ids=torch.tensor(
            [
                CHECKPOINT_TO_ID["back_token"],
                CHECKPOINT_TO_ID["restored_state_semicolon"],
            ],
            dtype=torch.int8,
        ).repeat(n_units),
        source_table_name="back_transitions",
        source_rows=rows.repeat_interleave(repeats),
        record_pos=record_pos.repeat_interleave(repeats),
        cache_idx=backs[
            "cache_idx"
        ][rows].repeat_interleave(repeats),
        target=backs[
            "target"
        ][rows].repeat_interleave(repeats),
        label=backs[
            "post_binary_reachability_label"
        ][rows].repeat_interleave(repeats),
        generated_idx=positions.reshape(-1),
        operation_idx=torch.full(
            (n_units * repeats,),
            -1,
            dtype=torch.int16,
        ),
        backtrack_idx=backs[
            "backtrack_idx"
        ][rows].repeat_interleave(repeats),
        generated=generated,
        trace_split=trace_split,
        trace_family=trace_family,
    )


def collect_trajectory_examples(
    evaluated: dict[str, Any],
    labels: dict[str, Any],
    generated: dict[str, torch.Tensor],
    trace_split: torch.Tensor,
    trace_family: torch.Tensor,
    contract: TokenContract,
) -> dict[str, torch.Tensor]:
    backs = labels["back_transitions"]
    record_all = backs["record_pos"].to(torch.int64)
    n_records = int(generated["cache_idx"].numel())

    valid_row = valid_pre_back_row_mask(
        evaluated,
        backs,
        contract,
    )
    total_count = torch.bincount(
        record_all,
        minlength=n_records,
    )
    valid_count = torch.bincount(
        record_all[valid_row],
        minlength=n_records,
    )

    valid_trace = (
        (total_count > 0)
        & (valid_count == total_count)
    )
    if REQUIRE_PARSEABLE_EXPR:
        valid_trace &= evaluated[
            "expr_parseable"
        ].to(torch.bool)

    rows = torch.nonzero(
        valid_row & valid_trace[record_all],
        as_tuple=False,
    ).flatten().to(torch.int64)
    record_pos = record_all[rows]

    return make_examples(
        task_name="trace_outcome_trajectory",
        checkpoint_ids=torch.full(
            (rows.numel(),),
            CHECKPOINT_TO_ID["pre_back"],
            dtype=torch.int8,
        ),
        source_table_name="back_transitions",
        source_rows=rows,
        record_pos=record_pos,
        cache_idx=backs["cache_idx"][rows],
        target=backs["target"][rows],
        label=evaluated[
            "expr_target_correct"
        ][record_pos],
        generated_idx=backs[
            "pre_back_generated_idx"
        ][rows],
        operation_idx=torch.full(
            (rows.numel(),),
            -1,
            dtype=torch.int16,
        ),
        backtrack_idx=backs["backtrack_idx"][rows],
        generated=generated,
        trace_split=trace_split,
        trace_family=trace_family,
    )


def collect_terminal_examples(
    evaluated: dict[str, Any],
    generated: dict[str, torch.Tensor],
    trace_split: torch.Tensor,
    trace_family: torch.Tensor,
    contract: TokenContract,
) -> dict[str, torch.Tensor]:
    rows, success_idx, expr_end_idx = scan_terminal_positions(
        evaluated,
        contract,
    )
    positions = torch.stack(
        [success_idx, expr_end_idx],
        dim=1,
    )

    n_units = int(rows.numel())
    repeats = 2

    return make_examples(
        task_name="terminal_outcome_controls",
        checkpoint_ids=torch.tensor(
            [
                CHECKPOINT_TO_ID["success_token"],
                CHECKPOINT_TO_ID["expr_final_parenthesis"],
            ],
            dtype=torch.int8,
        ).repeat(n_units),
        source_table_name="generated_traces",
        source_rows=rows.repeat_interleave(repeats),
        record_pos=rows.repeat_interleave(repeats),
        cache_idx=generated[
            "cache_idx"
        ][rows].repeat_interleave(repeats),
        target=generated[
            "target"
        ][rows].repeat_interleave(repeats),
        label=evaluated[
            "expr_target_correct"
        ][rows].repeat_interleave(repeats),
        generated_idx=positions.reshape(-1),
        operation_idx=torch.full(
            (n_units * repeats,),
            -1,
            dtype=torch.int16,
        ),
        backtrack_idx=torch.full(
            (n_units * repeats,),
            -1,
            dtype=torch.int16,
        ),
        generated=generated,
        trace_split=trace_split,
        trace_family=trace_family,
    )


def collect_all_examples(
    evaluated: dict[str, Any],
    labels: dict[str, Any],
    generated: dict[str, torch.Tensor],
    trace_split: torch.Tensor,
    trace_family: torch.Tensor,
    contract: TokenContract,
) -> dict[str, torch.Tensor]:
    tables = [
        collect_operation_examples(
            evaluated,
            labels,
            generated,
            trace_split,
            trace_family,
            contract,
        ),
        collect_back_examples(
            evaluated,
            labels,
            generated,
            trace_split,
            trace_family,
            contract,
        ),
        collect_trajectory_examples(
            evaluated,
            labels,
            generated,
            trace_split,
            trace_family,
            contract,
        ),
        collect_terminal_examples(
            evaluated,
            generated,
            trace_split,
            trace_family,
            contract,
        ),
    ]

    examples = {
        field: torch.cat(
            [table[field] for table in tables],
            dim=0,
        ).to(dtype)
        for field, dtype in EXAMPLE_DTYPES.items()
    }

    if not examples["task_id"].numel():
        raise RuntimeError(
            "No probe examples were collected."
        )

    return examples


def deduplicate_positions(
    examples: dict[str, torch.Tensor],
    generated: dict[str, torch.Tensor],
    trace_split: torch.Tensor,
    trace_family: torch.Tensor,
) -> dict[str, torch.Tensor]:
    record_pos = examples["record_pos"].to(torch.int64)
    absolute_idx = examples[
        "absolute_token_idx"
    ].to(torch.int64)

    if bool((absolute_idx < 0).any()) or bool(
        (absolute_idx >= POSITION_KEY_BASE).any()
    ):
        raise RuntimeError(
            "An absolute token index exceeds POSITION_KEY_BASE."
        )

    keys = (
        record_pos * POSITION_KEY_BASE
        + absolute_idx
    )
    unique_keys, inverse = torch.unique(
        keys,
        sorted=True,
        return_inverse=True,
    )

    position_record = torch.div(
        unique_keys,
        POSITION_KEY_BASE,
        rounding_mode="floor",
    ).to(torch.int64)
    position_absolute = (
        unique_keys % POSITION_KEY_BASE
    ).to(torch.int64)
    position_generated = (
        position_absolute
        - generated[
            "prompt_length"
        ][position_record].to(torch.int64)
    )

    if bool((position_generated < 0).any()):
        raise RuntimeError(
            "A probe position falls inside the prompt."
        )

    positions = {
        "record_pos": position_record.to(torch.int32),
        "cache_idx": generated[
            "cache_idx"
        ][position_record].to(torch.int64),
        "absolute_token_idx": position_absolute.to(torch.int16),
        "generated_token_idx": position_generated.to(torch.int16),
        "split_id": trace_split[
            position_record
        ].to(torch.int8),
        "family_id": trace_family[
            position_record
        ].to(torch.int32),
        "n_examples": torch.bincount(
            inverse,
            minlength=unique_keys.numel(),
        ).to(torch.int16),
    }

    activation_index = inverse.to(torch.int32)
    examples["activation_index"] = activation_index

    activation_index_long = activation_index.to(torch.int64)
    for field in (
        "record_pos",
        "cache_idx",
        "absolute_token_idx",
        "generated_token_idx",
        "split_id",
        "family_id",
    ):
        if not torch.equal(
            examples[field],
            positions[field][activation_index_long],
        ):
            raise RuntimeError(
                f"Activation-position join disagrees on {field}."
            )

    return positions


def build_summary(
    examples: dict[str, torch.Tensor],
    positions: dict[str, torch.Tensor],
) -> dict[str, Any]:
    print("\nCOLLECTED PROBE EXAMPLES")
    tasks: dict[str, Any] = {}

    for task_name in TASK_NAMES:
        task_id = TASK_TO_ID[task_name]
        task_mask = examples["task_id"] == task_id
        if not bool(task_mask.any()):
            continue

        print(
            f"  {task_name}: "
            f"examples={int(task_mask.sum().item()):,}, "
            f"traces="
            f"{int(torch.unique(examples['record_pos'][task_mask]).numel()):,}"
        )

        splits: dict[str, Any] = {}
        for split_name in SPLIT_NAMES:
            split_id = SPLIT_TO_ID[split_name]
            mask = (
                task_mask
                & (examples["split_id"] == split_id)
            )
            labels = examples["label"][mask]
            positive = int(labels.sum().item())
            negative = int(labels.numel() - positive)

            splits[split_name] = {
                "examples": int(labels.numel()),
                "positive": positive,
                "negative": negative,
            }

            print(
                f"    {split_name:<5} "
                f"examples={int(labels.numel()):>10,} "
                f"positive={positive:>10,} "
                f"negative={negative:>10,}"
            )

        checkpoints: dict[str, int] = {}
        for checkpoint_name in CHECKPOINT_NAMES:
            checkpoint_id = CHECKPOINT_TO_ID[checkpoint_name]
            count = int(
                (
                    task_mask
                    & (
                        examples["checkpoint_id"]
                        == checkpoint_id
                    )
                ).sum().item()
            )
            if count:
                checkpoints[checkpoint_name] = count
                print(
                    f"      {checkpoint_name:<28} "
                    f"{count:>10,}"
                )

        tasks[task_name] = {
            "splits": splits,
            "checkpoints": checkpoints,
        }

    logical_examples = int(examples["task_id"].numel())
    unique_positions = int(positions["record_pos"].numel())
    unique_traces = int(
        torch.unique(positions["record_pos"]).numel()
    )

    print()
    print(
        f"  Logical probe examples:      "
        f"{logical_examples:>12,}"
    )
    print(
        f"  Unique activation positions: "
        f"{unique_positions:>12,}"
    )
    print(
        f"  Unique replay traces:        "
        f"{unique_traces:>12,}"
    )
    print(
        f"  Reused position links:       "
        f"{logical_examples - unique_positions:>12,}"
    )

    return {
        "tasks": tasks,
        "logical_examples": logical_examples,
        "unique_activation_positions": unique_positions,
        "unique_replay_traces": unique_traces,
        "reused_position_links": (
            logical_examples - unique_positions
        ),
    }
  
def audit_activation_cache(
    examples: dict[str, torch.Tensor],
    positions: dict[str, torch.Tensor],
    attention: torch.Tensor,
    mlp: torch.Tensor,
    residual: torch.Tensor,
    final: torch.Tensor,
) -> dict[str, Any]:
    logical_count = int(examples["task_id"].numel())
    position_count = int(positions["record_pos"].numel())
    activation_index = examples["activation_index"].to(torch.int64)

    if bool((activation_index < 0).any()) or bool(
        (activation_index >= position_count).any()
    ):
        raise RuntimeError("An example has an invalid activation_index.")

    links_per_position = torch.bincount(
        activation_index,
        minlength=position_count,
    )
    if not torch.equal(
        links_per_position,
        positions["n_examples"].to(torch.int64),
    ):
        raise RuntimeError(
            "Logical rows disagree with position reuse counts."
        )
    if bool((links_per_position == 0).any()):
        raise RuntimeError(
            "A cached activation position is unreferenced."
        )

    tensors = {
        "attention": attention,
        "mlp": mlp,
        "residual": residual,
        "final_hidden": final,
    }
    for name, tensor in tensors.items():
        if int(tensor.shape[0]) != position_count:
            raise RuntimeError(
                f"{name} has {int(tensor.shape[0]):,} rows, "
                f"expected {position_count:,}."
            )
        if tensor.dtype != ACTIVATION_DTYPE:
            raise RuntimeError(
                f"{name} has dtype {tensor.dtype}, "
                f"expected {ACTIVATION_DTYPE}."
            )

        for start in range(
            0,
            position_count,
            ACTIVATION_AUDIT_CHUNK_SIZE,
        ):
            chunk = tensor[
                start:start + ACTIVATION_AUDIT_CHUNK_SIZE
            ]
            if not bool(torch.isfinite(chunk).all()):
                raise RuntimeError(
                    f"{name} contains non-finite values "
                    f"near row {start:,}."
                )

    checkpoint_counts: dict[str, Any] = {}

    print("\nACTIVATION CACHE AUDIT")
    print(
        f"  {'checkpoint':<29} "
        f"{'logical':>12} "
        f"{'unique cached':>14}"
    )
    print("  " + "-" * 57)

    for checkpoint_name in CHECKPOINT_NAMES:
        mask = (
            examples["checkpoint_id"]
            == CHECKPOINT_TO_ID[checkpoint_name]
        )
        logical = int(mask.sum().item())
        if not logical:
            continue

        unique_cached = int(
            torch.unique(
                activation_index[mask]
            ).numel()
        )
        checkpoint_counts[checkpoint_name] = {
            "logical_examples": logical,
            "unique_activation_positions": unique_cached,
        }

        print(
            f"  {checkpoint_name:<29} "
            f"{logical:>12,} "
            f"{unique_cached:>14,}"
        )

    logical_per_trace = torch.bincount(
        examples["record_pos"].to(torch.int64)
    )
    unique_per_trace = torch.bincount(
        positions["record_pos"].to(torch.int64)
    )
    logical_per_trace = logical_per_trace[
        logical_per_trace > 0
    ].to(torch.float32)
    unique_per_trace = unique_per_trace[
        unique_per_trace > 0
    ].to(torch.float32)

    used_once = int(
        (links_per_position == 1).sum().item()
    )
    used_twice = int(
        (links_per_position == 2).sum().item()
    )
    used_three_or_more = int(
        (links_per_position >= 3).sum().item()
    )

    logical_mean = float(
        logical_per_trace.mean().item()
    )
    logical_median = float(
        logical_per_trace.median().item()
    )
    logical_max = int(
        logical_per_trace.max().item()
    )

    unique_mean = float(
        unique_per_trace.mean().item()
    )
    unique_median = float(
        unique_per_trace.median().item()
    )
    unique_max = int(
        unique_per_trace.max().item()
    )

    print("\n  COVERAGE")
    print(
        f"    Logical examples:            "
        f"{logical_count:>12,}"
    )
    print(
        f"    Unique cached positions:     "
        f"{position_count:>12,}"
    )
    print(
        f"    Unfilled positions:          "
        f"{0:>12,}"
    )
    print(
        "    Positions used once / twice / 3+: "
        f"{used_once:,} / "
        f"{used_twice:,} / "
        f"{used_three_or_more:,}"
    )
    print(
        "    Attention/MLP/residual/final values finite: yes"
    )
    print(
        "    Logical examples/question: "
        f"mean={logical_mean:.2f}, "
        f"median={logical_median:.0f}, "
        f"max={logical_max:,}"
    )
    print(
        "    Unique positions/question: "
        f"mean={unique_mean:.2f}, "
        f"median={unique_median:.0f}, "
        f"max={unique_max:,}"
    )

    return {
        "logical_examples": logical_count,
        "unique_activation_positions": position_count,
        "checkpoint_counts": checkpoint_counts,
        "positions_used_once": used_once,
        "positions_used_twice": used_twice,
        "positions_used_three_or_more": (
            used_three_or_more
        ),
        "all_values_finite": True,
        "unfilled_positions": 0,
        "mean_logical_examples_per_trace": (
            logical_mean
        ),
        "mean_unique_positions_per_trace": (
            unique_mean
        ),
    }


def audit_saved_shards(
    shard_dir: str,
    shard_manifest: list[dict[str, Any]],
    n_positions: int,
    attention_shape: tuple[int, ...],
    mlp_shape: tuple[int, ...],
    residual_shape: tuple[int, ...],
    final_shape: tuple[int, ...],
) -> dict[str, Any]:
    expected_start = 0

    for shard_idx, shard in enumerate(
        shard_manifest
    ):
        start = int(shard["start"])
        end = int(shard["end"])

        if start != expected_start or end <= start:
            raise RuntimeError(
                f"Shard {shard_idx} has a gap, "
                "overlap, or empty range."
            )

        path = Path(shard_dir) / str(
            shard["file"]
        )
        if (
            not path.is_file()
            or int(path.stat().st_size)
            != int(shard["bytes"])
        ):
            raise RuntimeError(
                f"Saved shard is missing or changed: {path}"
            )

        expected_start = end

    manifest_path = Path(shard_dir) / "manifest.pt"
    saved = torch.load(
        manifest_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        expected_start != n_positions
        or not bool(saved.get("complete", False))
        or int(saved["n_positions"]) != n_positions
        or len(saved["shards"]) != len(shard_manifest)
        or tuple(saved["attention_shape"]) != attention_shape
        or tuple(saved["mlp_shape"]) != mlp_shape
        or tuple(saved["residual_shape"]) != residual_shape
        or tuple(saved["final_shape"]) != final_shape
    ):
        raise RuntimeError(
            "Saved shard manifest failed "
            "its coverage audit."
        )

    print("\nSAVED SHARD AUDIT")
    print(
        f"  Shard files:          "
        f"{len(shard_manifest):>12,}"
    )
    print(
        f"  Covered positions:    "
        f"{expected_start:>12,}"
    )
    print(
        f"  Gaps or overlaps:     "
        f"{0:>12,}"
    )
    print("  Manifest complete:             yes")

    return {
        "complete": True,
        "shard_files": len(shard_manifest),
        "covered_positions": expected_start,
        "gaps_or_overlaps": 0,
        "manifest_path": str(manifest_path),
    }

# =========================
# ACTIVATION CACHING
# =========================


def estimated_activation_bytes(
    n_positions: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    d_model: int,
    dtype: torch.dtype,
) -> int:
    values = n_layers * n_heads * head_dim + n_layers * d_model + n_layers * d_model + d_model 
    return n_positions * values * torch.empty((), dtype=dtype).element_size()


def cache_activations(
    *,
    evaluated: dict[str, Any],
    generated: dict[str, torch.Tensor],
    positions: dict[str, torch.Tensor],
    blob: dict[str, Any],
    tok: pm.Tokenizer,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    model_max_len: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_positions = int(positions["record_pos"].numel())
    n_layers, d_model, n_heads, head_dim = infer_model_dimensions(model, model_config)

    attention = torch.empty(
        (n_positions, n_layers, n_heads, head_dim),
        dtype=ACTIVATION_DTYPE,
    )
    mlp = torch.empty(
        (n_positions, n_layers, d_model),
        dtype=ACTIVATION_DTYPE,
    )
    residual = torch.empty((n_positions, n_layers, d_model), dtype=ACTIVATION_DTYPE)
    final = torch.empty((n_positions, d_model), dtype=ACTIVATION_DTYPE)
    filled = torch.zeros(n_positions, dtype=torch.bool)

    by_trace: dict[int, list[int]] = defaultdict(list)
    for activation_idx, record_pos in enumerate(positions["record_pos"].tolist()):
        by_trace[int(record_pos)].append(activation_idx)

    full_lengths = (
        generated["prompt_length"].to(torch.int64)
        + generated["generated_length"].to(torch.int64)
    )
    traces = sorted(by_trace, key=lambda row: int(full_lengths[row].item()))
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    capture = ActivationCapture(model)
    try:
        for start in tqdm(
            range(0, len(traces), batch_size),
            desc="Caching traces",
            dynamic_ncols=True,
        ):
            trace_rows = traces[start:start + batch_size]
            sequences: list[list[int]] = []
            position_rows: list[list[int]] = []
            output_rows: list[list[int]] = []

            for record_pos in trace_rows:
                cache_idx = int(generated["cache_idx"][record_pos].item())
                prompt_ids = prompt_ids_from_cache(blob, cache_idx, tok)
                generated_ids = generated_ids_at(evaluated, record_pos)
                expected_prompt = int(generated["prompt_length"][record_pos].item())
                if len(prompt_ids) != expected_prompt:
                    raise RuntimeError(f"Prompt length mismatch at trace {record_pos}.")

                full_ids = prompt_ids + generated_ids
                if len(full_ids) > model_max_len:
                    raise RuntimeError(f"Trace {record_pos} exceeds model_max_len.")

                output_indices = by_trace[record_pos]
                absolute = [int(positions["absolute_token_idx"][idx].item()) for idx in output_indices]
                if any(index < 0 or index >= len(full_ids) for index in absolute):
                    raise RuntimeError(f"Activation position outside trace {record_pos}.")

                sequences.append(full_ids)
                position_rows.append(absolute)
                output_rows.append(output_indices)

            max_length = max(map(len, sequences))
            max_positions = max(map(len, position_rows))
            x = torch.full(
                (len(sequences), max_length),
                tok.pad_id,
                dtype=torch.long,
                device=device,
            )
            position_tensor = torch.zeros(
                (len(sequences), max_positions),
                dtype=torch.long,
                device=device,
            )
            valid = torch.zeros(
                (len(sequences), max_positions),
                dtype=torch.bool,
                device=device,
            )
            output_tensor = torch.full(
                (len(sequences), max_positions),
                -1,
                dtype=torch.long,
            )

            for row, sequence in enumerate(sequences):
                x[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
                count = len(position_rows[row])
                position_tensor[row, :count] = torch.tensor(position_rows[row], device=device)
                valid[row, :count] = True
                output_tensor[row, :count] = torch.tensor(output_rows[row])

            capture.clear()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                final_hidden = model.forward_features(x)

            if set(capture.heads) != set(range(n_layers)):
                raise RuntimeError("Did not capture every attention layer.")
            if set(capture.mlps) != set(range(n_layers)):
                raise RuntimeError("Did not capture every MLP layer.")
            if set(capture.residuals) != set(range(n_layers)):
                raise RuntimeError("Did not capture every residual layer.")

            head_layers: list[torch.Tensor] = []
            mlp_layers: list[torch.Tensor] = []
            residual_layers: list[torch.Tensor] = []
            for layer_idx, block in enumerate(model.blocks):
                merged = capture.heads[layer_idx]
                batch_count, seq_len, hidden_size = merged.shape
                if hidden_size != n_heads * head_dim:
                    raise RuntimeError("Unexpected merged attention width.")
                mlp_output = capture.mlps[layer_idx]
                if tuple(mlp_output.shape) != (batch_count, seq_len, d_model):
                    raise RuntimeError(
                        f"Unexpected MLP output shape in layer {layer_idx}: "
                        f"{tuple(mlp_output.shape)}"
                    )
                head_layers.append(
                    merged.reshape(batch_count, seq_len, n_heads, head_dim)
                )
                mlp_layers.append(mlp_output)
                residual_layers.append(capture.residuals[layer_idx])

            all_heads = torch.stack(head_layers, dim=2)
            all_mlps = torch.stack(mlp_layers, dim=2)
            all_residuals = torch.stack(residual_layers, dim=2)
            gathered_heads = gather_positions(all_heads, position_tensor, valid)
            gathered_mlps = gather_positions(all_mlps, position_tensor, valid)
            gathered_residuals = gather_positions(all_residuals, position_tensor, valid)
            gathered_final = gather_positions(final_hidden, position_tensor, valid)

            output_indices = output_tensor[valid.cpu()]
            mlp[output_indices] = gathered_mlps[valid].to(ACTIVATION_DTYPE).cpu()
            attention[output_indices] = gathered_heads[valid].to(ACTIVATION_DTYPE).cpu()
            residual[output_indices] = gathered_residuals[valid].to(ACTIVATION_DTYPE).cpu()
            final[output_indices] = gathered_final[valid].to(ACTIVATION_DTYPE).cpu()
            filled[output_indices] = True

            del x, position_tensor, valid, final_hidden
            del all_heads, all_mlps, all_residuals, gathered_heads, gathered_mlps, gathered_residuals, gathered_final
    finally:
        capture.close()

    if not bool(filled.all()):
        raise RuntimeError(f"Failed to fill {(~filled).sum().item():,} activation positions.")
    return attention, mlp, residual, final


# =========================
# MAIN
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all A* probe checkpoints "
            "and cache their model activations."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ACTIVATION_TRACE_BATCH_SIZE,
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    width = 104
    print("=" * width)
    print("BUILD A* PROBE DATASET".center(width))
    print("=" * width)
    print(f"Evaluated traces:       {EVALUATED_PATH}")
    print(f"Reachability labels:    {LABEL_PATH}")
    print(f"Token cache:            {TOKEN_CACHE_PATH}")
    print(f"Checkpoint:             {CHECKPOINT_PATH}")
    print(f"Dataset metadata:       {DATASET_PATH}")
    print(f"Activation shard root:  {ACTIVATION_SHARD_ROOT}")
    print(f"Split weights:          {SPLIT_WEIGHTS}")
    print(f"Random seed:            {RANDOM_SEED}")
    print(f"Batch size:             {args.batch_size}")
    print(
        f"Activation dtype:       "
        f"{str(ACTIVATION_DTYPE).replace('torch.', '')}"
    )
    print(
        f"Require parseable EXPR: "
        f"{REQUIRE_PARSEABLE_EXPR}"
    )
    print("=" * width)

    for path in (
        EVALUATED_PATH,
        LABEL_PATH,
        TOKEN_CACHE_PATH,
        CHECKPOINT_PATH,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    if os.path.exists(DATASET_PATH):
        print(
            f"\nNote: the existing metadata at {DATASET_PATH} "
            "will remain in place until the new metadata "
            "is saved successfully."
        )

    stale_tmp = f"{DATASET_PATH}.tmp"
    if os.path.exists(stale_tmp):
        print(
            f"WARNING: stale partial output exists at {stale_tmp}",
            flush=True,
        )


    print("\nLoading source artifacts...", flush=True)
    evaluated = torch.load(
        EVALUATED_PATH,
        map_location="cpu",
        weights_only=False,
    )
    labels = torch.load(
        LABEL_PATH,
        map_location="cpu",
        weights_only=False,
    )
    generated = validate_sources(evaluated, labels)

    tok = pm.Tokenizer()
    contract = TokenContract(tok)

    (
        trace_split,
        trace_family,
        family_values,
        family_split,
    ) = build_family_splits(
        generated["input_numbers"].to(torch.int64)
    )

    print("INPUT-FAMILY SPLITS")
    for split_name in SPLIT_NAMES:
        split_id = SPLIT_TO_ID[split_name]
        print(
            f"  {split_name:<5} "
            f"traces="
            f"{int((trace_split == split_id).sum()):>8,} "
            f"families="
            f"{int((family_split == split_id).sum()):>8,}"
        )

    print(
        "\nCollecting every eligible checkpoint...",
        flush=True,
    )
    examples = collect_all_examples(
        evaluated,
        labels,
        generated,
        trace_split,
        trace_family,
        contract,
    )
    positions = deduplicate_positions(
        examples,
        generated,
        trace_split,
        trace_family,
    )
    summary = build_summary(
        examples,
        positions,
    )

    print("\nLoading token cache and model...", flush=True)
    blob, _, _, _, _, _ = pm._load_enc_blob(
        TOKEN_CACHE_PATH,
        tok,
        safe_max_len=2048,
        long_seq_policy="drop",
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

        if hasattr(
            torch,
            "set_float32_matmul_precision",
        ):
            torch.set_float32_matmul_precision("high")

        if hasattr(
            torch.backends.cuda,
            "enable_flash_sdp",
        ):
            torch.backends.cuda.enable_flash_sdp(True)

        if hasattr(
            torch.backends.cuda,
            "enable_mem_efficient_sdp",
        ):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        if hasattr(
            torch.backends.cuda,
            "enable_math_sdp",
        ):
            torch.backends.cuda.enable_math_sdp(True)

    model, model_config, model_max_len = load_model(
        tok,
        device,
    )
    (
        n_layers,
        d_model,
        n_heads,
        head_dim,
    ) = infer_model_dimensions(
        model,
        model_config,
    )

    print(f"Activation device: {device}")
    print(
        f"Model dimensions: layers={n_layers}, "
        f"d_model={d_model}, heads={n_heads}, "
        f"head_dim={head_dim}"
    )

    if device.type != "cuda":
        print(
            "WARNING: CUDA is unavailable; "
            "activation caching will run on CPU."
        )

    estimated = estimated_activation_bytes(
        int(positions["record_pos"].numel()),
        n_layers,
        n_heads,
        head_dim,
        d_model,
        ACTIVATION_DTYPE,
    )
    print(
        f"Estimated raw activation storage: "
        f"{estimated / (1024 ** 3):.2f} GiB"
    )

    attention, mlp, residual, final = cache_activations(
        evaluated=evaluated,
        generated=generated,
        positions=positions,
        blob=blob,
        tok=tok,
        model=model,
        model_config=model_config,
        model_max_len=model_max_len,
        device=device,
        batch_size=int(args.batch_size),
    )

    attention_shape = tuple(attention.shape)
    mlp_shape = tuple(mlp.shape)
    residual_shape = tuple(residual.shape)
    final_shape = tuple(final.shape)

    print(
        "\nAuditing cached activations...",
        flush=True,
    )
    activation_audit = audit_activation_cache(
        examples,
        positions,
        attention,
        mlp,
        residual,
        final,
    )

    # The GPU replay is complete. Release source artifacts and model memory
    # before serializing the activation cache.
    del evaluated
    del labels
    del generated
    del blob
    del model
    del tok
    del contract
    del trace_split
    del trace_family

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    shard_run_id = f"run_{int(time.time())}_{os.getpid()}"
    shard_dir = os.path.join(ACTIVATION_SHARD_ROOT, shard_run_id)
    n_shards = (
        int(attention_shape[0])
        + ACTIVATION_SHARD_SIZE
        - 1
    ) // ACTIVATION_SHARD_SIZE

    print(
        f"\nSaving {n_shards:,} activation shards to {shard_dir}...",
        flush=True,
    )
    shard_manifest = save_activation_shards(
        attention,
        mlp,
        residual,
        final,
        shard_dir,
        ACTIVATION_SHARD_SIZE,
    )
    shard_total_bytes = sum(
        int(shard["bytes"])
        for shard in shard_manifest
    )
    print(
        f"Activation shard save completed: "
        f"{len(shard_manifest):,} files.",
        flush=True,
    )

    shard_audit = audit_saved_shards(
        shard_dir,
        shard_manifest,
        int(attention_shape[0]),
        attention_shape,
        mlp_shape,
        residual_shape,
        final_shape,
    )

    # Activations are now safely stored on disk; the metadata file only
    # contains labels, indices, shapes, and the shard manifest.
    del attention
    del mlp
    del residual
    del final
    gc.collect()

    payload = {
        "format_version": 3,
        "complete": True,
        "run_tag": RUN_TAG,
        "random_seed": RANDOM_SEED,
        "require_parseable_expr": REQUIRE_PARSEABLE_EXPR,
        "source_evaluated_path": EVALUATED_PATH,
        "source_label_path": LABEL_PATH,
        "source_token_cache_path": TOKEN_CACHE_PATH,
        "source_checkpoint_path": CHECKPOINT_PATH,
        "split_names": list(SPLIT_NAMES),
        "split_to_id": dict(SPLIT_TO_ID),
        "split_weights": dict(SPLIT_WEIGHTS),
        "task_names": list(TASK_NAMES),
        "task_to_id": dict(TASK_TO_ID),
        "task_label_names": dict(TASK_LABEL_NAMES),
        "checkpoint_names": list(CHECKPOINT_NAMES),
        "checkpoint_to_id": dict(CHECKPOINT_TO_ID),
        "source_table_names": list(SOURCE_TABLE_NAMES),
        "source_table_to_id": dict(SOURCE_TABLE_TO_ID),
        "input_family_table": {
            "values": family_values,
            "split_id": family_split,
        },
        "examples": examples,
        "activation_positions": positions,
        "activation_storage": "sharded",
        "activation_shard_dir": shard_dir,
        "activation_shard_size": ACTIVATION_SHARD_SIZE,
        "activation_shards": shard_manifest,
        "attention_head_output_pre_projection_shape": attention_shape,
        "mlp_output_before_residual_shape": mlp_shape,
        "residual_stream_after_block_shape": residual_shape,
        "final_layer_normalized_hidden_shape": final_shape,
        "model_config": model_config,
        "model_max_len": int(model_max_len),
        "activation_dtype": str(
            ACTIVATION_DTYPE
        ).replace("torch.", ""),
        "summary": summary,
        "activation_audit": activation_audit,
        "shard_audit": shard_audit,
    }

    print(
        f"\nSaving dataset metadata to {DATASET_PATH}...",
        flush=True,
    )

    try:
        safe_torch_save(payload, DATASET_PATH)
    except BaseException:
        print(
            "Metadata save failed, but the completed activation shards "
            f"remain at {shard_dir}",
            flush=True,
        )
        raise
    print("Dataset metadata save completed.", flush=True)

    metadata_bytes = os.path.getsize(DATASET_PATH)
    bundle_bytes = metadata_bytes + shard_total_bytes

    print("\n" + "=" * width)
    print("A* PROBE DATASET".center(width))
    print("=" * width)
    print(
        f"Logical probe examples:                "
        f"{summary['logical_examples']:>12,}"
    )
    print(
        f"Unique activation positions:           "
        f"{summary['unique_activation_positions']:>12,}"
    )
    print(
        f"Attention-head activations shape:      "
        f"{attention_shape}"
    )
    print(
        f"MLP-output activations shape:          "
        f"{mlp_shape}"
    )
    print(
        f"Residual-stream activations shape:     "
        f"{residual_shape}"
    )
    print(
        f"Final hidden activations shape:        "
        f"{final_shape}"
    )
    print(
        f"Dataset metadata:                      "
        f"{DATASET_PATH}"
    )
    print(
        f"Activation shard directory:            "
        f"{shard_dir}"
    )
    print(
        f"Activation shard files:                "
        f"{len(shard_manifest):>12,}"
    )
    print(
        f"Metadata size:                         "
        f"{metadata_bytes / (1024 ** 3):.2f} GiB"
    )
    print(
        f"Activation shard size:                 "
        f"{shard_total_bytes / (1024 ** 3):.2f} GiB"
    )
    print(
        f"Total dataset bundle size:             "
        f"{bundle_bytes / (1024 ** 3):.2f} GiB"
    )
    print(
        "Activation coverage and numerical "
        "checks passed."
    )
    print(
        "Saved shard ranges are contiguous "
        "and complete."
    )
    print(
        "Input-number families are disjoint "
        "across train/validation/test."
    )
    print(
        "All eligible locations were collected "
        "without balancing or quotas."
    )
    print("=" * width)


if __name__ == "__main__":
    main()