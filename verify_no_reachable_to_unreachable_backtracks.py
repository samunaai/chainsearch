#!/usr/bin/env python3

from pathlib import Path

import torch


LABEL_PATH = Path(
    "/workspace/user/chainsearch_archives/data_pts/astar/probing/"
    "exp34_generated_state_solvability_for_probing_samples.pt"
)


def count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def main() -> None:
    if not LABEL_PATH.is_file():
        raise FileNotFoundError(LABEL_PATH)

    print(f"Loading {LABEL_PATH}...")
    data = torch.load(
        LABEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not bool(data.get("complete", False)):
        raise RuntimeError("Reachability-label artifact is not complete.")

    backs = data["back_transitions"]

    required = {
        "pre_solvability_known",
        "post_solvability_known",
        "pre_is_forward_solvable",
        "post_is_forward_solvable",
        "pre_min_steps_remaining",
        "post_min_steps_remaining",
        "record_pos",
        "cache_idx",
        "target",
        "backtrack_idx",
    }
    missing = required - set(backs)
    if missing:
        raise KeyError(f"Missing BACK fields: {sorted(missing)}")

    pre_known = backs["pre_solvability_known"].to(torch.bool)
    post_known = backs["post_solvability_known"].to(torch.bool)
    pre_reachable = backs["pre_is_forward_solvable"].to(torch.bool)
    post_reachable = backs["post_is_forward_solvable"].to(torch.bool)

    paired = pre_known & post_known

    reachable_to_reachable = paired & pre_reachable & post_reachable
    reachable_to_unreachable = paired & pre_reachable & ~post_reachable
    unreachable_to_reachable = paired & ~pre_reachable & post_reachable
    unreachable_to_unreachable = paired & ~pre_reachable & ~post_reachable

    counts = {
        "reachable_to_reachable": count(reachable_to_reachable),
        "reachable_to_unreachable": count(reachable_to_unreachable),
        "unreachable_to_reachable": count(unreachable_to_reachable),
        "unreachable_to_unreachable": count(unreachable_to_unreachable),
        "known_pairs": count(paired),
        "excluded_pairs": count(~paired),
    }

    transition_sum = (
        counts["reachable_to_reachable"]
        + counts["reachable_to_unreachable"]
        + counts["unreachable_to_reachable"]
        + counts["unreachable_to_unreachable"]
    )
    if transition_sum != counts["known_pairs"]:
        raise RuntimeError("BACK transition counts do not sum correctly.")

    print()
    print("=" * 72)
    print("BACK REACHABILITY TRANSITION CHECK".center(72))
    print("=" * 72)
    print(
        f"Known pre/post pairs:              "
        f"{counts['known_pairs']:>12,}"
    )
    print(
        f"Reachable   -> reachable:          "
        f"{counts['reachable_to_reachable']:>12,}"
    )
    print(
        f"Reachable   -> unreachable:        "
        f"{counts['reachable_to_unreachable']:>12,}"
    )
    print(
        f"Unreachable -> reachable:          "
        f"{counts['unreachable_to_reachable']:>12,}"
    )
    print(
        f"Unreachable -> unreachable:        "
        f"{counts['unreachable_to_unreachable']:>12,}"
    )
    print(
        f"Excluded because a label unknown:  "
        f"{counts['excluded_pairs']:>12,}"
    )

    summary = data.get("summary", {})
    saved_count = summary.get("back_solvable_to_unsolvable")
    if saved_count is not None:
        saved_count = int(saved_count)
        print(
            f"Saved-summary reachable -> unreachable: "
            f"{saved_count:>8,}"
        )
        if saved_count != counts["reachable_to_unreachable"]:
            raise RuntimeError(
                "Recomputed count disagrees with the saved summary."
            )

    violations = torch.nonzero(
        reachable_to_unreachable,
        as_tuple=False,
    ).flatten()

    print()
    if not violations.numel():
        print(
            "PASS — no valid, fully labelled BACK transition moved "
            "from a reachable state to an unreachable state."
        )
        print("=" * 72)
        return

    print(f"FAIL — found {violations.numel():,} violating BACK transitions.")
    print("First violating rows")

    for row_tensor in violations[:10]:
        row = int(row_tensor.item())
        print(
            f"  row={row:,} "
            f"record_pos={int(backs['record_pos'][row]):,} "
            f"cache_idx={int(backs['cache_idx'][row]):,} "
            f"BACK={int(backs['backtrack_idx'][row])} "
            f"target={int(backs['target'][row])} "
            f"pre_steps={int(backs['pre_min_steps_remaining'][row])} "
            f"post_steps={int(backs['post_min_steps_remaining'][row])}"
        )

    print("=" * 72)
    raise SystemExit(1)


if __name__ == "__main__":
    main()