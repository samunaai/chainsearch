# ChainSearch

Code for **On Reasoning Correctness Directions**.

📄 **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

<p align="center">
  <a href="assets/chainsearch.pdf">
    <img src="assets/chainsearch.webp" alt="ChainSearch task overview" width="60%">
  </a>
</p>

## Overview

ChainSearch is a controlled arithmetic reasoning task that extends Countdown with explicit search and backtracking. Models combine numbers to reach a target, explore alternative paths, and return to earlier states when they encounter dead ends.

We train a 3-million-parameter language model on this task to investigate whether internal linear directions that encode reasoning correctness can also improve reasoning through steering. Using linear probes, mean activation differences, and direct optimisation, we find that correctness can be linearly decodable without yielding a consistently useful direction for steering.

## Running the Experiments

Start by cloning the repository:

```bash
git clone https://github.com/samunaai/chainsearch.git
cd chainsearch
```

The main command flow is provided in [`commands.sh`](commands.sh). Refer to this file for the commands used to run the experiments.

<!-- TODO: Add environment setup and step-by-step instructions from commands.sh. -->

## Citation

If you use this work, please cite our paper:

<!-- TODO: Replace this placeholder with the arXiv BibTeX entry once available. -->

**On Reasoning Correctness Directions.** 2026. arXiv:XXXX.XXXXX.
