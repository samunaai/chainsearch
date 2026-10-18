# ChainSearch

Code for **On Reasoning Correctness Directions**.

📄 **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Overview

ChainSearch is a controlled arithmetic reasoning task that extends Countdown
with explicit search and backtracking. Models combine numbers to reach a
target, explore alternative paths, and return to earlier states when they
encounter dead ends.

We train a 3-million-parameter language model on this task to investigate
whether internal linear directions that encode reasoning correctness can
also improve reasoning through steering.

This repository contains the code for the task and the experiments presented
in the paper.

<!-- Add the task illustration at assets/chainsearch.png, then uncomment:
![ChainSearch task overview](assets/chainsearch.png)
-->

## Running the experiments

The main command flow is provided in [`commands.sh`](commands.sh).

<!-- Add setup instructions and the step-by-step command flow here. -->
