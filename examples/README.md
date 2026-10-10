# Runnable examples

Install the root `requirements.txt` once. No installation of `rebunet` is needed.
From the project root:

```bash
python examples/train_synthetic.py --out results/api_example
python examples/plan_experiments.py --dataset m5 --out results/example_plan
```

The first example generates 48 series of 224 days, checks chronological splits,
trains for two epochs, scores held-out targets and writes figures. It exercises
the public API without calling a shell subprocess. It is not a paper accuracy
result. Use a fresh output directory for each run.

The second example writes the full five-seed experiment plan without requiring
real datasets or starting training. A plan does not imply completed experiments.

For the equivalent Bash workflow, run:

```bash
bash scripts/01_train_synthetic.sh --out results/shell_example
```

For real datasets and the full experiment procedure, see the root README.
