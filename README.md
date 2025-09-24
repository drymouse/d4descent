# Official Implementation of "Design for Descent: What Makes a Shape Grammar Easy to Optimize?" Siggraph Asia 2025

## Installation

```bash
git clone https://github.com/milmillin/d4descent.git
cd d4descent
uv sync
```

## Running

Running the experiments requires two steps: generating the shell scripts for the experiments and running them. This can be done with the following commands:

```bash
cd runs
uv run python _rungen_arclines.py
./_generated/arclines/AL-F_OneComp.sh
```

The `_rungen_*.py` will generate the shell scripts for the experiments. It will print all the generated shell scripts which you can then run.
Three grammars are tested in the paper: `arclines` for Arclines, `trees` for Tree, and `ur` for UnionRect grammars.
The suffixes denote objective functions: `_topopt` for topology optimization, `_prompts` for optimizing with SDS, `_pngs` for optimzing towards images. No suffix means optimizing towards shapes.

For ablations, uncomment parts of the code in `_rungen_*.py` and rerun to regenerate the shell scripts.
