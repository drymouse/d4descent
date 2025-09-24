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

## Code Structure

The repository is organized as follows:

### Core Components
- **`src/d4descent/`** - Main package containing the optimization framework
  - `optimizer.py` - Core optimization algorithms and procedures
  - `scheduler.py` - Learning rate scheduling and optimization control
  - `context.py` - Execution context management
  - `metrics.py` - Evaluation metrics and logging
  - `visualizer.py` - Visualization tools for results and debugging
  - `object_collection.py` - Management of optimizable objects
  - `types.py` - Type definitions and interfaces
  - `util.py` - Utility functions and helpers

### Shape Grammars & Tasks
- **`src/d4descent/objects/`** - Implementation of shape grammar objects
  - `arclines.py` - Arclines grammar implementation with curves and connectors
  - `tree.py` - Tree grammar for hierarchical branching structures
  - `ur.py` - UnionRect grammar for axis-aligned rectangle compositions
- **`src/d4descent/tasks/`** - Task-specific implementations for different optimization objectives
  - `_base.py` - Base task interface and common functionality
  - `arclines.py` - Arclines-specific optimization tasks
  - `tree.py` - Tree grammar optimization tasks
  - `ur.py` - UnionRect grammar optimization tasks
- **`src/d4descent/losses/`** - Loss functions for various optimization targets
  - `_base.py` - Base loss function interface
  - `raster.py` - Rasterization-based losses for image targets
  - `sds.py` - Score Distillation Sampling loss for text-to-shape
  - `topopt.py` - Topology optimization losses for structural design
- **`src/d4descent/third_party/`** - Third-party implementations and utilities
  - `pid.py` - PID controller implementation
  - `sds.py` - SDS-specific utilities and helpers
  - `topopt.py` - Topology optimization utilities

### Configuration & Experiments
- **`configs/`** - YAML configuration files for different experiments
  - `losses/` - Loss function configurations
  - `tasks/` - Task-specific configurations
- **`runs/`** - Experiment generation and execution scripts
  - `_rungen_*.py` - Scripts to generate shell scripts for different grammars
  - `_generated/` - Auto-generated shell scripts for experiments

### Scripts & Data
- **`scripts/`** - Standalone optimization scripts for specific tasks
  - `optimize_pngs.py` - Image-based optimization
  - `optimize_prompts.py` - SDS prompt optimization
  - `optimize_shc.py` - Shape optimization
- **`data/`** - Input data for experiments
  - `arclines/` - Arclines target shapes
  - `pngs/` - Target images for optimization

### Output
- **`output/`** - Results from optimization runs (generated during execution)

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
