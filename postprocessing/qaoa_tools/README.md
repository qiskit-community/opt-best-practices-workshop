# QAOA Postprocessing Tools

A modular toolkit for analyzing and postprocessing Quantum Approximate Optimization Algorithm (QAOA) results. This package provides comprehensive utilities for distribution analysis, visualization, and quantum error mitigation.

## 📁 Package Structure

```
qaoa_tools/
├── __init__.py                 # Package initialization and exports
├── utils_general.py            # Core utilities and data structures
├── postprocessor_greedy.py     # Greedy one-flip postprocessor
├── postprocessor_mqc.py        # MQC-style multi-qubit correction
├── visualization.py            # Plotting and visualization functions
├── analysis.py                 # End-to-end analysis workflow
├── requirements.txt            # Package dependencies
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install Dependencies

```bash
# Navigate to the qaoa_tools directory
cd postprocessing/qaoa_tools/

# Install required packages
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced functionality:

```bash
# Gurobi (commercial solver, requires license)
pip install gurobipy

# qopt-best-practices (if available)
# Install from source or private repository

# qaoa-training-pipeline (if available)
# Install from source or private repository
```

### Environment Variables

For IBM Quantum access:

```bash
export QISKIT_IBM_TOKEN="your_token_here"
export QISKIT_IBM_INSTANCE="your_instance_here"
```

## 📖 Usage

### Basic Import

```python
# Import from parent directory
import sys
sys.path.append('..')
from qaoa_tools import *

# Or import specific components
from qaoa_tools import (
    generate_d_regular_graph,
    get_backend,
    build_bound_executable_circuits,
    run_sampler_circuits,
    analyze_and_report,
    HighProbGreedy1Flip,
    MQCStyleMultiQubitCorrection,
    plot_training_diagnostics,
)
```

### Quick Start Example

```python
from pathlib import Path
import datetime
from qaoa_tools import *

# Configuration
CONFIG = {
    "USE_REAL_BACKEND": True,
    "RUN_MODE": "hardware",
    "BACKEND_NAME": "ibm_marrakesh",
    "GRAPH_N": 25,
    "GRAPH_D": 4,
    "GRAPH_SEED": 7,
    "REPS": 1,
    "SEED": 7,
    "SHOTS": 20000,
    "OPTIMIZATION_LEVEL": 1,
    "CLASSICAL_METHOD": "fast_heuristic",
    "USE_SAT_MAPPING": True,
    "SAT_TIMEOUT": 10,
    "PLOT_TOPK": 40,
    "SAVE_DIR": "qaoa_runs",
}

# Setup output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(CONFIG["SAVE_DIR"]) / f"run_{timestamp}"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Generate problem graph
graph = generate_d_regular_graph(
    CONFIG["GRAPH_N"], 
    CONFIG["GRAPH_D"], 
    CONFIG["GRAPH_SEED"], 
    auto_fix=True
)

# Get classical baseline
classical_opt, classical_bits = maxcut_exact_ilp(graph)
print(f"Classical optimal cut: {classical_opt}")

# Get backend
real_backend, exec_backend, backend_name = get_backend(CONFIG)

# ... (continue with QAOA training, circuit building, execution)

# Run comprehensive analysis
report = analyze_and_report(
    outdir=OUTDIR,
    config=CONFIG,
    graph=graph,
    backend_name=backend_name,
    circuits=circuits,
    artifacts=artifacts,
    classical_opt=classical_opt,
    sat_map=sat_map,
    sat_dir=sat_dir,
    postselect={"method": "mass", "mass_keep": 0.90},
)

# Access results
print(report["summaries_df"])
print(report["post_suite_df"])
```

## 🔧 Module Details

### `utils_general.py`

Core utilities including:
- **Data Structures**: `VariantArtifacts`, `DistributionRow`, `SummaryStats`
- **Graph Operations**: Graph generation, cut value calculation
- **Classical Baselines**: Fast heuristic and exact ILP solvers
- **Backend Management**: IBM Quantum and simulator setup
- **SAT Mapping**: Qubit mapping optimization
- **Circuit Building**: QAOA circuit construction and transpilation
- **Distribution Analysis**: Probability analysis and statistics

### `postprocessor_greedy.py`

Implements `HighProbGreedy1Flip` postprocessor:
- Applies greedy 1-bit flip local search to improve solutions
- Aggregates probability mass for converged solutions
- Reference: Chancellor 2017

### `postprocessor_mqc.py`

Implements `MQCStyleMultiQubitCorrection` postprocessor:
- Multi-qubit error correction using reference solutions
- Identifies connected components in difference graphs
- Applies group flips to improve objective values
- Reference: Ayanzadeh et al. Sci Rep 11, 16119 (2021)

### `visualization.py`

Plotting functions:
- `draw_graph()`: Visualize problem graphs
- `plot_backend_subset()`: Show qubit topology with highlighted subsets
- `plot_topk_bars()`: Bar chart of top bitstrings
- `plot_full_stem()`: Stem plot of full distribution
- `plot_training_diagnostics()`: Comprehensive training analysis

### `analysis.py`

Main analysis driver:
- `analyze_and_report()`: End-to-end analysis workflow
  - Raw distribution analysis
  - Post-selection (high-probability filtering)
  - Postprocessing suite application
  - Comprehensive reporting and visualization

## 🎯 Postprocessing Configuration

Configure postprocessors in your config dictionary:

```python
CONFIG["POSTPROCESSORS"] = [
    HighProbGreedy1Flip(),
    MQCStyleMultiQubitCorrection(
        top_m=500,
        max_passes=3,
        reference_mode="hybrid",  # "prob", "objective", or "hybrid"
    ),
]

# Post-selection policy
postselect_policy = {
    "method": "mass",        # "mass" or "threshold"
    "mass_keep": 0.90,       # Keep top 90% probability mass
    "prob_threshold": None,  # Alternative: threshold value
    "k_max": 2000           # Maximum number of bitstrings
}
```

## 📊 Output Structure

The analysis generates organized outputs:

```
output_directory/
├── plots/
│   ├── graph.png
│   ├── backend_subset.png
│   ├── circuit_*.png
│   ├── dist_full_*.png
│   ├── dist_top_*.png
│   └── dist_*_post_*.png
├── tables/
│   ├── metrics_*.csv
│   ├── summaries_*.csv
│   ├── post_*.csv
│   └── *.html (formatted tables)
└── data/
    ├── raw_artifacts_*.json
    ├── analysis_summary_*.json
    └── postprocessed_suite_*.json
```

## 🔬 Key Features

### 1. Modular Design
- Clean separation of concerns
- Easy to extend and maintain
- Reusable components

### 2. Comprehensive Analysis
- Raw distribution statistics
- Post-selection filtering
- Multiple postprocessing methods
- Detailed visualizations

### 3. Flexible Configuration
- Hardware or simulator execution
- Multiple backend options
- Configurable postprocessing pipeline
- Customizable output

### 4. Production Ready
- Type hints throughout
- Comprehensive documentation
- Error handling
- Logging and progress tracking

## 📄 License

See the main project LICENSE file.

## Acknowledgments

Postprocessing methods based on:
- Chancellor 2017 (Greedy local search)
- Ayanzadeh et al. 2021 (MQC-style correction)

## 📧 Support

For issues or questions, please refer to the main project documentation.