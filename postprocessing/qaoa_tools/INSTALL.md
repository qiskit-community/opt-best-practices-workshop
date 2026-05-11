# Installation Guide for QAOA Tools

This guide will help you install all required dependencies for the QAOA postprocessing toolkit.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Virtual environment tool (venv, conda, etc.)

## Step-by-Step Installation

### 1. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv qaoa_env
source qaoa_env/bin/activate  # On Windows: qaoa_env\Scripts\activate

# OR using conda
conda create -n qaoa_env python=3.9
conda activate qaoa_env
```

### 2. Navigate to the qaoa_tools Directory

```bash
cd postprocessing/qaoa_tools/
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- qiskit>=1.0.0
- qiskit-ibm-runtime>=0.20.0
- qiskit-aer>=0.13.0
- numpy>=1.24.0
- scipy>=1.10.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- networkx>=3.0
- rustworkx>=0.13.0

### 4. Install Optional Dependencies

#### Gurobi (Commercial Solver)

If you have a Gurobi license:

```bash
pip install gurobipy
```

#### QAOA Training Pipeline

If you have access to the qaoa-training-pipeline package:

```bash
# Install from local source
cd /path/to/qaoa-training-pipeline
pip install -e .

# OR install from git repository
pip install git+https://github.com/your-org/qaoa-training-pipeline.git
```

#### qopt-best-practices

If you have access to the qopt-best-practices package:

```bash
# Install from local source
cd /path/to/qopt-best-practices
pip install -e .

# OR install from git repository
pip install git+https://github.com/your-org/qopt-best-practices.git
```

### 5. Set Up IBM Quantum Credentials

```bash
export QISKIT_IBM_TOKEN="your_ibm_quantum_token_here"
export QISKIT_IBM_INSTANCE="your_instance_here"
```

To make these permanent, add them to your shell profile (~/.bashrc, ~/.zshrc, etc.):

```bash
echo 'export QISKIT_IBM_TOKEN="your_token"' >> ~/.bashrc
echo 'export QISKIT_IBM_INSTANCE="your_instance"' >> ~/.bashrc
source ~/.bashrc
```

### 6. Verify Installation

```bash
python -c "from qaoa_tools import *; print('✓ Installation successful!')"
```

## Troubleshooting

### ModuleNotFoundError: No module named 'qiskit_ibm_runtime'

**Solution**: Install qiskit-ibm-runtime:
```bash
pip install qiskit-ibm-runtime
```

### ModuleNotFoundError: No module named 'qaoa_training_pipeline'

**Solution**: This is an optional dependency. Either:
1. Install it from source if you have access
2. Use the toolkit without training features (some notebook cells will need to be skipped)

### ModuleNotFoundError: No module named 'qopt_best_practices'

**Solution**: This is an optional dependency. Either:
1. Install it from source if you have access
2. The toolkit will fall back to standard Qiskit transpilation

### Import errors in Jupyter notebook

**Solution**: Make sure your Jupyter kernel is using the correct Python environment:

```bash
# Install ipykernel in your environment
pip install ipykernel

# Add your environment as a Jupyter kernel
python -m ipykernel install --user --name=qaoa_env --display-name="Python (QAOA)"

# Then select this kernel in Jupyter
```

## Minimal Installation (Without Optional Packages)

If you want to use the toolkit without qaoa-training-pipeline or qopt-best-practices:

```bash
pip install qiskit>=1.0.0 qiskit-ibm-runtime>=0.20.0 qiskit-aer>=0.13.0
pip install numpy>=1.24.0 scipy>=1.10.0 pandas>=2.0.0
pip install matplotlib>=3.7.0 networkx>=3.0 rustworkx>=0.13.0
```

Then modify the notebook to skip cells that use training pipeline features.

## Testing Your Installation

Run the example script:

```bash
cd postprocessing/qaoa_tools/
python example_usage.py
```

Or open the Jupyter notebook:

```bash
cd postprocessing/
jupyter notebook postprocessing.ipynb
```

## Getting Help

If you encounter issues:
1. Check that all dependencies are installed: `pip list`
2. Verify Python version: `python --version` (should be 3.9+)
3. Check environment variables: `echo $QISKIT_IBM_TOKEN`
4. Review the error message carefully - it usually indicates which package is missing

## Package Versions

The toolkit has been tested with:
- Python 3.9, 3.10, 3.11
- Qiskit 1.0+
- qiskit-ibm-runtime 0.20+
- NumPy 1.24+
- Pandas 2.0+

For the most up-to-date compatibility information, refer to the requirements.txt file.