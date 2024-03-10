# Bernstein Optimizer

## Overview

This repository contains the implementation of the Bernstein Optimizer, a tool designed for optimizing control systems. The optimizer utilizes Bernstein polynomials to provide a robust and efficient method for solving optimization problems in control theory. The code is intended to accompany an upcoming paper detailing the methodology, applications and results.

## Getting Started

To use the Bernstein Optimizer, clone this repository to your local machine:

```
git clone https://github.com/your-username/bernstein-optimizer.git
```

Set up a Python environment and install the required dependencies:

```bash
cd bernstein-optimizer
python -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib
```

## Examples

The repository includes several example files demonstrating the application of the Bernstein Optimizer to various control problems. These examples are located in the `main_analytical_example_*.py` and `main_nonanalytical_example_*.py` files. Each example file provides a different scenario in which the optimizer is applied, to be detailed in the paper.

## Usage

To run an example, navigate to the repository directory and execute the corresponding Python file. For instance, to run the first analytical example:

```
python main_analytical_example_1.py
```

## Requirements

The Bernstein Optimizer requires the following Python libraries:

- NumPy
- SciPy
- MatPlotLib

Ensure that these libraries are installed on your system before running the examples.

## TODO

- [ ] Package the optimizer for distribution via PyPI (pip)
