# Artin S Constant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Raj123-0/Artin-s-Constant/actions/workflows/ci.yml/badge.svg)](https://github.com/Raj123-0/Artin-s-Constant/actions)


High-precision mathematical computation and OEIS digit generator for Artin S Constant.

## Overview

`Artin-s-Constant` implements high-precision evaluation of the **Artin S Constant** using arbitrary-precision mathematical routines (`mpmath` and C-accelerated `gmpy2`). The engine generates exact decimal digits, formats standard OEIS b-file sequences, and includes an automated performance benchmark.

## Features

- **Arbitrary-Precision Calculation**: Configurable digit targets with optimized guard precision.
- **OEIS b-file Output**: Generates 1-based index sequence files ready for OEIS submission.
- **Performance Profiling**: Built-in benchmark suite to evaluate digits/sec scaling.
- **Robust CLI**: Easy command-line interface with argument parsing.

## Installation

```bash
git clone https://github.com/Raj123-0/Artin-s-Constant.git
cd Artin-s-Constant
pip install -r requirements.txt
```

## Usage

Calculate digits with the CLI:

```bash
python "Artin's Constant.py" --digits 1000
```

Run precision benchmarks:

```bash
python benchmarks/bench_precision.py
```

Run automated tests:

```bash
pytest tests/
```

## License

This project is licensed under the [MIT License](LICENSE).
