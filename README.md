# Optimizing the Bernstein Collocation Approach: Chebyshev-Gauss-Lobatto Nodes in Singular Perturbation

[![Published](https://img.shields.io/badge/Published-Physica%20Scripta%202025-blue?style=flat-square)](https://doi.org/10.1088/1402-4896/adb703)
[![DOI](https://img.shields.io/badge/DOI-10.1088%2F1402--4896%2Fadb703-green?style=flat-square)](https://doi.org/10.1088/1402-4896/adb703)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-red?style=flat-square)](LICENSE)
[![IOP Publishing](https://img.shields.io/badge/Journal-IOP%20Publishing-orange?style=flat-square)](https://iopscience.iop.org/journal/1402-4896)

> **Official implementation** of the paper:
> *"Optimizing the Bernstein Collocation Approach: Chebyshev-Gauss-Lobatto Nodes in Singular Perturbation"*
> **Kirti Beniwal** and Vivek Kumar
> *Physica Scripta*, Volume 100, Number 3 (2025), Article 035239
> IOP Publishing · DOI: [10.1088/1402-4896/adb703](https://doi.org/10.1088/1402-4896/adb703)

---

## Overview

This repository contains the complete Python implementation for solving **Singularly Perturbed Differential Equations (SPDEs)** using three competing numerical methods, with a focus on demonstrating the superiority of the **Bernstein Collocation Method (BCM) with Chebyshev-Gauss-Lobatto (CGL) nodes**.

SPDEs are differential equations containing a small positive parameter **ε (0 < ε ≪ 1)** multiplying the highest-order derivative. As ε → 0, the solution develops **sharp boundary layers** — regions of rapid change near the domain boundary — making these problems extremely challenging for standard numerical methods.

---

## The Problem

We solve the general singularly perturbed boundary value problem:

$$-\varepsilon u''(x) + r(x)u'(x) + s(x)u(x) = f(x), \quad x \in (a, b)$$

with Dirichlet boundary conditions:

$$u(a) = \alpha, \qquad u(b) = \beta$$

where $\varepsilon$ is a **small perturbation parameter**. The boundary layer width is approximately $O(\sqrt{\varepsilon})$, becoming thinner and steeper as $\varepsilon \to 0$.

---

## Methods Compared

| Method | Description | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **Variational (Galerkin)** | Weak formulation with Bernstein basis functions | Fast matrix assembly | Large errors for small ε |
| **BCM — Equispaced Nodes** | Collocation at uniform grid points | Simple to implement | Runge phenomenon; poor boundary layer resolution |
| **BCM — CGL Nodes** ⭐ | Collocation at Chebyshev-Gauss-Lobatto points | **Best accuracy; excellent stability** | Slightly higher setup cost |

### Why Chebyshev-Gauss-Lobatto Nodes?

CGL nodes are defined as:

$$x_i = \frac{1}{2}\left(1 + \cos\left(\frac{\pi i}{n}\right)\right), \quad i = 0, 1, \ldots, n$$

They **cluster near the endpoints** of the interval [0, 1] — precisely where boundary layers form. This clustering provides:
- **Higher resolution** exactly where it is needed most
- **Minimised Runge phenomenon** for high-degree polynomial approximations
- **Better conditioned system matrices**, leading to more stable solutions
- **Faster convergence** as polynomial degree increases

---

## Key Results

### Maximum Error for Example 1 (Linear Convection-Diffusion), N = 32

| ε | Variational Method | BCM Equispaced | BCM CGL Nodes |
|---|-------------------|----------------|---------------|
| 0.1 | 4.44e-09 | 2.07e-09 | **4.09e-10** |
| 0.01 | 3.59e-03 | 0.184 | **1.36e-03** |
| 0.001 | 0.83 | 0.75 | **0.37** |
| 0.0001 | 43.27 | 0.88 | **0.45** |
| 1e-05 | 35.87 | 0.88 | **0.44** |

> **BCM with CGL nodes achieves errors as small as 2.803 × 10⁻¹¹** for ε = 0.0001 at N = 64, compared to 7.862 × 10⁻⁸ for equispaced nodes and 5.267 × 10⁻⁶ for the variational method.

Statistical significance of improvements was confirmed via **paired t-tests** (p-values < 0.001 across all N and ε combinations).

---

## Repository Structure

```
bernstein-collocation-spde/
│
├── bernstein_collocation_SPDEs.py   # Main implementation — all 4 examples
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
│
├── outputs/                         # Generated plots (after running)
│   ├── example1_solutions.png
│   ├── example2_solutions.png
│   ├── example3_solutions.png
│   └── example4_solutions.png
│
└── paper/
    └── ps_100_3_035239.pdf          # Published paper (IOP Physica Scripta)
```

---

## Numerical Examples

### Example 1 — 1D Linear Convection-Diffusion
$$-\varepsilon u''(x) - 2u'(x) = 0, \quad u(0) = 1,\ u(1) = 0$$

**Exact solution:** $u(x) = \dfrac{e^{-2x/\varepsilon} - e^{-2/\varepsilon}}{1 - e^{-2/\varepsilon}}$

Applications: Electrochemical systems, chemical reaction kinetics, chromatography.

---

### Example 2 — Variable Coefficient SPDE (Double Boundary Layers)
$$-\varepsilon u''(x) + 2(2x-1)u'(x) + 4u(x) = 0, \quad u(0) = 1,\ u(1) = 1$$

**Exact solution:** $u(x) = e^{-2x(1-x)/\varepsilon}$

Features an interior turning point at x = 0.5 and simultaneous boundary layers at both endpoints.

---

### Example 3 — Nonlinear Reaction-Diffusion
$$-\varepsilon u''(x) + u(x) + u(x)^2 = e^{-2x/\sqrt{\varepsilon}}, \quad u(0) = 1,\ u(1) = e^{-1/\sqrt{\varepsilon}}$$

**Exact solution:** $u(x) = e^{-x/\sqrt{\varepsilon}}$

Demonstrates BCM's effectiveness on nonlinear SPDEs via Newton-Raphson iteration.

---

### Example 4 — SPDE with Trigonometric Forcing
$$-\varepsilon u''(x) + u(x) = -\cos^2(\pi x) - 2\varepsilon^2\pi^2\cos(2\pi x)$$

**Exact solution:** $u(x) = \dfrac{e^{-x/\sqrt{\varepsilon}} + e^{(x-1)/\sqrt{\varepsilon}}}{1 + e^{-1/\sqrt{\varepsilon}}} - \cos^2(\pi x)$

Multi-scale problem: oscillatory forcing + sharp boundary layers at both ends.

---

## Installation & Usage

### Requirements

```bash
pip install numpy scipy matplotlib
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

### Run All Examples

```bash
python bernstein_collocation_SPDEs.py
```

### Run a Specific Example

```python
from bernstein_collocation_SPDEs import run_example1, run_example2, run_example3, run_example4

# Example 1: Linear convection-diffusion with ε = 0.001
results = run_example1(epsilon=0.001, n=32)

# Access solution arrays
x      = results['x']
exact  = results['y_exact']
cheby  = results['y_cheby']

# Access errors
print(results['errors']['cheby'])   # (max_error, rmse)
print(results['errors']['equi'])
print(results['errors']['galerkin'])
```

### Convergence Study

```python
from bernstein_collocation_SPDEs import run_example1_convergence_study

# Replicate Table 1 from the paper (N = 32)
run_example1_convergence_study(n=32)

# Try with N = 64 for higher accuracy
run_example1_convergence_study(n=64)
```

### Expected Output (Example 1, ε = 0.01, n = 32)

```
============================================================
EXAMPLE 1 — Linear Convection-Diffusion  (ε = 0.01)
============================================================
  Variational Method    : Max Error = 3.59e-03,  RMSE = 8.45e-04
  BCM (Equispaced)      : Max Error = 1.84e-01,  RMSE = 2.31e-02
  BCM (CGL Nodes)       : Max Error = 1.36e-03,  RMSE = 1.98e-04
```

---

## Mathematical Background

### Bernstein Polynomials

The $k$-th Bernstein polynomial of degree $n$ is defined as:

$$B_{k,n}(t) = \binom{n}{k} t^k (1-t)^{n-k}, \quad t \in [0,1]$$

Key properties:
- **Non-negativity:** $B_{k,n}(t) \geq 0$ for all $t \in [0,1]$
- **Partition of unity:** $\sum_{k=0}^n B_{k,n}(t) = 1$
- **Symmetry:** $B_{k,n}(t) = B_{n-k,n}(1-t)$
- **Recursion:** $B_{k,n}(t) = (1-t)B_{k,n-1}(t) + t B_{k-1,n-1}(t)$

### Condition Number Analysis

The condition number $K(A) = \|A\| \cdot \|A^{-1}\|$ measures numerical stability. The paper demonstrates that BCM with CGL nodes consistently achieves **lower condition numbers** than equispaced nodes, directly explaining its superior accuracy — lower condition numbers mean less sensitivity to rounding errors.

### Error Bound (Theorem 4)

If $u(x)$ is sufficiently smooth on $[0,1]$, the Bernstein approximation at CGL nodes satisfies:

$$|B_n(u, x) - u(x)| < \xi$$

for sufficiently large $n$, confirmed analytically via Korovkin's second theorem.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{beniwal2025optimizing,
  title   = {Optimizing the Bernstein collocation approach:
             Chebyshev-Gauss-Lobatto nodes in singular perturbation},
  author  = {Beniwal, Kirti and Kumar, Vivek},
  journal = {Physica Scripta},
  volume  = {100},
  number  = {3},
  pages   = {035239},
  year    = {2025},
  publisher = {IOP Publishing},
  doi     = {10.1088/1402-4896/adb703},
  url     = {https://doi.org/10.1088/1402-4896/adb703}
}
```

---

## Related Work

This paper is part of a broader research programme on **numerical methods and machine learning for differential equations**:

- *Gradient-based Physics-Informed Neural Networks* — 3rd Congress on Intelligent Systems, Springer, 2022
- *Solving Fractional PDEs using Gradient-based Neural Networks* — NUMTA 2023, Italy
- *Predator-Prey-Scavenger Model using Neural Networks* — Int. J. Applied & Computational Mathematics, Springer, 2025
- *Tropical Cyclone Energy Prediction using ANNs* — MAUSAM Journal, 2025

---

## Author

**Kirti Beniwal**
Department of Applied Mathematics
Delhi Technological University, Delhi-110042, India
Email: kirtibnwl1912@gmail.com

**Supervisor: Dr. Vivek Kumar**
Associate Professor, Department of Applied Mathematics
Delhi Technological University

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*This implementation accompanies the peer-reviewed publication in Physica Scripta (IOP Publishing). The code is provided for reproducibility and to support further research in numerical methods for singularly perturbed problems.*
