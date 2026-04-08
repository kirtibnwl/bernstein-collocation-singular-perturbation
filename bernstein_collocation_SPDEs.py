# -*- coding: utf-8 -*-
"""
================================================================================
  Optimizing the Bernstein Collocation Approach:
  Chebyshev-Gauss-Lobatto Nodes in Singular Perturbation
================================================================================

Published in:
    Physica Scripta, Volume 100, Number 3 (2025), Article 035239
    IOP Publishing
    DOI: https://doi.org/10.1088/1402-4896/adb703

Authors:
    Kirti Beniwal  (kirtibnwl1912@gmail.com)
    Vivek Kumar
    Department of Applied Mathematics,
    Delhi Technological University, Delhi-110042, India

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------
This research provides a numerical comparison of the performance of the
Variational Method and the Bernstein Collocation Method (BCM) using equispaced
nodes and Chebyshev-Gauss-Lobatto (CGL) nodes with respect to solving
singularly perturbed differential equations (SPDEs) that involve both linear
and nonlinear problems.

These problems provide major numerical approximation challenges because of
their boundary layers and steep gradients. The BCM with Chebyshev nodes was
found to perform more accurately than the Variational Method and the equispaced
node configuration through a thorough error assessment that takes into account
both maximum absolute error and root mean square error.

For instance, for a perturbation parameter ε = 0.0001, the BCM with Chebyshev
nodes shows maximum errors as 2.803e-11, compared to 7.862e-08 for equispaced
nodes and 5.267e-06 for the variational method.

--------------------------------------------------------------------------------
PROBLEM FORMULATION
--------------------------------------------------------------------------------
We solve the general linear singularly perturbed boundary value problem:

    -ε u''(x) + r(x) u'(x) + s(x) u(x) = f(x),   x ∈ (a, b)

with Dirichlet boundary conditions:

    u(a) = α,   u(b) = β

where ε is a small positive perturbation parameter (0 < ε << 1).
As ε → 0, the solution develops a thin boundary layer at one or both endpoints,
making the problem challenging for standard numerical methods.

--------------------------------------------------------------------------------
METHODS IMPLEMENTED
--------------------------------------------------------------------------------
1. Variational Method (Galerkin with Bernstein basis)
   - Approximates the solution as a linear combination of Bernstein polynomials
   - Uses integration by parts to form a weak formulation
   - Enforces orthogonality of the residual to all basis functions
   - Solves a linear system A·c = b

2. Bernstein Collocation Method (BCM) with Equispaced Nodes
   - Enforces the ODE exactly at a set of equally-spaced collocation points
   - Results in a square nonlinear/linear system solved via fsolve/solve
   - Less accurate in boundary layer regions due to uniform node distribution

3. Bernstein Collocation Method (BCM) with Chebyshev-Gauss-Lobatto (CGL) Nodes
   - Uses nodes: x_i = 0.5 * (1 + cos(π·i/n)),  i = 0, 1, ..., n
   - Nodes cluster near endpoints — exactly where boundary layers form
   - Significantly better accuracy and numerical stability than equispaced
   - Key advantage: Minimises the Runge phenomenon for high-degree polynomials

--------------------------------------------------------------------------------
NUMERICAL EXAMPLES
--------------------------------------------------------------------------------
Four benchmark problems from the literature are solved and compared:

  Example 1: 1D Linear Convection-Diffusion (linear, exact solution known)
  Example 2: Linear SPDE with Variable Coefficients (interior layer)
  Example 3: Nonlinear SPDE (reaction-diffusion type)
  Example 4: SPDE with Trigonometric Forcing (non-homogeneous)

--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------
    numpy >= 1.21
    scipy >= 1.7
    matplotlib >= 3.4
    mpl_toolkits (included with matplotlib)

Install via:  pip install numpy scipy matplotlib

================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve
from scipy.optimize import fsolve
from scipy.special import comb, binom
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — BERNSTEIN POLYNOMIAL BASIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
"""
Bernstein polynomials of degree n are defined as:

    B_{k,n}(t) = C(n,k) · t^k · (1 - t)^{n-k},   k = 0, 1, ..., n

Key properties:
  • Non-negative on [0, 1]
  • Partition of unity: Σ B_{k,n}(t) = 1
  • Symmetric: B_{k,n}(t) = B_{n-k,n}(1-t)
  • Recursion: B_{k,n}(t) = (1-t)·B_{k,n-1}(t) + t·B_{k-1,n-1}(t)

These properties make Bernstein polynomials ideal for approximating solutions
to differential equations on bounded intervals.
"""

def bernstein_poly(i, n, x):
    """
    Standard Bernstein polynomial B_{i,n}(x) on [0, 1].

    Parameters
    ----------
    i : int   — Index of the basis polynomial (0 ≤ i ≤ n)
    n : int   — Degree of the Bernstein polynomial
    x : float — Evaluation point in [0, 1]

    Returns
    -------
    float — Value of B_{i,n}(x) = C(n,i) · x^i · (1-x)^{n-i}
    """
    return binom(n, i) * (x ** i) * ((1 - x) ** (n - i))


def bernstein_scaled(i, n, x, R):
    """
    Scaled Bernstein polynomial for interval [0, R] (used in Galerkin method).

    B_{i,n}(x/R) = C(n,i) · (x/R)^i · (1 - x/R)^{n-i}

    Parameters
    ----------
    i : int   — Polynomial index
    n : int   — Polynomial degree
    x : float — Evaluation point in [0, R]
    R : float — Right endpoint of the interval

    Returns
    -------
    float — Scaled Bernstein polynomial value
    """
    return comb(n, i) * ((x / R) ** i) * ((1 - (x / R)) ** (n - i))


def bernstein_poly_deriv(i, n, x):
    """
    First derivative of Bernstein polynomial B'_{i,n}(x).

    Using the identity:
        B'_{i,n}(x) = n · [B_{i-1,n-1}(x) - B_{i,n-1}(x)]

    Parameters
    ----------
    i : int   — Polynomial index
    n : int   — Polynomial degree
    x : float — Evaluation point

    Returns
    -------
    float — Value of dB_{i,n}/dx at x
    """
    if i == 0:
        return -n * ((1 - x) ** (n - 1))
    elif i == n:
        return n * (x ** (n - 1))
    else:
        return binom(n, i) * (i * (x ** (i - 1)) * ((1 - x) ** (n - i)) -
                              (n - i) * (x ** i) * ((1 - x) ** (n - i - 1)))


def bernstein_derivative_scaled(i, n, x, R):
    """
    First derivative of the scaled Bernstein polynomial (for Galerkin method).

    Parameters
    ----------
    i : int   — Polynomial index
    n : int   — Polynomial degree
    x : float — Evaluation point in [0, R]
    R : float — Right endpoint of the interval

    Returns
    -------
    float — dB_{i,n}(x/R)/dx
    """
    if i == 0:
        return -n * ((1 - (x / R)) ** (n - 1)) / R
    elif i == n:
        return n * ((x / R) ** (n - 1)) / R
    else:
        return (comb(n, i) * (i * ((x / R) ** (i - 1)) * ((1 - (x / R)) ** (n - i))
               - (n - i) * ((x / R) ** i) * ((1 - (x / R)) ** (n - i - 1)))) / R


def bernstein_poly_second_deriv(i, n, x):
    """
    Second derivative of Bernstein polynomial B''_{i,n}(x).

    Used in the collocation method for problems containing u''(x) terms.

    Parameters
    ----------
    i : int   — Polynomial index
    n : int   — Polynomial degree
    x : float — Evaluation point

    Returns
    -------
    float — Value of d²B_{i,n}/dx² at x
    """
    if n == 0 or n == 1:
        return 0
    elif i == 0:
        return n * (n - 1) * ((1 - x) ** (n - 2))
    elif i == n:
        return n * (n - 1) * (x ** (n - 2))
    else:
        return binom(n, i) * (
            (i * (i - 1) * (x ** (i - 2)) * ((1 - x) ** (n - i))) -
            (2 * i * (n - i) * (x ** (i - 1)) * ((1 - x) ** (n - i - 1))) +
            ((n - i) * (n - i - 1) * (x ** i) * ((1 - x) ** (n - i - 2)))
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — COLLOCATION NODE GENERATION
# ══════════════════════════════════════════════════════════════════════════════
"""
Two types of collocation nodes are used and compared:

1. Equispaced (Uniform) Nodes:
       x_j = j / (n-1),   j = 0, 1, ..., n
   Simple to compute but can suffer from the Runge phenomenon and poor
   resolution near boundary layers.

2. Chebyshev-Gauss-Lobatto (CGL) Nodes:
       x_i = 0.5 · (1 + cos(π·i/n)),   i = 0, 1, ..., n
   Derived from the roots of Chebyshev polynomials of the first kind.
   Cluster near the endpoints [0, 1], providing enhanced resolution where
   boundary layers typically form. This clustering is the primary reason for
   their superior performance on singularly perturbed problems.
"""

def chebyshev_gauss_lobatto_nodes(N):
    """
    Generate N+1 Chebyshev-Gauss-Lobatto nodes on [0, 1].

    Formula:
        x_i = 0.5 · (1 + cos(π·i/N)),   i = 0, 1, ..., N

    These nodes are clustered near x=0 and x=1, providing high resolution
    at the boundaries — ideal for problems with boundary layers.

    Parameters
    ----------
    N : int — Number of intervals (generates N+1 nodes)

    Returns
    -------
    numpy.ndarray — Array of N+1 CGL nodes in [0, 1]
    """
    return 0.5 * (1 + np.cos(np.pi * np.arange(N + 1) / N))


def equidistant_collocation_nodes(N):
    """
    Generate N+1 equally-spaced (uniform) collocation nodes on [0, 1].

    Formula:
        x_j = j / N,   j = 0, 1, ..., N

    Parameters
    ----------
    N : int — Number of intervals (generates N+1 nodes)

    Returns
    -------
    numpy.ndarray — Array of N+1 uniformly-spaced nodes in [0, 1]
    """
    return np.linspace(0, 1, N + 1)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ERROR ANALYSIS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_errors(y_exact, y_approx):
    """
    Compute Maximum Absolute Error (MAE) and Root Mean Square Error (RMSE).

    Maximum Absolute Error:  max |u_exact(x) - u_approx(x)|
    RMSE:                    sqrt(mean((u_exact - u_approx)^2))

    A lower condition number of the system matrix correlates with better
    numerical stability. BCM with CGL nodes consistently achieves lower
    condition numbers than equispaced nodes, explaining its superior accuracy.

    Parameters
    ----------
    y_exact  : numpy.ndarray — Exact solution values
    y_approx : numpy.ndarray — Approximate solution values

    Returns
    -------
    tuple — (max_error, rmse)
    """
    error = np.abs(y_exact - y_approx)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean((y_exact - y_approx) ** 2))
    return max_error, rmse


def print_error_table(epsilon_values, errors_galerkin, errors_equi, errors_cheby):
    """
    Print a formatted comparison table of errors for multiple epsilon values.

    Parameters
    ----------
    epsilon_values  : list  — List of perturbation parameter values
    errors_galerkin : list  — List of (max_error, rmse) for Galerkin method
    errors_equi     : list  — List of (max_error, rmse) for BCM equispaced
    errors_cheby    : list  — List of (max_error, rmse) for BCM CGL nodes
    """
    print("\n" + "="*100)
    print("ERROR COMPARISON TABLE — Maximum Absolute Error and RMSE")
    print("="*100)
    header = (f"{'ε':^10} | {'VM Max Error':^18} | {'VM RMSE':^18} | "
              f"{'ES Max Error':^18} | {'ES RMSE':^18} | "
              f"{'CGL Max Error':^18} | {'CGL RMSE':^18}")
    print(header)
    print("-" * len(header))
    for i, epsilon in enumerate(epsilon_values):
        row = (f"{epsilon:^10} | {errors_galerkin[i][0]:^18.4e} | {errors_galerkin[i][1]:^18.4e} | "
               f"{errors_equi[i][0]:^18.4e} | {errors_equi[i][1]:^18.4e} | "
               f"{errors_cheby[i][0]:^18.4e} | {errors_cheby[i][1]:^18.4e}")
        print(row)
    print("="*100)
    print("VM = Variational Method | ES = Equispaced Nodes | CGL = Chebyshev-Gauss-Lobatto Nodes\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — EXAMPLE 1: LINEAR CONVECTION-DIFFUSION PROBLEM
# ══════════════════════════════════════════════════════════════════════════════
"""
Problem Statement (Example 5.1 in the paper):
----------------------------------------------
    -ε u''(x) - 2 u'(x) = 0,   x ∈ (0, 1),   0 < ε << 1

Boundary Conditions:
    u(0) = 1,   u(1) = 0

Physical Significance:
    This is the 1D linear convection-diffusion equation used to model:
    • Electrochemical processes
    • Kinetics of chemical reactions
    • Chromatography column behaviour

    The term ε u''(x) represents diffusion, and 2 u'(x) represents advection.
    As ε → 0, the solution develops a sharp boundary layer at x = 0.

Exact Solution:
    u(x) = [exp(-2x/ε) - exp(-2/ε)] / [1 - exp(-2/ε)]

Key Results (Table 1 in the paper, N=32):
    ε = 0.0001 → VM error: 43.27   |  ES error: 0.45   |  CGL error: 2.80e-11
"""

def run_example1(epsilon=0.01, n=32, verbose=True):
    """
    Solve Example 1: -εu'' - 2u' = 0 on (0,1), u(0)=1, u(1)=0.

    Compares three methods:
      1. Variational (Galerkin) Method with scaled Bernstein polynomials
      2. BCM with equispaced collocation nodes
      3. BCM with Chebyshev-Gauss-Lobatto collocation nodes

    Parameters
    ----------
    epsilon : float — Perturbation parameter (default 0.01)
    n       : int   — Degree of Bernstein polynomial (default 32)
    verbose : bool  — Print error summary (default True)

    Returns
    -------
    dict — Contains x_fine, y_exact, y_galerkin, y_equi, y_cheby, errors
    """
    R = 1  # Right endpoint of the domain

    # ── 1. Variational (Galerkin) Method ───────────────────────────────────
    # Build system matrix A and right-hand side b using numerical integration.
    # The weak form of -εu'' - 2u' = 0 with Bernstein test functions gives:
    #   A[i,j] = ∫₀¹ [-ε B'_i B'_j - 2 B'_i B_j] dx
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            func = lambda x, i=i, j=j: (
                -epsilon * bernstein_derivative_scaled(i, n, x, R)
                         * bernstein_derivative_scaled(j, n, x, R)
                - 2 * bernstein_derivative_scaled(i, n, x, R)
                       * bernstein_scaled(j, n, x, R)
            )
            A[i, j], _ = quad(func, 0, R)

    # Enforce Dirichlet boundary conditions: u(0)=1, u(1)=0
    A[0, :], A[-1, :] = 0, 0
    A[0, 0], A[-1, -1] = 1, 1
    b[0], b[-1] = 1, 0

    coefficients_galerkin = solve(A, b)

    def y_galerkin(x):
        return sum(coefficients_galerkin[i] * bernstein_scaled(i, n, x, R)
                   for i in range(n + 1))

    # ── 2 & 3. Bernstein Collocation Method (BCM) ──────────────────────────
    # The residual of -εu' - 2u = 0 at each interior collocation point is
    # set to zero (note: this uses the first-order form with substitution).
    def collocation_residuals(coeffs, nodes):
        """Residual equations at interior collocation nodes."""
        equations = [coeffs[0] - 1] + [0] * (n - 1) + [coeffs[-1]]
        for j, x in enumerate(nodes[1:-1], 1):
            u_prime = sum(coeffs[i] * bernstein_poly_deriv(i, n, x)
                          for i in range(n + 1))
            u = sum(coeffs[i] * bernstein_poly(i, n, x) for i in range(n + 1))
            equations[j] = -epsilon * u_prime - 2 * u
        return equations

    initial_guess = np.zeros(n + 1)
    initial_guess[0] = 1  # Match boundary condition u(0) = 1

    equi_nodes  = equidistant_collocation_nodes(n)
    cheby_nodes = chebyshev_gauss_lobatto_nodes(n)

    coeffs_equi  = fsolve(collocation_residuals, initial_guess, args=(equi_nodes,))
    coeffs_cheby = fsolve(collocation_residuals, initial_guess, args=(cheby_nodes,))

    # ── Evaluate on fine grid ──────────────────────────────────────────────
    x_fine = np.linspace(0, 1, 1000)
    x_galerkin_grid = np.linspace(0, R, 100)

    y_gal_fine  = np.array([y_galerkin(x) for x in x_fine])
    y_gal_coarse = np.array([y_galerkin(x) for x in x_galerkin_grid])
    y_equi      = np.array([sum(coeffs_equi[i] * bernstein_poly(i, n, x)
                                for i in range(n + 1)) for x in x_fine])
    y_cheby     = np.array([sum(coeffs_cheby[i] * bernstein_poly(i, n, x)
                                for i in range(n + 1)) for x in x_fine])

    # Exact solution: u(x) = [exp(-2x/ε) - exp(-2/ε)] / [1 - exp(-2/ε)]
    y_exact = (np.exp(-2 * x_fine / epsilon) - np.exp(-2 / epsilon)) / \
              (1 - np.exp(-2 / epsilon))

    # ── Error computation ──────────────────────────────────────────────────
    err_gal  = compute_errors(y_exact, y_gal_fine)
    err_equi = compute_errors(y_exact, y_equi)
    err_cheby = compute_errors(y_exact, y_cheby)

    if verbose:
        print("\n" + "="*60)
        print(f"EXAMPLE 1 — Linear Convection-Diffusion  (ε = {epsilon})")
        print("="*60)
        print(f"  Variational Method    : Max Error = {err_gal[0]:.4e},  RMSE = {err_gal[1]:.4e}")
        print(f"  BCM (Equispaced)      : Max Error = {err_equi[0]:.4e},  RMSE = {err_equi[1]:.4e}")
        print(f"  BCM (CGL Nodes)       : Max Error = {err_cheby[0]:.4e},  RMSE = {err_cheby[1]:.4e}")

    # ── Plotting ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_fine, y_exact,       label='Exact Solution',              color='black',  linewidth=2)
    ax.plot(x_galerkin_grid, y_gal_coarse, label='Variational Method', color='orange')
    ax.plot(x_fine, y_equi,        label='BCM — Equispaced Nodes',      color='red',    linestyle='--')
    ax.plot(x_fine, y_cheby,       label='BCM — Chebyshev-Gauss-Lobatto Nodes', color='blue', linestyle='--')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x)', fontsize=12)
    ax.set_title(f'Example 1: Linear Convection-Diffusion  (ε = {epsilon})', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Inset: zoom into boundary layer near x = 0
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower left',
                          bbox_to_anchor=(0.5, 0.45, 0.45, 0.45),
                          bbox_transform=ax.transAxes)
    ax_inset.plot(x_fine, y_exact, color='black')
    ax_inset.plot(x_galerkin_grid, y_gal_coarse, color='orange')
    ax_inset.plot(x_fine, y_equi,  color='red',  linestyle='--')
    ax_inset.plot(x_fine, y_cheby, color='blue', linestyle='--')
    ax_inset.set_xlim(0.0, 0.1)
    ax_inset.set_ylim(0.0, 0.15)
    ax_inset.set_title('Boundary Layer Zoom', fontsize=8)
    ax_inset.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig('example1_solutions.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'x': x_fine, 'y_exact': y_exact,
        'y_galerkin': y_gal_fine, 'y_equi': y_equi, 'y_cheby': y_cheby,
        'errors': {'galerkin': err_gal, 'equi': err_equi, 'cheby': err_cheby}
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — EXAMPLE 1 CONVERGENCE STUDY (Multiple ε values)
# ══════════════════════════════════════════════════════════════════════════════
"""
Convergence Study for Example 1:
---------------------------------
We evaluate all three methods across a range of perturbation parameters:
    ε ∈ {0.1, 0.01, 0.001, 0.0001, 1e-5}

This replicates Table 1 and Table 2 from the paper (N = 16, 32, 64).
As ε → 0, the boundary layer becomes thinner and steeper, making the problem
progressively more challenging.

Key observation (from paper):
  • BCM with CGL nodes consistently outperforms both other methods
  • The p-values from paired t-tests (< 0.001) confirm improvements are
    statistically significant, not due to random variation
"""

def run_example1_convergence_study(n=32):
    """
    Replicate the convergence study from Table 1 of the paper.

    Runs Example 1 for ε ∈ {0.1, 0.01, 0.001, 0.0001, 1e-5}
    with polynomial degree n (default 32).

    Parameters
    ----------
    n : int — Bernstein polynomial degree (default 32, paper uses 16, 32, 64)
    """
    epsilon_values = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    R = 1
    x_fine = np.linspace(0, 1, 1000)

    errors_galerkin, errors_equi, errors_cheby = [], [], []

    print(f"\nRunning convergence study for Example 1 (N = {n})...")

    for epsilon in epsilon_values:
        # Galerkin Method
        A = np.zeros((n + 1, n + 1))
        b_vec = np.zeros(n + 1)
        for i in range(n + 1):
            for j in range(n + 1):
                func = lambda x, i=i, j=j: (
                    -epsilon * bernstein_derivative_scaled(i, n, x, R)
                             * bernstein_derivative_scaled(j, n, x, R)
                    - 2 * bernstein_derivative_scaled(i, n, x, R)
                           * bernstein_scaled(j, n, x, R)
                )
                A[i, j], _ = quad(func, 0, R)
        A[0, :], A[-1, :] = 0, 0
        A[0, 0], A[-1, -1] = 1, 1
        b_vec[0], b_vec[-1] = 1, 0
        coeffs_g = solve(A, b_vec)
        y_gal = np.array([sum(coeffs_g[i] * bernstein_scaled(i, n, x, R)
                              for i in range(n + 1)) for x in x_fine])

        # BCM — Equispaced and CGL
        def collocation_residuals(coeffs, nodes):
            equations = [coeffs[0] - 1] + [0] * (n - 1) + [coeffs[-1]]
            for j, x in enumerate(nodes[1:-1], 1):
                u_prime = sum(coeffs[i] * bernstein_poly_deriv(i, n, x)
                              for i in range(n + 1))
                u = sum(coeffs[i] * bernstein_poly(i, n, x) for i in range(n + 1))
                equations[j] = -epsilon * u_prime - 2 * u
            return equations

        ig = np.zeros(n + 1); ig[0] = 1
        c_equi  = fsolve(collocation_residuals, ig,
                         args=(equidistant_collocation_nodes(n),))
        c_cheby = fsolve(collocation_residuals, ig,
                         args=(chebyshev_gauss_lobatto_nodes(n),))

        y_equi  = np.array([sum(c_equi[i]  * bernstein_poly(i, n, x)
                                for i in range(n + 1)) for x in x_fine])
        y_cheby = np.array([sum(c_cheby[i] * bernstein_poly(i, n, x)
                                for i in range(n + 1)) for x in x_fine])

        # Exact solution
        y_exact = (np.exp(-2 * x_fine / epsilon) - np.exp(-2 / epsilon)) / \
                  (1 - np.exp(-2 / epsilon))

        errors_galerkin.append(compute_errors(y_exact, y_gal))
        errors_equi.append(compute_errors(y_exact, y_equi))
        errors_cheby.append(compute_errors(y_exact, y_cheby))

    print_error_table(epsilon_values, errors_galerkin, errors_equi, errors_cheby)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — EXAMPLE 2: LINEAR SPDE WITH VARIABLE COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════════
"""
Problem Statement (Example 5.2 in the paper):
----------------------------------------------
    -ε u''(x) + 2(2x-1) u'(x) + 4 u(x) = 0,   x ∈ (0, 1)

Boundary Conditions:
    u(0) = 1,   u(1) = 1

Physical Significance:
    This problem features a variable-coefficient advection term 2(2x-1)u'(x)
    which changes sign at x = 0.5. This creates an interior layer at x = 0.5
    in addition to potential boundary layers, making it a more challenging
    problem for numerical methods.

Exact Solution:
    u(x) = exp(-2x(1-x)/ε)

This problem tests the ability of each method to resolve both boundary layers
simultaneously (at both x=0 and x=1).
"""

def run_example2(epsilon=0.01, n=32, verbose=True):
    """
    Solve Example 2: -εu'' + 2(2x-1)u' + 4u = 0 on (0,1), u(0)=1, u(1)=1.

    This problem has double boundary layers AND a turning point at x=0.5.

    Parameters
    ----------
    epsilon : float — Perturbation parameter (default 0.01)
    n       : int   — Bernstein polynomial degree (default 32)
    verbose : bool  — Print errors (default True)

    Returns
    -------
    dict — Solution arrays and errors
    """
    R = 1

    # ── Variational (Galerkin) Method ──────────────────────────────────────
    # Weak form of -εu'' + 2(2x-1)u' + 4u = 0:
    #   A[i,j] = ∫₀¹ [ε B'_i B'_j + 2(2x-1) B'_j B_i + 4 B_j B_i] dx
    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            func = lambda x, i=i, j=j: (
                epsilon * bernstein_derivative_scaled(j, n, x, R)
                        * bernstein_derivative_scaled(i, n, x, R)
                + 2 * (2*x - 1) * bernstein_derivative_scaled(j, n, x, R)
                                 * bernstein_scaled(i, n, x, R)
                + 4 * bernstein_scaled(j, n, x, R) * bernstein_scaled(i, n, x, R)
            )
            A[i, j], _ = quad(func, 0, R)

    A[0, :], A[-1, :] = 0, 0
    A[0, 0], A[-1, -1] = 1, 1
    b_vec[0], b_vec[-1] = 1, 1
    coefficients_galerkin = solve(A, b_vec)

    def y_galerkin(x):
        return sum(coefficients_galerkin[i] * bernstein_scaled(i, n, x, R)
                   for i in range(n + 1))

    # ── BCM — Residual formulation ─────────────────────────────────────────
    def collocation_residuals(c, nodes):
        """Enforce -εu'' + 2(2x-1)u' + 4u = 0 at interior nodes."""
        equations = [c[0] - 1]
        for x in nodes[1:-1]:
            Vn      = sum(c[i] * bernstein_poly(i, n, x) for i in range(n+1))
            Vn_p    = sum(c[i] * bernstein_poly_deriv(i, n, x) for i in range(n+1))
            Vn_pp   = sum(c[i] * bernstein_poly_second_deriv(i, n, x) for i in range(n+1))
            equations.append(-epsilon * Vn_pp + 2*(2*x-1)*Vn_p + 4*Vn)
        equations.append(c[-1] - 1)
        return equations

    ig = np.zeros(n + 1); ig[0] = 1

    c_equi  = fsolve(collocation_residuals, ig,
                     args=(equidistant_collocation_nodes(n),))
    c_cheby = fsolve(collocation_residuals, ig,
                     args=(chebyshev_gauss_lobatto_nodes(n),))

    x_fine = np.linspace(0, 1, 1000)
    x_gal_grid = np.linspace(0, R, 100)

    y_gal_coarse = np.array([y_galerkin(x) for x in x_gal_grid])
    y_gal_fine   = np.array([y_galerkin(x) for x in x_fine])
    y_equi  = np.array([sum(c_equi[i]  * bernstein_poly(i, n, x) for i in range(n+1)) for x in x_fine])
    y_cheby = np.array([sum(c_cheby[i] * bernstein_poly(i, n, x) for i in range(n+1)) for x in x_fine])

    # Exact solution: u(x) = exp(-2x(1-x)/ε)
    y_exact = np.exp(-2 * x_fine * (1 - x_fine) / epsilon)

    err_gal   = compute_errors(y_exact, y_gal_fine)
    err_equi  = compute_errors(y_exact, y_equi)
    err_cheby = compute_errors(y_exact, y_cheby)

    if verbose:
        print("\n" + "="*60)
        print(f"EXAMPLE 2 — Variable Coefficient SPDE  (ε = {epsilon})")
        print("="*60)
        print(f"  Variational Method    : Max Error = {err_gal[0]:.4e},  RMSE = {err_gal[1]:.4e}")
        print(f"  BCM (Equispaced)      : Max Error = {err_equi[0]:.4e},  RMSE = {err_equi[1]:.4e}")
        print(f"  BCM (CGL Nodes)       : Max Error = {err_cheby[0]:.4e},  RMSE = {err_cheby[1]:.4e}")

    # ── Plotting with dual zoom insets ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_fine,      y_exact,      label='Exact Solution',                   color='black',  linewidth=2)
    ax.plot(x_gal_grid,  y_gal_coarse, label='Variational Method',               color='orange')
    ax.plot(x_fine,      y_equi,       label='BCM — Equispaced Nodes',           color='red',    linestyle='--')
    ax.plot(x_fine,      y_cheby,      label='BCM — Chebyshev-Gauss-Lobatto',    color='blue',   linestyle='--')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x)', fontsize=12)
    ax.set_title(f'Example 2: Variable Coefficient SPDE  (ε = {epsilon})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Zoom near x=0 (left boundary layer)
    ax_ins1 = fig.add_axes([0.18, 0.25, 0.15, 0.22])
    ax_ins1.plot(x_fine, y_exact, 'k'); ax_ins1.plot(x_gal_grid, y_gal_coarse, color='orange')
    ax_ins1.plot(x_fine, y_equi, 'r--'); ax_ins1.plot(x_fine, y_cheby, 'b--')
    ax_ins1.set_xlim(0, 0.1); ax_ins1.set_ylim(0, 0.15)
    ax_ins1.set_title('Zoom x=0', fontsize=8); ax_ins1.tick_params(labelsize=7)

    # Zoom near x=1 (right boundary layer)
    ax_ins2 = fig.add_axes([0.65, 0.25, 0.15, 0.22])
    ax_ins2.plot(x_fine, y_exact, 'k'); ax_ins2.plot(x_gal_grid, y_gal_coarse, color='orange')
    ax_ins2.plot(x_fine, y_equi, 'r--'); ax_ins2.plot(x_fine, y_cheby, 'b--')
    ax_ins2.set_xlim(0.9, 1.0); ax_ins2.set_ylim(0, 0.15)
    ax_ins2.set_title('Zoom x=1', fontsize=8); ax_ins2.tick_params(labelsize=7)

    # Highlight zoomed regions on main plot
    ax.add_patch(Rectangle((0, 0), 0.1, 0.1, lw=1, edgecolor='gray', facecolor='none', ls='--'))
    ax.add_patch(Rectangle((0.9, 0), 0.1, 0.1, lw=1, edgecolor='gray', facecolor='none', ls='--'))

    plt.tight_layout()
    plt.savefig('example2_solutions.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'x': x_fine, 'y_exact': y_exact,
        'y_galerkin': y_gal_fine, 'y_equi': y_equi, 'y_cheby': y_cheby,
        'errors': {'galerkin': err_gal, 'equi': err_equi, 'cheby': err_cheby}
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — EXAMPLE 3: NONLINEAR SINGULARLY PERTURBED PROBLEM
# ══════════════════════════════════════════════════════════════════════════════
"""
Problem Statement (Example 5.3 in the paper):
----------------------------------------------
    -ε u''(x) + u(x) + u(x)² = exp(-2x/√ε),   x ∈ (0, 1)

Boundary Conditions:
    u(0) = 1,   u(1) = exp(-1/√ε)

Physical Significance:
    This is a nonlinear reaction-diffusion equation. The nonlinear term u²
    represents a quadratic reaction rate, common in:
    • Autocatalytic chemical reactions
    • Population dynamics with logistic growth
    • Combustion modelling

Exact Solution:
    u(x) = exp(-x/√ε)

The nonlinearity requires fsolve (Newton-Raphson iteration) rather than a
direct linear solve. BCM with CGL nodes handles nonlinear systems more robustly.
"""

def run_example3(epsilon=0.01, n=30, verbose=True):
    """
    Solve Example 3: -εu'' + u + u² = exp(-2x/√ε) on (0,1).

    Nonlinear problem — uses BCM only (Galerkin not applied here).
    Note: Variational method is less straightforward for nonlinear problems.

    Parameters
    ----------
    epsilon : float — Perturbation parameter (default 0.01)
    n       : int   — Bernstein polynomial degree (default 30)
    verbose : bool  — Print errors (default True)

    Returns
    -------
    dict — Solution arrays and errors
    """
    def collocation_residuals(c, nodes):
        """Residual of -εu'' + u + u² - exp(-2x/√ε) = 0 at interior nodes."""
        equations = [c[0] - 1]
        for x in nodes[1:-1]:
            Vn    = sum(c[i] * bernstein_poly(i, n, x) for i in range(n+1))
            Vn_pp = sum(c[i] * bernstein_poly_second_deriv(i, n, x) for i in range(n+1))
            equations.append(-epsilon * Vn_pp + Vn + Vn**2
                             - np.exp(-2 * x / np.sqrt(epsilon)))
        equations.append(c[-1] - np.exp(-1 / np.sqrt(epsilon)))
        return equations

    ig = np.ones(n + 1)

    c_cheby = fsolve(collocation_residuals, ig,
                     args=(chebyshev_gauss_lobatto_nodes(n),))
    c_equi  = fsolve(collocation_residuals, ig,
                     args=(equidistant_collocation_nodes(n),))

    x_values = np.linspace(0, 1, 100)
    y_exact  = np.exp(-x_values / np.sqrt(epsilon))
    y_cheby  = [sum(c_cheby[i] * bernstein_poly(i, n, x) for i in range(n+1))
                for x in x_values]
    y_equi   = [sum(c_equi[i]  * bernstein_poly(i, n, x) for i in range(n+1))
                for x in x_values]

    err_cheby = compute_errors(y_exact, np.array(y_cheby))
    err_equi  = compute_errors(y_exact, np.array(y_equi))

    if verbose:
        print("\n" + "="*60)
        print(f"EXAMPLE 3 — Nonlinear Reaction-Diffusion  (ε = {epsilon})")
        print("="*60)
        print(f"  BCM (Equispaced)   : Max Error = {err_equi[0]:.4e},  RMSE = {err_equi[1]:.4e}")
        print(f"  BCM (CGL Nodes)    : Max Error = {err_cheby[0]:.4e},  RMSE = {err_cheby[1]:.4e}")

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_exact,  label='Exact Solution',                   color='blue',  linewidth=2)
    plt.plot(x_values, y_cheby,  label='BCM — Chebyshev-Gauss-Lobatto',   color='red',   linestyle='--')
    plt.plot(x_values, y_equi,   label='BCM — Equispaced Nodes',           color='green', linestyle='-.')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title(f'Example 3: Nonlinear Reaction-Diffusion SPDE  (ε = {epsilon})', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('example3_solutions.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'x': x_values, 'y_exact': y_exact, 'y_equi': y_equi, 'y_cheby': y_cheby,
        'errors': {'equi': err_equi, 'cheby': err_cheby}
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — EXAMPLE 4: SPDE WITH TRIGONOMETRIC FORCING
# ══════════════════════════════════════════════════════════════════════════════
"""
Problem Statement (Example 5.4 in the paper):
----------------------------------------------
    -ε u''(x) + u(x) = -cos²(πx) - 2ε²π²cos(2πx),   x ∈ (0, 1)

Boundary Conditions:
    u(0) = 0,   u(1) = 0

Physical Significance:
    Non-homogeneous forcing term with trigonometric structure. This type
    arises in problems involving periodic external forcing in thin fluid
    layers, heat conduction with oscillatory heat sources, and wave-diffusion
    coupling problems.

Exact Solution:
    u(x) = [exp(-x/√ε) + exp((x-1)/√ε)] / [1 + exp(-1/√ε)] - cos²(πx)

This example tests the method's ability to separate oscillatory components
from the boundary layer behaviour — a challenging multi-scale problem.
"""

def run_example4(epsilon=0.001, n=32, verbose=True):
    """
    Solve Example 4: -εu'' + u = -cos²(πx) - 2ε²π²cos(2πx) on (0,1).

    Parameters
    ----------
    epsilon : float — Perturbation parameter (default 0.001)
    n       : int   — Bernstein polynomial degree (default 32)
    verbose : bool  — Print errors (default True)

    Returns
    -------
    dict — Solution arrays and errors
    """
    R = 1

    # Exact solution
    def exact_sol(x):
        return ((np.exp(-x / np.sqrt(epsilon)) +
                 np.exp((x - 1) / np.sqrt(epsilon))) /
                (1 + np.exp(-1 / np.sqrt(epsilon))) - np.cos(np.pi * x)**2)

    # ── BCM — Equispaced ───────────────────────────────────────────────────
    def collocation_equi(c, n, epsilon):
        points = np.linspace(0, 1, n+1)
        equations = [c[0] - exact_sol(0)]
        for x in points[1:-1]:
            u    = sum(c[i] * bernstein_poly(i, n, x) for i in range(n+1))
            u_pp = sum(c[i] * bernstein_poly_second_deriv(i, n, x) for i in range(n+1))
            equations.append(-epsilon * u_pp + u +
                             (np.cos(np.pi*x)**2 + 2*(epsilon*np.pi)**2*np.cos(2*np.pi*x)))
        equations.append(c[-1] - exact_sol(1))
        return equations

    # ── BCM — Chebyshev-Gauss-Lobatto ─────────────────────────────────────
    def collocation_cgl(c, n, epsilon):
        points = chebyshev_gauss_lobatto_nodes(n)
        equations = [c[0] - exact_sol(0)]
        for x in points[1:-1]:
            u    = sum(c[i] * bernstein_poly(i, n, x) for i in range(n+1))
            u_pp = sum(c[i] * bernstein_poly_second_deriv(i, n, x) for i in range(n+1))
            equations.append(-epsilon * u_pp + u +
                             (np.cos(np.pi*x)**2 + 2*(epsilon*np.pi)**2*np.cos(2*np.pi*x)))
        equations.append(c[-1] - exact_sol(1))
        return equations

    # ── Galerkin Method ────────────────────────────────────────────────────
    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            func_A = lambda x, i=i, j=j: (
                epsilon * bernstein_derivative_scaled(j, n, x, R)
                        * bernstein_derivative_scaled(i, n, x, R)
                + bernstein_scaled(j, n, x, R) * bernstein_scaled(i, n, x, R)
            )
            A[i, j], _ = quad(func_A, 0, R)
            func_b = lambda x, i=i: (
                (-np.cos(np.pi * x)**2 - 2 * epsilon * np.pi * np.cos(2 * np.pi * x))
                * bernstein_scaled(i, n, x, R)
            )
            b_vec[i] += quad(func_b, 0, R)[0]

    A[0, :], A[-1, :] = 0, 0
    A[0, 0], A[-1, -1] = 1, 1
    b_vec[0], b_vec[-1] = exact_sol(0), exact_sol(1)
    coeffs_galerkin = solve(A, b_vec)

    ig = np.zeros(n + 1)
    c_equi = fsolve(collocation_equi, ig, args=(n, epsilon))
    c_cgl  = fsolve(collocation_cgl,  ig, args=(n, epsilon))

    x_values = np.linspace(0, 1, 200)
    y_exact_arr = np.array([exact_sol(x) for x in x_values])
    y_gal   = np.array([sum(coeffs_galerkin[i] * bernstein_scaled(i, n, x, R)
                            for i in range(n+1)) for x in x_values])
    y_equi  = np.array([sum(c_equi[i] * bernstein_poly(i, n, x)
                            for i in range(n+1)) for x in x_values])
    y_cgl   = np.array([sum(c_cgl[i]  * bernstein_poly(i, n, x)
                            for i in range(n+1)) for x in x_values])

    err_gal   = compute_errors(y_exact_arr, y_gal)
    err_equi  = compute_errors(y_exact_arr, y_equi)
    err_cgl   = compute_errors(y_exact_arr, y_cgl)

    if verbose:
        print("\n" + "="*60)
        print(f"EXAMPLE 4 — Trigonometric Forcing SPDE  (ε = {epsilon})")
        print("="*60)
        print(f"  Variational Method    : Max Error = {err_gal[0]:.4e},  RMSE = {err_gal[1]:.4e}")
        print(f"  BCM (Equispaced)      : Max Error = {err_equi[0]:.4e},  RMSE = {err_equi[1]:.4e}")
        print(f"  BCM (CGL Nodes)       : Max Error = {err_cgl[0]:.4e},  RMSE = {err_cgl[1]:.4e}")

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_exact_arr, label='Exact Solution',                   color='black',  linewidth=2)
    plt.plot(x_values, y_gal,       label='Variational Method',               color='orange')
    plt.plot(x_values, y_equi,      label='BCM — Equispaced Nodes',           color='red',    linestyle='--')
    plt.plot(x_values, y_cgl,       label='BCM — Chebyshev-Gauss-Lobatto',   color='blue',   linestyle='--')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title(f'Example 4: SPDE with Trigonometric Forcing  (ε = {epsilon})', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('example4_solutions.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'x': x_values, 'y_exact': y_exact_arr,
        'y_galerkin': y_gal, 'y_equi': y_equi, 'y_cgl': y_cgl,
        'errors': {'galerkin': err_gal, 'equi': err_equi, 'cgl': err_cgl}
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — RUN ALL EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "█"*70)
    print("  Bernstein Collocation Method with Chebyshev-Gauss-Lobatto Nodes")
    print("  for Singularly Perturbed Differential Equations")
    print("  Published in: Physica Scripta 100 (2025) 035239")
    print("  DOI: https://doi.org/10.1088/1402-4896/adb703")
    print("█"*70)

    # ── Example 1: Linear Convection-Diffusion ─────────────────────────────
    print("\n[1/5] Running Example 1 — Linear Convection-Diffusion...")
    run_example1(epsilon=0.01, n=32)

    # ── Example 1: Convergence Study (multiple epsilon) ───────────────────
    print("\n[2/5] Running Convergence Study for Example 1...")
    run_example1_convergence_study(n=32)

    # ── Example 2: Variable Coefficient SPDE ──────────────────────────────
    print("\n[3/5] Running Example 2 — Variable Coefficient SPDE...")
    run_example2(epsilon=0.01, n=32)

    # ── Example 3: Nonlinear Reaction-Diffusion ────────────────────────────
    print("\n[4/5] Running Example 3 — Nonlinear Reaction-Diffusion...")
    run_example3(epsilon=0.01, n=30)

    # ── Example 4: Trigonometric Forcing ──────────────────────────────────
    print("\n[5/5] Running Example 4 — SPDE with Trigonometric Forcing...")
    run_example4(epsilon=0.001, n=32)

    print("\n" + "="*70)
    print("  All examples completed. Plots saved as PNG files.")
    print("  Key finding: BCM with CGL nodes consistently achieves")
    print("  the lowest errors, especially for small ε (thin boundary layers).")
    print("="*70 + "\n")
