# Overview

This repository contains the materials and code related to my Master's thesis titled **"CONSTRUCTING EQUILIBRIUM OF PLASMA MAGNETIC CONFINEMENT THROUGH THE APPLICATION OF PHYSICS-INFORMED NEURAL NETWORKS."** The thesis investigates the dynamics of plasma instabilities and aims to construct equilibrium solutions using the Magnetohydrodynamics (MHD) framework. To address these complex plasma physics problems, this research introduces the application of Physics-Informed Neural Networks (PINNs) as a computationally efficient and accurate method for solving the MHD equilibrium equation.

My thesis can be accessed [here](https://drive.google.com/file/d/1VPo87p9ThL6-uUqPIZTl7FdW2i9rr1uB/view?usp=sharing).

The repository is organized into several folders, each containing relevant documents, scripts, and resources utilized throughout the research.

## References

### 1. MHD Equilibria
This section includes key references and literature on the computation and theoretical understanding of Magnetohydrodynamic (MHD) equilibria, particularly in tokamak configurations. It covers foundational texts, computational methods, and studies on ideal MHD equilibria.

- Frances Bauer, Octavio Betancourt, Paul Garabedian (Auth.) - A Computational Method in Plasma Physics. Springer-Verlag Berlin Heidelberg (1978)
- Axisymmetric Ideal MHD Tokamak Equilibria
- Computation of MHD Equilibrium of Tokamak Plasma
- Computing Ideal Magnetohydrodynamic Equilibria
- Steepest-Descent Moment Method for 3D MHD Equilibria

### 2. Past Thesis
This section contains previous thesis works that are relevant to the current research. These theses focus on nonlinear MHD simulations, plasma instabilities, and other related areas in stellarators and tokamaks.

- Equilibrium and Initial Value Problem Simulation Studies of Nonlinear MHD in Stellarator
- Free-Boundary Simulations of MHD Plasma Instabilities in Tokamaks
- Non-Linear Magnetohydrodynamic Simulations of Edge Localised Modes

### 3. Solve GS Equation
This section comprises key references on solving the Grad-Shafranov (GS) equation, a fundamental equation in plasma physics. The documents cover a variety of solution techniques, including spectral elements, finite difference methods, and conformal mapping.

- A Fast, High-Order Solver for GS Equation
- Computation of Fixed Boundary Tokamak Equilibria Using Method Based on Approximate Particular Solution
- One Size Fits All Analytic Solutions GS Equation
- A Review on the Solution of Grad-Shafranov Equation in the Cylindrical Coordinates Based on the Chebyshev Collocation Technique
- Analytical Solution of the Grad-Shafranov Equation in an Elliptical Prolate Geometry
- Analytical Solutions to the Grad-Shafranov Equation for Tokamak Equilibrium with Dissimilar Source Functions
- Fixed Boundary Grad-Shafranov Solver Using Finite Difference Method in Nonhomogeneous Meshgrid
- Solving the Grad-Shafranov Equation with Spectral Elements (2014)
- Solving the Grad-Shafranov Equation Using Spectral Elements for Tokamak Equilibrium with Toroidal Rotation
- Stephen Jardin - Computational Methods in Plasma Physics (Chapman & Hall CRC Computational Science). CRC Press (2010)

### Construction of Model PINN

This section provides code and methods to solve a 1-dimensional differential equation with perturbation elements and Dirichlet boundary conditions using Physics-Informed Neural Networks (PINN). The equation solved is:

\[
y'' + y + \varepsilon y^3 = 0,
\]
with boundary conditions \(y(0) = 0\) and \(y(\frac{\pi}{2}) = B\).

Two scripts are provided:

- **`regular_perturbation.py`**: Contains the implementation of the PINN to solve the differential equation. It defines the loss function, configures the network architecture with 5 hidden layers and 20 neurons each, and trains the model using the Adam optimizer.

- **`search_activation_function.py`**: Evaluates the performance of different activation functions in solving the differential equation and helps determine the best activation function for this specific problem.

### Solov'ev Equilibrium

The `PINN_Solver_GSeq_parametric_boundary.ipynb` notebook implements a Physics-Informed Neural Network (PINN) to approximate the solution of the Solov'ev equilibrium equation:

\[
\Delta^*\hat{\psi} = A_1 r^2 + A_2,
\]
with the boundary condition \(\hat{\psi} = 0\) on a D-shaped boundary.

Key components of the notebook include:

- **Network Architecture**: The model is constructed with 8 hidden layers, each containing 20 neurons, and utilizes the SiLU activation function for nonlinear transformations.
  
- **Boundary Condition**: The boundary condition is implemented parametrically for a D-shaped boundary, characterized by the parameters \(\varepsilon = 0.78\), \(\alpha = 0.3576\), and \(\kappa = 2\).

- **Optimization**: The PINN is trained using the Adam optimizer, which minimizes the mean squared error (MSE) of the differential residual and boundary conditions.

- **Results Visualization**: The notebook visualizes the PINN solution for the given values \(A_1 = 1\) and \(A_2 = 0\). It compares the PINN solution with the analytical solution, showing the evolution of error, the differential residual, and the difference between the PINN and analytical solutions.

This implementation demonstrates the capability of the PINN to effectively approximate the Solov'ev equilibrium equation in a controlled magnetic confinement setting.

### Eigenvalue Problem in Ordinary Differential Equations

The notebooks `evp_standard_neg.ipynb` and `evp_standard_rayleigh_quotient.ipynb` are implementations related to solving eigenvalue problems in ordinary differential equations.

- **`evp_standard_neg.ipynb`**: This notebook applies an iterative method to solve the eigenvalue problem \(y'' = \lambda y\) with boundary conditions \(y(0) = y(L) = 0\), starting with a negative initial guess for \(\lambda_0\). The goal is to approximate the smallest eigenvalue \(\lambda_n\) and the corresponding eigenfunction \(y_n\), using a special procedure to estimate the supremum \(\|y\|_\infty\).

- **`evp_standard_rayleigh_quotient.ipynb`**: This notebook explores the same eigenvalue problem, but the iterative method is enhanced by incorporating the Rayleigh quotient, which helps in refining the eigenvalue estimates more efficiently.

Both notebooks illustrate the effectiveness of these methods in converging to the smallest eigenvalue and its corresponding eigenfunction, even when starting from different initial guesses for \(\lambda_0\).

### Linear Eigenvalue Problem of the Grad-Shafranov Equation

The notebooks provided tackle the linear eigenvalue problem associated with the Grad-Shafranov equation, aiming to find pairs of eigenvalues and eigenfunctions with the minimum norm using an iterative method combined with PINN. Specifically:

- **`PINN_Solver_GSeq_parametric_boundary_linear_rhs_evp.ipynb`**: This notebook addresses **Case 1** where \(A_1 = 0\) and \(A_2 = 1\).** It iteratively solves the Grad-Shafranov equation with an initial eigenvalue guess of \(\lambda_0 = -10\), using the PINN solution from the Solov'ev equilibrium as the initial guess for \(\hat{\psi}_0\).

- **`PINN_Solver_GSeq_parametric_boundary_linear_rhs_evp copy.ipynb`**: This notebook is for **Case 2** where \(A_1 = 0.5\) and \(A_2 = 0.5\).** Similar to Case 1, it uses the same iterative method and PINN approach to solve the eigenvalue problem.

- **`PINN_Solver_GSeq_parametric_boundary_linear_rhs_evp copy2.ipynb`**: This notebook handles **Case 3** where \(A_1 = 1\) and \(A_2 = 0\).** The iterative method and PINN are again employed to find the corresponding eigenvalues and eigenfunctions.

These notebooks explore different parameter values to illustrate how the iterative method and PINN can be used to solve the linear eigenvalue problem of the Grad-Shafranov equation under various conditions.