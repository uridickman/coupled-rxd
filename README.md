# 2D Reaction-Diffusion Solver

*Author*: Uri Dickman

### Description

The code in this repository solves the system of $N$ reaction-diffusion equations
$$\frac{\partial \boldsymbol{u}}{\partial t}=\boldsymbol{D}\nabla^2\boldsymbol{u}+\boldsymbol{R}(\boldsymbol{u})$$

The code uses the following method:
- Peaceman-Rachford Alternating Direction Implicit operator splitting on the Laplacian operator $\nabla^2$
- Crank-Nicolson method on the reaction term $\boldsymbol{R}(\boldsymbol{u})$
- Predictor-corrector (Heun) method to approximate the implicit term in the Crank-Nicolson reaction term
- Equipped both with no-flux and $0$-Dirichlet boundary conditions

Order and accuracy:
- Second-order centered differencing in the Laplacian
- Second-order Neumann ghost points for no-flux BCs

The order of the method is therefore 
$$\mathcal{O}(\Delta t^2 + \Delta x^2 + \Delta y^2)$$

### Usage

This project was implemented using Python version 3.14.

Install essential packages:
```bash
pip install -r requirements.txt
```

Run the examples:
```bash
python gray_scott.py
python fitzhugh_nagumo.py
```