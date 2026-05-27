import numpy as np
from scipy.sparse.linalg import splu,SuperLU
from scipy.sparse import csc_array,csc_matrix,eye,lil_matrix,block_diag,save_npz
from enum import Enum
from tqdm import trange
from dataclasses import dataclass,field
from pathlib import Path
from typing import Tuple, List, Callable

class BC_TYPE(Enum):
    DIRICHLET   = 1
    NEUMANN     = 2


@dataclass
class Operators:
    LU_A_LXX:   SuperLU
    LU_A_LYY:   SuperLU
    B_LXX:      csc_matrix
    B_LYY:      csc_matrix
    LAPLACIAN:  csc_matrix


class RxD_2d:
    def __init__(
    self,
    kx                  : int                           = 32,
    ky                  : int                           = 32,
    dt                  : float                         = 1.0,
    xrange              : tuple[float, float]           = (0.0, 1.0),
    yrange              : tuple[float, float]           = (0.0, 1.0),
    tmax                : float                         = 1.0,
    save_every          : int                           = 10,
    diffusion_coeffs    : list[float]                   = field(default_factory=list),
    reaction_functions  : list[Callable[..., Any]]      = field(default_factory=lambda: [lambda u, v: 0]),
    reaction_params     : dict[str, float]              = field(default_factory=dict),
    bc_type             : BC_TYPE                       = BC_TYPE.NEUMANN,

    ):
        self.kx,self.ky = kx,ky
        self.xmin,self.xmax = xrange
        self.ymin,self.ymax = yrange
        self.dx = (self.xmax - self.xmin) / kx
        self.dy = (self.ymax - self.ymin) / ky
        self.tmax = tmax
        self.dt = dt
        self.save_every = save_every
        self.bc_type = bc_type

        self.diffusion_coeffs = diffusion_coeffs
        self.reaction_functions = reaction_functions
        self.reaction_params = reaction_params
        self.num_equations = len(diffusion_coeffs)

        assert len(reaction_functions) == self.num_equations

        self.T = np.arange(0,tmax,step=dt)
        self.num_steps = len(self.T)
        self.num_saved = self.num_steps // save_every + 1

        x = np.linspace(*xrange,num=kx)
        y = np.linspace(*yrange,num=ky)
        self.X,self.Y = np.meshgrid(x,y,indexing="ij")

        self.operators = tuple(self._make_operators(D, self.dt) for D in self.diffusion_coeffs)

        self.tmp_reactions              = tuple(np.zeros(kx*ky) for _ in range(self.num_equations))
        self.tmp_sol_predictions        = tuple(np.zeros(kx*ky) for _ in range(self.num_equations))
        self.tmp_reaction_predictions   = tuple(np.zeros(kx*ky) for _ in range(self.num_equations))

        self.sol = tuple(np.zeros((kx*ky,self.num_saved)) for _ in self.diffusion_coeffs)


    def _make_operators(self, D, dt):
        Lxx, Lyy = self._compute_difference_operators(D)
        return Operators(*self._setup_linear_system(Lxx, Lyy, dt), Lxx + Lyy)


    def _compute_difference_operators(self,D):
        kx,ky = self.kx, self.ky

        shape = (kx*ky,kx*ky)
        
        cx = D / self.dx / self.dx
        cy = D / self.dy / self.dy
        Lxx = lil_matrix(shape,dtype=float)
        Lyy = lil_matrix(shape,dtype=float)
        
        for i in range(kx):
            for j in range(ky):
                row = i * ky + j
                Lxx[row,row] = -2 * cx
                if i > 0:
                    Lxx[row, (i - 1) * ky + j] = cx
                if i < kx - 1:
                    Lxx[row, (i + 1) * ky + j] = cx

        for i in range(kx):
            for j in range(ky):
                row = i * ky + j
                Lyy[row,row] = -2 * cy
                if j > 0:
                    Lyy[row, i * ky + (j - 1)] = cy
                if j < ky - 1:
                    Lyy[row, i * ky + (j + 1)] = cy

        if self.bc_type == BC_TYPE.DIRICHLET:

            for j in range(ky):

                left = j
                right = (kx-1)*ky + j

                Lxx[left, left] = 0
                Lxx[right, right] = 0

            for i in range(kx):

                bottom = i*ky
                top = i*ky + (ky-1)

                Lyy[bottom, bottom] = 0
                Lyy[top, top] = 0

        if self.bc_type == BC_TYPE.NEUMANN:

            for j in range(ky):

                left = j
                right = (kx-1)*ky + j

                Lxx[left, left] = -2*cx
                Lxx[left, ky + j] = 2*cx

                Lxx[right, right] = -2*cx
                Lxx[right, (kx-2)*ky + j] = 2*cx

            for i in range(kx):

                bottom = i*ky
                top = i*ky + (ky-1)

                Lyy[bottom, bottom] = -2*cy
                Lyy[bottom, i*ky + 1] = 2*cy

                Lyy[top, top] = -2*cy
                Lyy[top, i*ky + (ky-2)] = 2*cy
                
        return Lxx.tocsc(),Lyy.tocsc()


    def _setup_linear_system(self,Lxx,Lyy,dt):
        I = eye(Lxx.shape[0],format="csc")

        lu_A_Lxx = splu(I - dt / 2 * Lxx)
        lu_A_Lyy = splu(I - dt / 2 * Lyy)

        B_Lxx = I + dt / 2 * Lxx
        B_Lyy = I + dt / 2 * Lyy

        return lu_A_Lxx,lu_A_Lyy,B_Lxx,B_Lyy


    def _reshape_solution(self,sols):
        out = []
        for sol in sols:
            out.append(sol.reshape(self.kx,self.ky,self.num_saved))
        return out


    def enforce_dirichlet_bcs(self,u):
        kx = self.kx
        u[:kx]      = 0  # bottom row (i=0)
        u[-kx:]     = 0  # top row    (i=ky-1)
        u[::kx]     = 0  # left col   (j=0)
        u[kx-1::kx] = 0  # right col  (j=kx-1)


    def recompute_reaction(self, solutions, params):
        for i, f in enumerate(self.reaction_functions):
            self.tmp_reactions[i][:] = f(*solutions, **params)


    def recompute_prediction(self, solutions, reactions, dt):
        for i, (u, r, ops) in enumerate(zip(solutions, reactions, self.operators)):
            self.tmp_sol_predictions[i][:] = u + dt * (ops.LAPLACIAN @ u + r)


    def adi_pec_advance(self, sol_nm1, dirichlet=False):
        """
        ADI-PEC(EC) advance: Predict-Evaluate-Correct scheme using
        Alternating Direction Implicit splitting.
        
        Step 1: Evaluate reactions at current solution
        Step 2: Predict with Euler step
        Step 3: ADI half-step in X (using predicted reactions)
        Step 4: ADI half-step in Y (using corrected reactions)
        """

        # --- Evaluate reactions at u^n ---
        self.recompute_reaction(sol_nm1,self.reaction_params)

        # --- Euler prediction: u* = u^n + dt * L(u^n) + R(u^n) ---
        self.recompute_prediction(sol_nm1, self.tmp_reactions, self.dt)

        if dirichlet:
            for pred in self.tmp_sol_predictions:
                self.enforce_dirichlet_bcs(pred)

        # --- Evaluate reactions at predicted solution ---
        for i, f in enumerate(self.reaction_functions):
            self.tmp_reaction_predictions[i][:] = f(*self.tmp_sol_predictions, **self.reaction_params)

        # --- First ADI half-sweep: implicit in X, explicit in Y ---
        # u^{n+1/2} = (I - dt/2 Lxx)^{-1} [ (I + dt/2 Lyy) u^n + dt/4 * (R* + R^n) ]
        solutions_half = tuple(
            ops.LU_A_LXX.solve(
                ops.B_LYY @ u_nm1
                + self.dt / 4 * (r_pred + r_curr)
            )
            for u_nm1, r_curr, r_pred, ops in zip(
                sol_nm1,
                self.tmp_reactions,
                self.tmp_reaction_predictions,
                self.operators,
            )
        )

        if dirichlet:
            for u_half in solutions_half:
                self.enforce_dirichlet_bcs(u_half)

        # --- Re-evaluate and re-predict at half-step solution ---
        self.recompute_reaction(solutions_half,self.reaction_params)
        self.recompute_prediction(solutions_half, self.tmp_reactions, self.dt / 2)
        if dirichlet:
            for pred in self.tmp_sol_predictions:
                self.enforce_dirichlet_bcs(pred)

        # --- Re-evaluate reactions at new prediction ---
        for i, f in enumerate(self.reaction_functions):
            self.tmp_reaction_predictions[i][:] = f(*self.tmp_sol_predictions, **self.reaction_params)

        # --- Second ADI half-sweep: implicit in Y, explicit in X ---
        # u^{n+1} = (I - dt/2 Lyy)^{-1} [ (I + dt/2 Lxx) u^{n+1/2} + dt/4 * (R* + R^n) ]
        solutions_next = tuple(
            ops.LU_A_LYY.solve(
                ops.B_LXX @ u_half
                + self.dt / 4 * (r_pred + r_curr)
            )
            for u_half, r_curr, r_pred, ops in zip(
                solutions_half,
                self.tmp_reactions,
                self.tmp_reaction_predictions,
                self.operators,
            )
        )

        if dirichlet:
            for u_next in solutions_next:
                self.enforce_dirichlet_bcs(u_next)

        return solutions_next


    def solve(self,initial_conditions):
        
        for ic,sol in zip(initial_conditions,self.sol):
            ic_reshaped = ic.reshape(self.kx*self.ky)
            sol[:,0] = ic_reshaped

        sol_nm1 = tuple(sol[:,0] for sol in self.sol)
        N = self.num_steps
        idx = 1
        for n in trange(1,N+1):
            sol_nm1 = self.adi_pec_advance(sol_nm1, dirichlet=False)
            if n % self.save_every == 0:
                for i, sol in enumerate(self.sol):
                    sol[:, idx] = sol_nm1[i]
                idx += 1

        self.sol = self._reshape_solution(self.sol)
        return self.sol