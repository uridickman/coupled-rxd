import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import csc_array,eye,lil_matrix,block_diag
from enum import Enum
from tqdm import trange
from dataclasses import dataclass
from typing import Any

class BC_TYPE(Enum):
    DIRICHLET   = 1
    NEUMANN     = 2

@dataclass
class Operators:
    LU_A_LXX: Any
    LU_A_LYY: Any
    B_LXX: Any
    B_LYY: Any
    LAPLACIAN: Any

def compute_difference_operators(D,kx,ky,xmin,xmax,ymin,ymax,bc_type=BC_TYPE.NEUMANN):
    shape = (kx*ky,kx*ky)
    dx = (xmax - xmin) / kx
    dy = (ymax - ymin) / ky
    cx = D / dx / dx
    cy = D / dy / dy
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

    if bc_type == BC_TYPE.DIRICHLET:

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

    if bc_type == BC_TYPE.NEUMANN:

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


def enforce_dirichlet(u, kx, ky):
    u[:kx]      = 0  # bottom row (i=0)
    u[-kx:]     = 0  # top row    (i=ky-1)
    u[::kx]     = 0  # left col   (j=0)
    u[kx-1::kx] = 0  # right col  (j=kx-1)

def setup_linear_system(Lxx,Lyy,dt):
    I = eye(Lxx.shape[0],format="csc")

    lu_A_Lxx = splu(I - dt / 2 * Lxx)
    lu_A_Lyy = splu(I - dt / 2 * Lyy)

    B_Lxx = I + dt / 2 * Lxx
    B_Lyy = I + dt / 2 * Lyy

    return lu_A_Lxx,lu_A_Lyy,B_Lxx,B_Lyy


def recompute_reaction(u,v,f,g,reaction_vecs,params):
    u_reaction, v_reaction = reaction_vecs
    u_reaction[:] = f(u,v,**params)
    v_reaction[:] = g(u,v,**params)


def recompute_prediction(prediction_u,prediction_v,u,v,reaction_u,reaction_v,dt,params,u_ops,v_ops):
    prediction_u[:],prediction_v[:] = euler_step(u,v,reaction_u,reaction_v,dt,params,u_ops,v_ops)


def euler_step(u_nm1,v_nm1,reaction_u,reaction_v,dt,params,u_ops,v_ops):
    laplace_u = u_ops.LAPLACIAN @ u_nm1
    laplace_v = v_ops.LAPLACIAN @ v_nm1
    return (
        u_nm1 + dt * (laplace_u + reaction_u),
        v_nm1 + dt * (laplace_v + reaction_v)
    )


def adi_pec_advance(u_nm1,v_nm1,f,g,kx,ky,dt,pred_vecs,reaction_vecs,pred_reaction_vecs,params,u_ops,v_ops,dirichlet=False):
    
    recompute_reaction(u_nm1,v_nm1,f,g,reaction_vecs,params)
    recompute_prediction(*pred_vecs,u_nm1,v_nm1,*reaction_vecs,dt,params,u_ops,v_ops)
    
    if dirichlet:
        enforce_dirichlet(pred_vecs[0],kx,ky)
        enforce_dirichlet(pred_vecs[1],kx,ky)
    
    recompute_reaction(*pred_vecs,f,g,pred_reaction_vecs,params)
    u_rhs = u_ops.B_LYY @ u_nm1 + dt * (pred_reaction_vecs[0] + reaction_vecs[0]) / 4
    v_rhs = v_ops.B_LYY @ v_nm1 + dt * (pred_reaction_vecs[1] + reaction_vecs[1]) / 4
    u_next = u_ops.LU_A_LXX.solve(u_rhs)
    v_next = v_ops.LU_A_LXX.solve(v_rhs)

    if dirichlet:
        enforce_dirichlet(u_next,kx,ky)
        enforce_dirichlet(v_next,kx,ky)

    recompute_reaction(u_next,v_next,f,g,reaction_vecs,params)
    recompute_prediction(*pred_vecs,u_next,v_next,*reaction_vecs,dt,params,u_ops,v_ops)
    
    if dirichlet:
        enforce_dirichlet(pred_vecs[0],kx,ky)
        enforce_dirichlet(pred_vecs[1],kx,ky)

    recompute_reaction(*pred_vecs,f,g,pred_reaction_vecs,params)
    u_rhs = u_ops.B_LXX @ u_next + dt * (pred_reaction_vecs[0] + reaction_vecs[0]) / 4
    v_rhs = v_ops.B_LXX @ v_next + dt * (pred_reaction_vecs[1] + reaction_vecs[1]) / 4
    u_next = u_ops.LU_A_LYY.solve(u_rhs)
    v_next = v_ops.LU_A_LYY.solve(v_rhs)

    if dirichlet:
        enforce_dirichlet(u_next,kx,ky)
        enforce_dirichlet(v_next,kx,ky)

    return u_next,v_next


def solve_rxd(u0,v0,f,g,Du,Dv,dt,kx,ky,xrange,yrange,tmax,params,bc_type=BC_TYPE.NEUMANN,save_every=10):

    T = np.arange(0,tmax,step=dt*save_every)
    x = np.linspace(*xrange,num=kx)
    y = np.linspace(*yrange,num=ky)
    X,Y = np.meshgrid(x,y,indexing="ij")

    N = int(tmax / dt)
    num_saved = N // save_every + 1

    U0 = np.ravel(u0)
    V0 = np.ravel(v0)

    tmp_pred_vecs = (np.zeros(kx*ky),np.zeros(kx*ky))
    tmp_reaction_vecs = (np.zeros(kx*ky),np.zeros(kx*ky))
    tmp_pred_reaction_vecs = (np.zeros(kx*ky),np.zeros(kx*ky))

    Lxx,Lyy = compute_difference_operators(Du,kx,ky,*xrange,*yrange,bc_type=bc_type)
    U_OPERATORS = Operators(
        *setup_linear_system(Lxx,Lyy,dt),
        Lxx + Lyy
    )

    Lxx,Lyy = compute_difference_operators(Dv,kx,ky,*xrange,*yrange,bc_type=bc_type)
    V_OPERATORS = Operators(
        *setup_linear_system(Lxx,Lyy,dt),
        Lxx + Lyy
    )

    sol_U = np.zeros((kx*ky,num_saved))
    sol_V = np.zeros((kx*ky,num_saved))

    U_nm1 = U0
    sol_U[:,0] = U_nm1

    V_nm1 = V0
    sol_V[:,0] = V_nm1

    idx = 1
    for n in trange(1,N+1):
        U_nm1,V_nm1 = adi_pec_advance(U_nm1,V_nm1,f,g,kx,ky,dt,tmp_pred_vecs,tmp_reaction_vecs,tmp_pred_reaction_vecs,params,U_OPERATORS,V_OPERATORS,dirichlet=(bc_type == BC_TYPE.DIRICHLET))
        if n % save_every == 0:
            sol_U[:,idx] = U_nm1
            sol_V[:,idx] = V_nm1
            idx += 1

    U = sol_U.reshape(kx,ky,num_saved)
    V = sol_V.reshape(kx,ky,num_saved)
    return T,X,Y,U,V