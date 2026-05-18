from rxd_coupled import *
import numpy as np
import matplotlib.pyplot as plt
from animate import animate_solution

def f(u,v,a,b):
    return -u*v*v + a * (1-u)

def g(u,v,a,b):
    return u*v*v - (a + b) * v

# Mitosis
Du, Dv = 2e-5, 1e-5
a, b   = 0.03,0.063
kx = 512
ky = 512
params = {"a": a, "b": b}
xrange = (0,1)
yrange = (0,1)
tmax = 6000
dt = 1.0

# Coral
# Du, Dv = 2e-4, 1e-4
# a, b   = 0.055,0.062
# kx = 512
# ky = 512
# params = {"a": a, "b": b}
# xrange = (0,16)
# yrange = (0,16)
# tmax = 12500
# dt = 2.5

u0 = np.ones((kx, ky))
v0 = np.zeros((kx, ky))

x = np.linspace(*xrange, kx)
y = np.linspace(*yrange, ky)
X, Y = np.meshgrid(x, y,indexing="ij")

rng = np.random.default_rng(1)

delta = 0.1
for _ in range(15):
    cx = rng.uniform(xrange[0]+delta, xrange[1]-delta, size=1)
    cy = rng.uniform(yrange[0]+delta, yrange[1]-delta, size=1)
    sigma = rng.uniform(0.01, 0.03)

    blob = np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*sigma**2))
    v0 += 0.5 * blob

v0 = np.clip(v0, 0, 1)
u0 -= 0.5 * v0

if __name__ == "__main__":
    save_every = int(tmax // dt // 100)
    T,X,Y,U,V = solve_rxd(u0,v0,f,g,Du,Dv,dt,kx,ky,xrange,yrange,tmax,params,bc_type=BC_TYPE.NEUMANN,save_every=save_every)

    # fig,axs = plt.subplots(2,3,figsize=(13,9),constrained_layout=True)

    # vmin = U.min()
    # vmax = U.max()
    # cmap = "hsv"
    
    # axs[1,0].pcolormesh(X,Y,U[:,:,0],vmin=vmin,vmax=vmax,cmap=cmap)
    # axs[1,1].pcolormesh(X,Y,U[:,:,int(len(T)/4)],vmin=vmin,vmax=vmax,cmap=cmap)
    # axs[1,2].pcolormesh(X,Y,U[:,:,-1],vmin=vmin,vmax=vmax,cmap=cmap)

    # vmin = V.min()
    # vmax = V.max()
    # axs[0,0].pcolormesh(X,Y,V[:,:,0],vmin=vmin,vmax=vmax,cmap=cmap)
    # axs[0,1].pcolormesh(X,Y,V[:,:,int(len(T)/4)],vmin=vmin,vmax=vmax,cmap=cmap)
    # axs[0,2].pcolormesh(X,Y,V[:,:,-1],vmin=vmin,vmax=vmax,cmap=cmap)

    # for ax in axs.flatten():
    #     ax.set_aspect("equal")
    #     ax.set_xlabel("X",fontsize=14)
    #     ax.set_ylabel("Y",fontsize=14)

    # fig.savefig("fig.png")

    ani_u = animate_solution(U, interval=80, time_per_frame=save_every,cmap="hsv")
    ani_u.save("u.gif", writer="pillow", fps=15,dpi=100)
    ani_v = animate_solution(V, interval=80, time_per_frame=save_every, cmap="hsv")
    ani_v.save("v.gif", writer="pillow", fps=15,dpi=100)