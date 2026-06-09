from rxd import RxD_2d
import numpy as np
import matplotlib.pyplot as plt

def f_fhn(u, v, a, b, epsilon):
    return u - (u**3) / 3.0 - v

def g_fhn(u, v, a, b, epsilon):
    return epsilon * (u + a - b * v)

params = {
    "spiral": {"a": 0.5, "b": 0.5, "epsilon": 0.08},
    "wave": {"a": 0.7, "b": 0.8, "epsilon": 0.04},
    "turing": {"a": 2.0, "b": 0.0, "epsilon": 1.0}
}

kx = ky = 128
xrange = yrange = (0.0, 50.0)
D = [0.01, 0.25]

dt = 0.1
tmax = 2000.0
save_every = 50
pattern_name = "turing"

rxd = RxD_2d(
    kx=kx,
    ky=ky,
    dt=dt,
    xrange=xrange,
    yrange=yrange,
    tmax=tmax,
    save_every=save_every,
    diffusion_coeffs=D,
    reaction_functions=[f_fhn, g_fhn],
    reaction_params=params["spiral"],
)

# # Center wave at (0.5, 0.5)
# # and use the steady-state values for u/v rather than arbitrary ±1.2
# u0 = np.zeros((kx, ky))
# v0 = np.zeros((kx, ky))

# X, Y = np.meshgrid(np.linspace(0, 1, kx), np.linspace(0, 1, ky), indexing="ij")

# # Activator: left half excited, right half at rest (use FHN nullcline values)
# u0[X < 0.5] = 2.0   # near the right branch of the u-nullcline
# u0[X >= 0.5] = -1.0  # near the left branch

# # Inhibitor: split horizontally — use values on the v-nullcline
# # v-nullcline: v = (u + a) / b  →  at u=2.0: v=(2.5)/0.5=5 (clamp to 1.5)
# #                                    at u=-1.0: v=(−0.5)/0.5=−1
# v0[Y > 0.5] = 1.5   # high inhibitor in upper half (blocks wavefront → creates break)
# v0[Y <= 0.5] = -0.5  # low inhibitor in lower half (allows propagation)

# 1. Find the true resting state (where du/dt = 0 and dv/dt = 0)
# For a=0.7, b=0.8, the resting state is roughly u = -1.19, v = -0.62
# u_rest = -1.19
# v_rest = -0.62

# u0 = np.full((kx, ky), u_rest)
# v0 = np.full((kx, ky), v_rest)

# # 2. Stimulate the left edge to trigger the wave
# u0[0:10, :] = 1.5  # High activator concentration along the left border

u0 = np.random.normal(0.0, 0.1, (kx, ky))
v0 = np.random.normal(0.0, 0.1, (kx, ky))

fig,axs = plt.subplots(2,3,figsize=(11,7),constrained_layout=True,sharex=True,sharey=True)

m = int(tmax / save_every)
T = rxd.T
X,Y = rxd.X,rxd.Y

u_vmin,u_vmax = (np.min(U),np.max(U))
v_vmin,v_vmax = (np.min(V),np.max(V))

idx_times = [0,m//4,-1]

for idx,i in zip(idx_times,range(3)):
    ax_u = axs[0,i]
    ax_u.set_aspect("equal")
    pc_u = ax_u.pcolormesh(X,Y,U[:,:,idx],vmin=u_vmin,vmax=u_vmax,cmap="RdBu_r")

    ax_v = axs[1,i]
    ax_v.set_aspect("equal")
    pc_v = ax_v.pcolormesh(X,Y,V[:,:,idx],vmin=v_vmin,vmax=v_vmax,cmap="RdBu_r")

for ax in axs[:, 0]:
    ax.set_ylabel("Y")

for ax in axs[1, :]:
    ax.set_xlabel("X")

fig.colorbar(pc_u, ax=axs[0, -1])
fig.colorbar(pc_v, ax=axs[1, -1])

for i in range(3):
    axs[0,i].set_title(f"t = {T[idx_times[i]]:.0f}")

plt.show()