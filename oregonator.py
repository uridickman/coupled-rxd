from rxd import RxD_2d
import numpy as np
from animate import animate_solution

def f(u,v,w,a,b,eps1,eps2):
    return (a*v - u*v + u*(1-u)) / eps1

def g(u,v,w,a,b,eps1,eps2):
    return (-a*v - u*v + b*w) / eps2

def h(u,v,w,a,b,eps1,eps2):
    return u - w

params = {
    "bz": {"a": 7.62e-5, "b": 1, "eps1": 9.9e-3, "eps2": 1.98e-5}
}

kx = ky = 64
dt = 0.00001
xrange = yrange = (0.0, 0.8)
tmax = 0.5
save_every = 100
D = [1e-4, 0.0, 6e-4]
pattern_name = "bz"

rxd = RxD_2d(
    kx=kx,
    ky=ky,
    dt=dt,
    xrange=xrange,
    yrange=yrange,
    tmax=tmax,
    save_every=save_every,
    diffusion_coeffs=D,
    reaction_functions=[f, g, h],
    reaction_params=params[pattern_name],
)

# Approximate steady state: u=q, v=q, w=q
u0 = np.full((kx, ky), 0.002)
v0 = np.full((kx, ky), 0.002)
w0 = np.full((kx, ky), 0.002)

# Perturbed square in the center to seed a wave
cx, cy = kx // 2, ky // 2
r = 2
u0[cx-r:cx+r, cy-r:cy+r] = 0.8
v0[cx-r:cx+r, cy-r:cy+r] = 0.5
w0[cx-r:cx+r, cy-r:cy+r] = 0.1

# Small noise to break symmetry
rng = np.random.default_rng(42)
for arr in (u0, v0, w0):
    arr += 0.001 * rng.standard_normal((kx, ky))
    arr[:] = np.clip(arr, 0, None)

U, V, W = rxd.solve((u0, v0, w0))

for name, sol, cmap in [("u", U, "hsv"), ("v", V, "plasma"), ("w", W, "viridis")]:
    ani, _ = animate_solution(sol, dt_per_frame=dt * save_every, cmap=cmap)
    ani.save(f"{name}.gif", writer="pillow", fps=20, dpi=100)