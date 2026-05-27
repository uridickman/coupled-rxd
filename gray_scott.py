from rxd import RxD_2d
import numpy as np
from animate import animate_solution

def f(u,v,a,b):
    return -u * v**2 + a * (1 - u)

def g(u,v,a,b):
    return  u * v**2 - (a + b) * v

params = {
    "flower": {"a": 0.055, "b": 0.062},
    "maze": {"a": 0.029, "b": 0.057},
    "mitosis": {"a": 0.03, "b": 0.062},
    "solitons": {"a": 0.03, "b": 0.06},
}

kx = ky = 256
dt = 0.5
xrange = yrange = (0.0,100)
tmax = 3000.0
save_every = 10
D = [0.2,0.1]
pattern_name = "mitosis"

rxd = RxD_2d(
    kx=kx,
    ky=ky,
    dt=dt,
    xrange=xrange,
    yrange=yrange,
    tmax=tmax,
    save_every=save_every,
    diffusion_coeffs=D,
    reaction_functions=[f, g],
    reaction_params=params[pattern_name],
)

u0 = np.ones((kx, ky))
v0 = np.zeros((kx, ky))

cx, cy = kx // 2, ky // 2
r = 20
rng = np.random.default_rng(42)

u0[cx-r:cx+r, cy-r:cy+r] = 0.5
v0[cx-r:cx+r, cy-r:cy+r] = 0.25

u0 += 0.01 * rng.standard_normal((kx, ky))
v0 += 0.01 * rng.standard_normal((kx, ky))
u0 = np.clip(u0, 0, 1)
v0 = np.clip(v0, 0, 1)

U, V = rxd.solve((u0, v0))

ani_u, _ = animate_solution(U, dt_per_frame=dt * save_every, cmap="hsv")
ani_u.save("u.gif", writer="pillow", fps=20, dpi=100)
ani_v, _ = animate_solution(V, dt_per_frame=dt * save_every, cmap="hsv")
ani_v.save("v.gif", writer="pillow", fps=20, dpi=100)