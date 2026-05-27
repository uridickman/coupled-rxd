from rxd import RxD_2d
import numpy as np
from animate import animate_solution

def f(u,v,a,b,tau,I):
    return u - u**3 / 3 - v + I

def g(u,v,a,b,tau,I):
    return  (u + a - b*v) / tau

params = {
    "waves": {"a": 0.7, "b": 0.8, "tau": 12.5, "I": 0.35}
}

kx = ky     = 256
dt          = 0.1
xrange      = yrange = (0.0, 4.0)   # ~800 grid cells per diffusion length
tmax        = 100.0                 # long enough to see many wave periods
save_every  = 10
D           = [1e-4, 0.0]
pattern_name = "waves"

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

u_rest, v_rest = -1.2, -0.623   # more precise rest state for I=0.6

u0 = np.full((kx, ky), u_rest)
v0 = np.full((kx, ky), v_rest)

# thin strip pacemaker
u0[:, :3] = 1.5
v0[:, :3] = -1.2

rng = np.random.default_rng(0)
u0 += 0.01 * rng.standard_normal((kx, ky))
v0 += 0.01 * rng.standard_normal((kx, ky))

U, V = rxd.solve((u0, v0))

ani_u, _ = animate_solution(U, dt_per_frame=dt * save_every, cmap="hsv")
ani_u.save("u.gif", writer="pillow", fps=20, dpi=100)
ani_v, _ = animate_solution(V, dt_per_frame=dt * save_every, cmap="hsv")
ani_v.save("v.gif", writer="pillow", fps=20, dpi=100)