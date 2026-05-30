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

kx = ky   = 128
xrange    = yrange = (0.0, 128.0)
D         = [0.16, 0.08]
dt        = 1.0
tmax      = 6000.0
save_every = 10
pattern_name = "maze"

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

c=0.1
u0,v0 = rxd.purturbed_steady_state(c=c,initial_guess=[1.0,0.0])
u0 = np.clip(u0,0.0,1.0)
v0 = np.clip(v0,0.0,1.0)

U, V = rxd.solve((u0, v0))

ani_u, fps = animate_solution(U, dt_per_frame=dt * save_every, sim_speed=400, cmap="hsv")
ani_u.save(f"gifs/u_{pattern_name}.gif", writer="pillow", fps=fps, dpi=100)
ani_v, fps = animate_solution(V, dt_per_frame=dt * save_every,  sim_speed=400, cmap="hsv")
ani_v.save(f"gifs/v_{pattern_name}.gif", writer="pillow", fps=fps, dpi=100)