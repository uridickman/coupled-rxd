from rxd import RxD_2d,radial_average
import numpy as np
import matplotlib.pyplot as plt

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
tmax      = 8000.0
save_every = 20
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

c=0.1
u0,v0 = rxd.purturbed_steady_state(c=c,initial_guess=[1.0,0.0])
u0 = np.clip(u0,0.0,1.0)
v0 = np.clip(v0,0.0,1.0)

U, V = rxd.solve((u0, v0))

# PLOT SOLUTION

fig,axs = plt.subplots(2,3,figsize=(11,7),constrained_layout=True,sharex=True,sharey=True)

m = int(tmax / save_every)
T = rxd.T
X,Y = rxd.X,rxd.Y

u_vmin,u_vmax = (np.min(U),np.max(U))
v_vmin,v_vmax = (np.min(V),np.max(V))

idx_times = [0,m//16,-1]

for idx,i in zip(idx_times,range(3)):
    ax_u = axs[0,i]
    ax_u.set_aspect("equal")
    pc_u = ax_u.pcolormesh(X,Y,U[:,:,idx],vmin=u_vmin,vmax=u_vmax,cmap="hsv")

    ax_v = axs[1,i]
    ax_v.set_aspect("equal")
    pc_v = ax_v.pcolormesh(X,Y,V[:,:,idx],vmin=v_vmin,vmax=v_vmax,cmap="hsv")

for ax in axs[:, 0]:
    ax.set_ylabel("Y")

for ax in axs[1, :]:
    ax.set_xlabel("X")

fig.colorbar(pc_u, ax=axs[0, -1])
fig.colorbar(pc_v, ax=axs[1, -1])

for i in range(3):
    axs[0,i].set_title(f"t = {T[idx_times[i]]:.0f}")

plt.show()

# PLOT FFT

z = U[:,:,-1]
z_hat = np.fft.fft2(z-np.mean(z))
z_hat_shifted = np.fft.fftshift(z_hat)

mag = np.abs(z_hat_shifted)

plt.figure(figsize=(8,6))
plt.imshow(np.log1p(mag), origin='lower', cmap='viridis')
plt.colorbar(label='log(1 + |FFT|)')
plt.xlabel('kx')
plt.ylabel('ky')
plt.grid(False)
plt.show()

# PLOT PEAK WAVELENGTH

import seaborn as sns
sns.set_theme("notebook",style="darkgrid")

power = np.abs(z_hat_shifted)**2

r,avg = radial_average(power)

from scipy.signal import find_peaks

peaks, _ = find_peaks(avg)

if len(peaks):
    peak = peaks[np.argmax(avg[peaks])]
print("Peak wavelength = ", 2*np.pi/kx*r[peak])

fig,ax = plt.subplots(1,1,constrained_layout=True)
ax.plot(2*np.pi/kx/r,avg,color="red")
ax.set_yscale("log")
ax.set_xlabel("Wavelength")
ax.set_ylabel("Azimuthal Power")
plt.show()
