import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_solution(solution, interval=100, time_per_frame=60, dpi=100,cmap='viridis', colorbar=True, title="Solution"):
    """
    Animate a 3D array solution[:, :, n] over the n (timestep) axis.

    Parameters
    ----------
    solution  : np.ndarray, shape (rows, cols, timesteps)
    interval  : int, delay between frames in milliseconds
    cmap      : str, matplotlib colormap name
    colorbar  : bool, whether to show a colorbar
    title     : str, base title (timestep is appended automatically)
    """
    vmin = solution.min()
    vmax = solution.max()
    n_steps = solution.shape[2]

    fig, ax = plt.subplots()
    im = ax.imshow(solution[:, :, 0], animated=True, cmap=cmap, vmin=vmin, vmax=vmax)

    if colorbar:
        fig.colorbar(im, ax=ax)

    title_obj = ax.set_title(f"{title} | t = 0")
    ax.axis("off")

    def update(n):
        im.set_data(solution[:, :, n])
        title_obj.set_text(f"{title} | t = {n*time_per_frame}")
        return im, title_obj

    ani = animation.FuncAnimation(
        fig, update, frames=n_steps, interval=interval, blit=True
    )

    plt.tight_layout()
    return ani  # keep a reference so it isn't garbage-collected