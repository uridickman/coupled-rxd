import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_solution(solution, sim_speed=10, fps=15, dpi=100, dt_per_frame=1.0,
                     cmap='viridis', colorbar=True, title="Solution"):
    """
    sim_speed    : simulation time units per real second
    fps          : frames per real second (output quality)
    dt_per_frame : how much simulation time each saved frame represents
    """
    # How many frames play per real second to achieve sim_speed?
    # frames/real_sec = sim_speed [sim_units/real_sec] / dt_per_frame [sim_units/frame]
    effective_fps = sim_speed / dt_per_frame
    interval = 1000 / effective_fps  # ms between frames

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
        title_obj.set_text(f"{title} | t = {n * dt_per_frame:.2f}")
        return im, title_obj

    ani = animation.FuncAnimation(
        fig, update, frames=n_steps, interval=interval, blit=True
    )

    plt.tight_layout()
    return ani, effective_fps