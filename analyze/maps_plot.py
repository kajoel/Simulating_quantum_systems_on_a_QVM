"""
Plotting maps from maps.py

@author = Joel
"""
import itertools
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from core import maps


def ball_cube_2d(map, mapi) -> plt.Figure:
    """
    Plots map from ball to cube and its inverse.

    :param map: Map from ball to cube.
    :param mapi: Map from cube to ball
    :return: Figure.
    """
    # Init plot object:
    fig, ax = plt.subplots(2, 2)

    # ### Ball to cube ###
    num_grid = 9
    n = 8 * (num_grid - 1) + 1
    grid_idx = np.linspace(0, n - 1, num_grid, dtype=int)
    r_line = np.linspace(0, 1, n)
    t_line = np.linspace(0, 2*np.pi, n)
    r_ball, t_ball = np.meshgrid(r_line, t_line, indexing='ij')
    color = r_ball

    # Map
    x_cube_from_ball = np.zeros(r_ball.shape)
    y_cube_from_ball = np.zeros(r_ball.shape)
    for i, j in itertools.product(range(r_line.size), range(t_line.size)):
        x_cube_from_ball[i, j], y_cube_from_ball[i, j] = map(
            np.array([r_line[i]*np.cos(t_line[j]), r_line[i]*np.sin(t_line[j])])
        )

    # Plot
    style_r = ['--', '-', '-', '-']
    style_t = [':', '--', '-.', '-']
    ax[0, 0].pcolormesh(r_ball*np.cos(t_ball), r_ball*np.sin(t_ball), color,
                        cmap='viridis')
    ax[0, 1].pcolormesh(x_cube_from_ball, y_cube_from_ball, color,
                        cmap='viridis')
    for k, t_idx in enumerate(grid_idx):
        ax[0, 0].plot(r_ball[t_idx, :]*np.cos(t_ball[t_idx, :]),
                      r_ball[t_idx, :]*np.sin(t_ball[t_idx, :]),
                      color='r', linewidth=1, linestyle=style_t[divmult(k)])
        ax[0, 1].plot(x_cube_from_ball[t_idx, :], y_cube_from_ball[t_idx, :],
                      color='r', linewidth=1, linestyle=style_t[divmult(k)])
    for k, r_idx in enumerate(grid_idx):
        ax[0, 0].plot(r_ball[:, r_idx] * np.cos(t_ball[:, r_idx]),
                      r_ball[:, r_idx] * np.sin(t_ball[:, r_idx]),
                      color='r', linewidth=1, linestyle=style_r[divmult(k)])
        ax[0, 1].plot(x_cube_from_ball[:, r_idx], y_cube_from_ball[:, r_idx],
                      color='r', linewidth=1, linestyle=style_r[divmult(k)])
    ax[0, 0].axis('equal')
    ax[0, 0].axis(1.1*np.array([-1, 1, -1, 1]))
    ax[0, 1].axis('equal')
    ax[0, 1].axis(1.1 * np.array([-1, 1, -1, 1]))

    # ### Cube to ball ###
    num_grid = 9
    n = 8*(num_grid-1) + 1
    grid_idx = np.linspace(0, n - 1, num_grid, dtype=int)
    x_line = np.linspace(-1, 1, n)
    y_line = np.linspace(-1, 1, n)
    x_cube, y_cube = np.meshgrid(x_line, y_line, indexing='ij')
    color = -(x_cube**2 + y_cube**2)

    # Map
    x_ball_from_cube = np.zeros(x_cube.shape)
    y_ball_from_cube = np.zeros(y_cube.shape)
    for i, j in itertools.product(range(x_line.size), range(y_line.size)):
        x_ball_from_cube[i, j], y_ball_from_cube[i, j] = mapi(
            np.array([x_line[i], y_line[j]])
        )

    # Plot
    style_x = [':', '--', '-.', '-']
    style_y = [':', '--', '-.', '-']
    ax[1, 0].pcolormesh(x_cube, y_cube, color,
                        cmap='plasma')
    ax[1, 1].pcolormesh(x_ball_from_cube, y_ball_from_cube, color,
                        cmap='plasma')
    for k, x_idx in enumerate(grid_idx):
        ax[1, 0].plot(x_cube[x_idx, :], y_cube[x_idx, :],
                      color='g', linestyle=style_x[divmult(k)])
        ax[1, 1].plot(x_ball_from_cube[x_idx, :], y_ball_from_cube[x_idx, :],
                      color='g', linestyle=style_x[divmult(k)])
    for k, y_idx in enumerate(grid_idx):
        ax[1, 0].plot(x_cube[:, y_idx], y_cube[:, y_idx],
                      color='g', linestyle=style_y[divmult(k)])
        ax[1, 1].plot(x_ball_from_cube[:, y_idx], y_ball_from_cube[:, y_idx],
                      color='g', linestyle=style_y[divmult(k)])
    ax[1, 0].axis('equal')
    ax[1, 0].axis(1.1 * np.array([-1, 1, -1, 1]))
    ax[1, 1].axis('equal')
    ax[1, 1].axis(1.1 * np.array([-1, 1, -1, 1]))

    return fig


def sphere_ball(map, mapi) -> plt.Figure:
    """
    Plots map from (3d) sphere to (2d) ball and its inverse.

    :param map: Map from sphere to ball.
    :param mapi: Map from ball to sphere.
    :return: Figure.
    """
    # Init plot object:
    fig = plt.figure()
    ax = np.array([[fig.add_subplot(2, 2, 1, projection='3d'),
                    fig.add_subplot(2, 2, 2)],
                   [fig.add_subplot(2, 2, 3),
                    fig.add_subplot(2, 2, 4, projection='3d')]])

    # ### Sphere to ball ###
    num_grid = 9
    n = 8 * (num_grid - 1) + 1
    grid_idx = np.linspace(0, n - 1, num_grid, dtype=int)
    t_line = np.linspace(1e-2, np.pi, n)
    p_line = np.linspace(0, 2 * np.pi, n)
    t_sphere, p_sphere = np.meshgrid(t_line, p_line, indexing='ij')
    color = (np.pi-t_sphere)
    color = color/np.max(color)+0.2

    # Map
    x_ball_from_sphere = np.zeros(t_sphere.shape)
    y_ball_from_sphere = np.zeros(t_sphere.shape)
    for i, j in itertools.product(range(t_line.size), range(p_line.size)):
        x_ball_from_sphere[i, j], y_ball_from_sphere[i, j] = map(
            np.array(
                [np.sin(t_line[i]) * np.cos(p_line[j]),
                 np.sin(t_line[i]) * np.sin(p_line[j]),
                 np.cos(t_line[i])])
        )

    # Plot
    style_t = ['--', '-', '-', '-']
    style_p = [':', '--', '-.', '-']
    cmap = cm.ScalarMappable(cmap='viridis')
    cmap.set_array(color)
    cmap.autoscale()
    ax[0, 0].plot_surface(np.sin(t_sphere) * np.cos(p_sphere),
                          np.sin(t_sphere) * np.sin(p_sphere),
                          np.cos(t_sphere),
                          linewidth=0, rcount=1000, ccount=1000,
                          edgecolor='none', antialiased=False,
                          facecolors=cmap.to_rgba(color))
    ax[0, 1].pcolormesh(x_ball_from_sphere, y_ball_from_sphere, color,
                        cmap='viridis')
    for k, p_idx in enumerate(grid_idx):
        # TODO: seems to be a display bug when using both surf and plot
        # the lines don't show
        # ax[0, 0].plot(1.1 * np.sin(t_sphere[p_idx, :])
        #              * np.cos(p_sphere[p_idx, :]),
        #              1.1 * np.sin(t_sphere[p_idx, :])
        #              * np.sin(p_sphere[p_idx, :]),
        #              1.1 * np.cos(t_sphere[p_idx, :]),
        #              color='r', linewidth=1, linestyle=style_p[divmult(k)])
        ax[0, 1].plot(x_ball_from_sphere[p_idx, :],
                      y_ball_from_sphere[p_idx, :],
                      color='r', linewidth=1, linestyle=style_p[divmult(k)])
    for k, t_idx in enumerate(grid_idx):
        # TODO: seems to be a display bug when using both surf and plot
        # the lines don't show
        # ax[0, 0].plot(1.1 * np.sin(t_sphere[:, t_idx])
        #              * np.cos(p_sphere[:, t_idx]),
        #              1.1 * np.sin(t_sphere[:, t_idx])
        #              * np.sin(p_sphere[:, t_idx]),
        #              1.1 * np.cos(t_sphere[:, t_idx]),
        #              color='r', linewidth=1, linestyle=style_t[divmult(k)])
        ax[0, 1].plot(x_ball_from_sphere[:, t_idx],
                      y_ball_from_sphere[:, t_idx],
                      color='r', linewidth=1, linestyle=style_t[divmult(k)])
    ax[0, 0].axis('equal')  # TODO: seems to be a bug in this, see:
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit
    # -length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    ax[0, 0].set_xlim3d(-1.1, 1.1)
    ax[0, 0].set_ylim3d(-1.1, 1.1)
    ax[0, 0].set_zlim3d(-1.1, 1.1)
    # ax[0, 0].axis(1.1 * np.array([-1, 1, -1, 1]))
    ax[0, 1].axis('equal')
    ax[0, 1].axis(1.1 * np.array([-1, 1, -1, 1]))

    # ### Ball to sphere ###

    return fig


def divmult(n: int, m: int = 2) -> int:
    """
    Checks how many times n is divisible by m. Returns -1 if n=0

    @author = Joel

    :param n: Numerator
    :param m: Denominator
    :return: Multiplicity
    """
    if n < 0:
        raise ValueError('Only non-negative integers are supported for n.')
    elif n == 0:
        return -1
    q, r = divmod(n, m)
    count = 0
    while not r:
        count += 1
        q, r = divmod(q, m)
    return count


def _main_1():
    fig_1 = ball_cube_2d(maps.ball_to_cube_linear, maps.cube_to_ball_linear)
    fig_1.suptitle('Linear stretch')


if __name__ == '__main__':
    fig = sphere_ball(lambda x: maps.sphere_to_ball(x, pole=2),
                      lambda x: maps.ball_to_sphere(x, pole=2))
