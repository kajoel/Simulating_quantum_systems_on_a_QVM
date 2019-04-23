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
    Plots map from ball to cube and its inverse. (Colorful)

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


def ball_cube_2d_2(map, mapi) -> plt.Figure:
    """
        Plots map from ball to cube and its inverse. (Not colorful)

        :param map: Map from ball to cube.
        :param mapi: Map from cube to ball
        :return: Figure.
        """
    # Init plot object:
    fig = plt.figure()
    ax = np.array([[fig.add_subplot(2, 2, 1),
                    fig.add_subplot(2, 2, 2)],
                   [fig.add_subplot(2, 2, 3),
                    fig.add_subplot(2, 2, 4)]])

    # ### Ball to cube ###
    grids = 17
    r_line = np.linspace(0, 1, grids)
    t_line = np.linspace(0, 2 * np.pi, grids)
    r_ball, t_ball = np.meshgrid(r_line, t_line, indexing='ij')

    # Map
    x_cube_from_ball = np.zeros(r_ball.shape)
    y_cube_from_ball = np.zeros(r_ball.shape)
    for i, j in itertools.product(range(r_line.size), range(t_line.size)):
        x_cube_from_ball[i, j], y_cube_from_ball[i, j] = map(
            np.array(
                [r_line[i] * np.cos(t_line[j]),
                 r_line[i] * np.sin(t_line[j])])
        )

    # Plot
    ax[0, 0].pcolormesh(r_ball * np.cos(t_ball), r_ball * np.sin(t_ball),
                        np.ones(r_ball.shape), cmap=cm.binary,
                        edgecolors='C0')
    ax[0, 1].pcolormesh(x_cube_from_ball, y_cube_from_ball,
                        np.ones(r_ball.shape), cmap=cm.binary,
                        edgecolors='C0')
    ax[0, 0].axis('equal')
    ax[0, 0].axis(1.1 * np.array([-1, 1, -1, 1]))
    ax[0, 1].axis('equal')
    ax[0, 1].axis(1.1 * np.array([-1, 1, -1, 1]))

    # ### Cube to ball ###
    grids = 17
    x_line = np.linspace(-1, 1, grids)
    y_line = np.linspace(-1, 1, grids)
    x_cube, y_cube = np.meshgrid(x_line, y_line, indexing='ij')

    # Map
    x_ball_from_cube = np.zeros(r_ball.shape)
    y_ball_from_cube = np.zeros(r_ball.shape)
    for i, j in itertools.product(range(x_line.size), range(y_line.size)):
        x_ball_from_cube[i, j], y_ball_from_cube[i, j] = mapi(
            np.array([x_cube[i, j], y_cube[i, j]])
        )

    # Plot
    ax[1, 0].pcolormesh(x_cube, y_cube,
                        np.ones(x_cube.shape), cmap=cm.binary,
                        edgecolors='C0')
    ax[1, 1].pcolormesh(x_ball_from_cube, y_ball_from_cube,
                        np.ones(r_ball.shape), cmap=cm.binary,
                        edgecolors='C0')
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
    grids = 16 + 1
    # num_grid = 9
    # m = 8
    # n = m * (num_grid - 1) + 1
    # grid_idx = np.linspace(0, n - 1, num_grid, dtype=int)
    # grid_2d = np.ix_(grid_idx, grid_idx)  # TODO: idea for grid on surface
    t_line = np.linspace(1e-2, np.pi, grids)  # TODO: change grids to n
    p_line = np.linspace(0, 2 * np.pi, grids)  # change grids to n
    t_sphere, p_sphere = np.meshgrid(t_line, p_line, indexing='ij')
    # color = (np.pi-t_sphere)  # Color map for surface
    # color = color/np.max(color)+0.2

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
    # style_t = ['--', '-', '-', '-']  # Gridstyles
    # style_p = [':', '--', '-.', '-']
    # cmap = cm.ScalarMappable(cmap='viridis') # Colomap
    # cmap.set_array(color)
    # cmap.autoscale()
    # Surface plot
    # ax[0, 0].plot_surface(np.sin(t_sphere) * np.cos(p_sphere),
    #                       np.sin(t_sphere) * np.sin(p_sphere),
    #                       np.cos(t_sphere),
    #                       linewidth=10, rcount=1000, ccount=1000,
    #                       edgecolor=edge_color,
    #                       antialiased=False,
    #                       facecolors=cmap.to_rgba(color))
    # Heatmap corresponding to surface plot:
    # ax[0, 1].pcolormesh(x_ball_from_sphere, y_ball_from_sphere, color,
    #                     cmap='viridis')
    # Grid plot:
    ax[0, 0].plot_wireframe(np.sin(t_sphere) * np.cos(p_sphere),
                            np.sin(t_sphere) * np.sin(p_sphere),
                            np.cos(t_sphere))
    ax[0, 1].pcolormesh(x_ball_from_sphere, y_ball_from_sphere,
                        np.ones(t_sphere.shape), cmap=cm.binary,
                        edgecolors='C0')

    ax[0, 0].set_aspect('equal')
    ax[0, 1].axis('equal')
    ax[0, 1].axis(1.1 * np.array([-1, 1, -1, 1]))

    # ### Ball to sphere ###
    grids = 16 + 1
    r_line = np.linspace(0, 1-1e-2, grids)    # TODO: change grids to n
    t_line = np.linspace(0, 2 * np.pi, grids)  # change grids to n
    r_ball, t_ball = np.meshgrid(r_line, t_line, indexing='ij')

    # Map
    x_sphere_from_ball = np.zeros(r_ball.shape)
    y_sphere_from_ball = np.zeros(r_ball.shape)
    z_sphere_from_ball = np.zeros(r_ball.shape)
    for i, j in itertools.product(range(r_line.size), range(t_line.size)):
        x_sphere_from_ball[i, j], \
            y_sphere_from_ball[i, j], \
            z_sphere_from_ball[i, j] = mapi(
                np.array([r_line[i] * np.cos(t_line[j]),
                          r_line[i] * np.sin(t_line[j])])
            )

    # Plot
    ax[1, 0].pcolormesh(r_ball * np.cos(t_ball), r_ball * np.sin(t_ball),
                        np.ones(r_ball.shape), cmap=cm.binary,
                        edgecolors='C0')
    ax[1, 1].plot_wireframe(x_sphere_from_ball,
                            y_sphere_from_ball,
                            z_sphere_from_ball)
    ax[1, 1].set_aspect('equal')
    ax[1, 0].axis('equal')
    ax[1, 0].axis(1.1 * np.array([-1, 1, -1, 1]))

    return fig


def sphere_cube(map, mapi) -> plt.Figure:
    """
    Plots map from (3d) sphere to (2d) cube and its inverse.

    :param map: Map from sphere to cube.
    :param mapi: Map from cube to sphere.
    :return: Figure.
    """
    # Init plot object:
    fig = plt.figure()
    ax = np.array([[fig.add_subplot(2, 2, 1, projection='3d'),
                    fig.add_subplot(2, 2, 2)],
                   [fig.add_subplot(2, 2, 3),
                    fig.add_subplot(2, 2, 4, projection='3d')]])

    # ### Sphere to cube ###
    grids = 16 + 1
    t_line = np.linspace(1e-2, np.pi, grids)
    p_line = np.linspace(0, 2 * np.pi, grids)
    t_sphere, p_sphere = np.meshgrid(t_line, p_line, indexing='ij')

    # Map
    x_cube_from_sphere = np.zeros(t_sphere.shape)
    y_cube_from_sphere = np.zeros(t_sphere.shape)
    for i, j in itertools.product(range(t_line.size), range(p_line.size)):
        x_cube_from_sphere[i, j], y_cube_from_sphere[i, j] = map(
            np.array(
                [np.sin(t_line[i]) * np.cos(p_line[j]),
                 np.sin(t_line[i]) * np.sin(p_line[j]),
                 np.cos(t_line[i])])
        )

    # Plot
    ax[0, 0].plot_wireframe(np.sin(t_sphere) * np.cos(p_sphere),
                            np.sin(t_sphere) * np.sin(p_sphere),
                            np.cos(t_sphere))
    ax[0, 1].pcolormesh(x_cube_from_sphere, y_cube_from_sphere,
                        np.ones(t_sphere.shape), cmap=cm.binary,
                        edgecolors='C0')

    ax[0, 0].set_aspect('equal')
    ax[0, 1].axis('equal')
    ax[0, 1].axis(1.1 * np.array([-1, 1, -1, 1]))

    # ### Cube to sphere ###
    grids = 16 + 1
    x_line = np.linspace(-1, 1, grids)  # +- 1e-2 if error
    y_line = np.linspace(-1, 1, grids)  # +- 1e-2 if error
    x_cube, y_cube = np.meshgrid(x_line, y_line, indexing='ij')

    # Map
    x_sphere_from_cube = np.zeros(x_cube.shape)
    y_sphere_from_cube = np.zeros(x_cube.shape)
    z_sphere_from_cube = np.zeros(x_cube.shape)
    for i, j in itertools.product(range(x_line.size), range(y_line.size)):
        x_sphere_from_cube[i, j], \
            y_sphere_from_cube[i, j], \
            z_sphere_from_cube[i, j] = mapi(
                np.array([x_line[i], y_line[j]])
            )

    # Plot
    ax[1, 0].pcolormesh(x_cube, y_cube,
                        np.ones(x_cube.shape), cmap=cm.binary,
                        edgecolors='C0')
    ax[1, 1].plot_wireframe(x_sphere_from_cube,
                            y_sphere_from_cube,
                            z_sphere_from_cube)
    ax[1, 1].set_aspect('equal')
    ax[1, 0].axis('equal')
    ax[1, 0].axis(1.1 * np.array([-1, 1, -1, 1]))

    return fig


def sphere_plane(map, mapi) -> plt.Figure:
    """
    Plots map from (3d) sphere to (2d) plane and its inverse.

    :param map: Map from sphere to plane.
    :param mapi: Map from plane to sphere.
    :return: Figure.
    """
    # Init plot object:
    fig = plt.figure()
    ax = np.array([[fig.add_subplot(2, 2, 1, projection='3d'),
                    fig.add_subplot(2, 2, 2)],
                   [fig.add_subplot(2, 2, 3),
                    fig.add_subplot(2, 2, 4, projection='3d')]])

    # ### Sphere to plane ###
    grids = 16 + 1
    t_line = np.linspace(1e-2, np.pi, grids)
    p_line = np.linspace(0, 2 * np.pi, grids)
    t_sphere, p_sphere = np.meshgrid(t_line, p_line, indexing='ij')

    # Map
    x_plane_from_sphere = np.zeros(t_sphere.shape)
    y_plane_from_sphere = np.zeros(t_sphere.shape)
    for i, j in itertools.product(range(t_line.size), range(p_line.size)):
        x_plane_from_sphere[i, j], y_plane_from_sphere[i, j] = map(
            np.array(
                [np.sin(t_line[i]) * np.cos(p_line[j]),
                 np.sin(t_line[i]) * np.sin(p_line[j]),
                 np.cos(t_line[i])])
        )

    # Plot
    ax[0, 0].plot_wireframe(np.sin(t_sphere) * np.cos(p_sphere),
                            np.sin(t_sphere) * np.sin(p_sphere),
                            np.cos(t_sphere))
    ax[0, 1].pcolormesh(x_plane_from_sphere, y_plane_from_sphere,
                        np.ones(t_sphere.shape), cmap=cm.binary,
                        edgecolors='C0')

    ax[0, 0].set_aspect('equal')
    ax[0, 1].axis('equal')
    ax[0, 1].axis(1.1 * np.array([-5, 5, -5, 5]))

    # ### Plane to sphere ###
    grids = 16 + 1
    r_line = np.linspace(0, 1, int(grids/2))
    r_line = np.concatenate((r_line, np.logspace(0, 1, int(grids/2))))
    t_line = np.linspace(0, 2 * np.pi, grids)
    r_plane, t_plane = np.meshgrid(r_line, t_line, indexing='ij')

    # Map
    x_sphere_from_plane = np.zeros(r_plane.shape)
    y_sphere_from_plane = np.zeros(r_plane.shape)
    z_sphere_from_plane = np.zeros(r_plane.shape)
    for i, j in itertools.product(range(r_line.size), range(t_line.size)):
        x_sphere_from_plane[i, j], \
            y_sphere_from_plane[i, j], \
            z_sphere_from_plane[i, j] = mapi(
            np.array([r_line[i] * np.cos(t_line[j]),
                      r_line[i] * np.sin(t_line[j])])
            )

    # Plot
    ax[1, 0].pcolormesh(r_plane * np.cos(t_plane), r_plane * np.sin(t_plane),
                        np.ones(r_plane.shape), cmap=cm.binary,
                        edgecolors='C0')
    ax[1, 1].plot_wireframe(x_sphere_from_plane,
                            y_sphere_from_plane,
                            z_sphere_from_plane)
    ax[1, 1].set_aspect('equal')
    ax[1, 0].axis('equal')
    ax[1, 0].axis(1.1 * np.array([-1, 1, -1, 1]))

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
    fig_1 = ball_cube_2d_2(maps.ball_to_cube_linear, maps.cube_to_ball_linear)
    fig_1.suptitle('2D ball to cube (linear)')

    fig_2 = sphere_ball(lambda x: maps.sphere_to_ball(x, pole=2),
                        lambda x: maps.ball_to_sphere(x, pole=2))
    fig_2.suptitle('3D sphere to 2D ball')

    fig_3 = sphere_cube(lambda x: maps.ball_to_cube_linear(
                                    maps.sphere_to_ball(x, pole=2)),
                        lambda x: maps.ball_to_sphere(
                                    maps.cube_to_ball_linear(x), pole=2))
    fig_3.suptitle('3D sphere to 2D cube')

    fig_4 = sphere_plane(lambda x: maps.sphere_to_plane(x, pole=2),
                         lambda x: maps.plane_to_sphere(x, pole=2))
    fig_4.suptitle('3D sphere to 2D plane')


if __name__ == '__main__':
    # _main_1()
    ball_cube_2d_2(lambda x: maps.ball_to_cube_norm(x, k=1),
                   lambda x: maps.cube_to_ball_norm(x, k=1))
    # plt.show()


