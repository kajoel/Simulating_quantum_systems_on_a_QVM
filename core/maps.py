"""
Maps for reducing dimensionality and transforming shapes. The native
optimization is done on a n-dimensional sphere but the Nelder-Mead algorithm
optimizes on K^N and Bayesian optimization is done on [a1, b1]x...x[aN, bN].

@author = Joel
"""
import numpy as np


def sphere_to_plane(x: np.ndarray, pole: int = 0) -> np.ndarray:
    """
    Stereographic projection from n-sphere to (n-1)-plane (K^(n-1)).

    The behavior for non-normalized vectors is not defined.

    @author = Joel

    :param x: Vector on n-sphere.
    :param pole: Index of the pole axis.
    :return: Mapped vector in (n-1)-plane (K^(n-1)).
    """
    # Pole vector:
    r = np.zeros(x.shape[0])
    r[pole] = 1

    x_r = np.vdot(r, x)  # projection of x on r
    y = (x - x_r*r)/(1 - x_r)  # stereographic projection

    return np.delete(y, pole)


def plane_to_sphere(x: np.ndarray, pole: int = 0) -> np.ndarray:
    """
    Inverse stereographic projection from (n-1)-plane (K^(n-1)) to n-sphere.

    @author = Joel

    :param x: Vector in (n-1)-plane (K^(n-1))
    :param pole: Index of the pole axis.
    :return: Mapped vector on n-sphere.
    """
    # Pole vector:
    r = np.zeros(x.shape[0]+1)
    r[pole] = 1

    return r + 2*(np.insert(x, pole, 0) - r)/(1 + np.vdot(x, x))


def sphere_to_ball(x: np.ndarray, pole: int = 0) -> np.ndarray:
    """
    Two stereographic projections, folding and scaling from n-sphere to
    (n-1)-ball. The southern hemisphere (with respect to the pole axis) is
    stereographically projected onto a ball with radius 1 and then scaled to
    radius 1/2. The northern hemisphere is projected stereographically with
    the opposite pole axis to a ball with radius 1 which is scaled down to
    radius 1/2 and lastly fold to an annulus with radii 1/2 and 1. This
    results in a mapping from the n-sphere to a (n-1)-ball of radius 1.

    The behavior for non-normalized vectors is not defined.

    @author = Joel

    :param x: Vector on n-sphere.
    :param pole: Index of the pole axis.
    :return: Mapped vector in (n-1)-ball.
    """
    if x[pole] >= 0:
        return sphere_to_plane(x, pole)/2
    else:
        y = -sphere_to_plane(-x, pole)/2
        y_norm = np.linalg.norm(y)
        return (1 - y_norm)*y/y_norm


def ball_to_sphere(x: np.ndarray, pole: int = 0) -> np.ndarray:
    """
    Inverse of sphere_to_ball mapping a (n-1)-ball to an n-sphere.

    @author = Joel

    :param x: Vector in (n-1)-ball.
    :param pole: Index of the pole axis.
    :return: Mapped vector on n-sphere.
    """
    x_norm = np.linalg.norm(x)
    if x_norm <= 0.5:
        return plane_to_sphere(2*x, pole)
    else:
        return -plane_to_sphere(-2*(1 - x_norm)*x/x_norm, pole)


def ball_to_cube_linear(x: np.ndarray) -> np.ndarray:
    """
    Linear stretching from ball to cube.

    :param x: Vector in ball.
    :return: Vector in cube.
    """
    x_norm = np.linalg.norm(x)
    if x_norm == 0:
        return np.zeros(x.shape)
    else:
        return np.linalg.norm(x)*x/np.max(np.abs(x))


def cube_to_ball_linear(x: np.ndarray) -> np.ndarray:
    """
    Linear stretching from cube to ball. Inverse of ball_to_cube_linear.

    :param x: Vector in cube.
    :return: Vector in ball.
    """
    x_norm = np.linalg.norm(x)
    if x_norm == 0:
        return np.zeros(x.shape)
    else:
        return np.max(np.abs(x)) * x / x_norm


def ball_to_cube_quadratic():
    pass


def cube_to_ball_quadratic():
    pass


def cube_to_ball_ellipsis():
    pass


def ball_to_cube_ellipsis():
    pass
