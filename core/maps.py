"""
Maps for reducing dimensionality and transforming shapes. The native
optimization is done on a n-dimensional sphere but the Nelder-Mead algorithm
optimizes on K^N and Bayesian optimization is done on [a1, b1]x...x[aN, bN].

@author = Joel
"""
import numpy as np

MAX_NORM_LIMIT = 5000  # np.linalg.norm is unstable for larger norm-orders.


def sphere_to_plane(x: np.ndarray, pole: int = 0) -> np.ndarray:
    """
    Stereographic projection from n-sphere to (n-1)-plane (K^(n-1)).

    The behavior for non-normalized vectors is not defined.

    @author = Joel

    :param x: Vector on n-sphere.
    :param pole: Index of the pole axis.
    :return: Mapped vector in (n-1)-plane (K^(n-1)).
    """
    x = np.array(x, copy=False)
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
    x = np.array(x, copy=False)
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
    if x[pole] <= 0:
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


def ball_to_cube_norm(x: np.ndarray, k: float = 1.):
    """
    Stretching from ball to cube by mapping spheres of radius r (in 2-norm) to
    a sphere with radius r^(2/p(r)) in p(r)-norm.

    p(r) = k/((1 - r) * (1 + 2(k-1)r))

    Note that p -> inf when r-> 1 (which is necessary to get a cube).

    @author = Joel

    :param x: Vector in ball.
    :param k: Se p(r) above.
    :return: Vector in cube.
    """
    r2 = np.linalg.norm(x, ord=2)
    if r2 == 0:
        return np.zeros(x.shape)
    else:
        p = np.true_divide(k, ((1 - r2) * (1 - 2*(k-1)*r2)))
        if p > MAX_NORM_LIMIT:
            p = np.inf
        rp = np.linalg.norm(x, ord=p)

        return r2*x/rp


def cube_to_ball_norm(x: np.ndarray, k: float = 1.):
    """
    Kind of difficult to describe...

    :param x: Vector in cube.
    :param k: Float defining the transformation.
    :return: Vector in ball.
    """
    r2 = np.linalg.norm(x, ord=2)
    r_inf = np.linalg.norm(x, ord=np.inf)
    if r_inf == 0:
        return np.zeros(x.shape)
    else:
        p = 2*np.true_divide(k, ((1 - r_inf) * (1 - 2*(k-1)*r_inf)))
        if p > MAX_NORM_LIMIT:
            p = np.inf
        rp = np.linalg.norm(x, ord=p)

        return rp*x/r2


