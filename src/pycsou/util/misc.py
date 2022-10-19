import pycsou.util as pycu
import pycsou.util.ptype as pyct


def peaks(x: pyct.NDArray, y: pyct.NDArray) -> pyct.NDArray:
    r"""
    Matlab 2D peaks function.
    Peaks is a function of two variables, obtained by translating and scaling Gaussian distributions (see `Matlab's peaks function <https://www.mathworks.com/help/matlab/ref/peaks.html>`).
    This function is useful for testing purposes.
    Parameters
    ----------
    x: NDArray
        X coordinates.
    y: NDArray
        Y coordinates.
    Returns
    -------
    NDArray
        Values of the 2D function ``peaks`` at the points specified by the entries of ``x`` and ``y``.
    Examples
    --------
    .. plot::
       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.util.misc import peaks
       x = np.linspace(-3,3, 1000)
       xx, yy = np.meshgrid(x,x)
       z = peaks(xx, yy)
       plt.figure()
       plt.imshow(z)
    """
    xp = pycu.get_array_module(x)
    z = (
        3 * ((1 - x) ** 2) * xp.exp(-(x**2) - (y + 1) ** 2)
        - 10 * (x / 5 - x**3 - y**5) * xp.exp(-(x**2) - y**2)
        - (1 / 3) * xp.exp(-((x + 1) ** 2) - y**2)
    )
    return z
