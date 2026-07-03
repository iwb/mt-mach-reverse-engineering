from matplotlib.colors import to_rgba
import numpy as np
from scipy.stats import norm


def bake_alpha(color, alpha, background=(1, 1, 1)):
    r, g, b, _ = to_rgba(color)
    br, bg, bb = background
    return (
        alpha * r + (1 - alpha) * br,
        alpha * g + (1 - alpha) * bg,
        alpha * b + (1 - alpha) * bb,
    )


def gaussian_stats(values_by_noise):
    """Return sorted noise levels, means, stds fitted from samples per noise level."""
    noises = np.array(sorted(values_by_noise.keys()))
    means, stds = [], []
    for n in noises:
        vals = np.asarray(values_by_noise[n], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            mu, sigma = np.nan, np.nan
        else:
            mu, sigma = norm.fit(vals)  # MLE Gaussian fit
        means.append(mu)
        stds.append(sigma)
    return noises, np.array(means), np.array(stds)


def plot_with_boundary_lines(ax, noises, mu, sigma, color, marker=None):
    ax.plot(noises, mu, color=color, zorder=4, marker=marker)
    ax.plot(noises, mu + sigma, color=color, lw=0.5, ls="--", zorder=3)
    ax.plot(noises, mu - sigma, color=color, lw=0.5, ls="--", zorder=3)
    ax.fill_between(noises, mu - sigma, mu + sigma, color=color, alpha=0.15, linewidth=0, zorder=2)
