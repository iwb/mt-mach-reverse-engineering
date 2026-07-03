"""Plotting helpers for the reconstructed workpiece geometry."""

from shapely import MultiPolygon, Polygon, unary_union
from shapely.plotting import plot_polygon


def plot_geometry_inner_shape(union, color, ax, facecolor: str = None):
    """Plot the enclosed (unmachined) region of a reconstructed geometry.

    For a ``MultiPolygon`` the free space inside the outer boundary is drawn; for a
    single ``Polygon`` its interior holes are drawn.

    :param union: Reconstructed machined area (shapely ``Polygon``/``MultiPolygon``).
    :param color: Edge colour for the drawn region.
    :param ax: Matplotlib axes to draw on.
    :param facecolor: Optional fill colour.
    """
    if isinstance(union, MultiPolygon):
        outer_boundary = union.geoms[0].exterior.coords
        outer = Polygon(outer_boundary)
        all_poly = unary_union(union)
        free_space = outer.difference(all_poly)
        if facecolor is None:
            plot_polygon(free_space, ax, add_points=False, edgecolor=color, zorder=2)
        else:
            plot_polygon(free_space, ax, add_points=False, edgecolor=color, facecolor=facecolor, zorder=2)
    else:
        for interior in union.interiors:
            hole = Polygon(interior.coords)
            if facecolor is None:
                plot_polygon(hole, ax, add_points=False, edgecolor=color, zorder=2)
            else:
                plot_polygon(hole, ax, add_points=False, edgecolor=color, facecolor=facecolor, zorder=2)
