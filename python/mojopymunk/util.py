from .geometry import convex_hull, is_convex


def calc_area(points):
    from .geometry import area_for_poly

    return area_for_poly(points)


def convexise(points):
    return convex_hull(points)
