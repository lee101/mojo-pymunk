from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._lib import addr, f64, lib
from .vec2d import Vec2d


class BB(NamedTuple):
    left: float = 0
    bottom: float = 0
    right: float = 0
    top: float = 0

    @staticmethod
    def newForCircle(p, r: float) -> "BB":
        return BB(p[0] - r, p[1] - r, p[0] + r, p[1] + r)

    def intersects(self, other: "BB") -> bool:
        return (
            self.left <= other.right
            and other.left <= self.right
            and self.bottom <= other.top
            and other.bottom <= self.top
        )

    def contains(self, other: "BB") -> bool:
        return (
            self.left <= other.left
            and self.bottom <= other.bottom
            and self.right >= other.right
            and self.top >= other.top
        )

    def contains_vect(self, v) -> bool:
        return self.left <= v[0] <= self.right and self.bottom <= v[1] <= self.top

    def merge(self, other: "BB") -> "BB":
        return BB(
            min(self.left, other.left),
            min(self.bottom, other.bottom),
            max(self.right, other.right),
            max(self.top, other.top),
        )

    def expand(self, v) -> "BB":
        return BB(
            min(self.left, v[0]),
            min(self.bottom, v[1]),
            max(self.right, v[0]),
            max(self.top, v[1]),
        )

    def center(self) -> Vec2d:
        return Vec2d((self.left + self.right) / 2, (self.bottom + self.top) / 2)

    def area(self) -> float:
        return (self.right - self.left) * (self.top - self.bottom)

    def merged_area(self, other: "BB") -> float:
        return self.merge(other).area()

    def clamp_vect(self, v) -> Vec2d:
        return Vec2d(
            min(max(v[0], self.left), self.right),
            min(max(v[1], self.bottom), self.top),
        )

    def segment_query(self, a, b) -> float:
        tmin, tmax = 0.0, 1.0
        for origin, delta, lower, upper in (
            (a[0], b[0] - a[0], self.left, self.right),
            (a[1], b[1] - a[1], self.bottom, self.top),
        ):
            if delta == 0:
                if origin < lower or origin > upper:
                    return float("inf")
                continue
            near, far = (lower - origin) / delta, (upper - origin) / delta
            if near > far:
                near, far = far, near
            tmin, tmax = max(tmin, near), min(tmax, far)
            if tmin > tmax:
                return float("inf")
        return tmin

    def intersects_segment(self, a, b) -> bool:
        return self.segment_query(a, b) != float("inf")


class Transform(NamedTuple):
    a: float = 1
    b: float = 0
    c: float = 0
    d: float = 1
    tx: float = 0
    ty: float = 0

    def __matmul__(self, other):
        if len(other) == 2:
            x, y = other
            return Vec2d(self.a * x + self.c * y + self.tx, self.b * x + self.d * y + self.ty)
        a, b, c, d, tx, ty = other
        return Transform(
            self.a * a + self.c * b,
            self.b * a + self.d * b,
            self.a * c + self.c * d,
            self.b * c + self.d * d,
            self.a * tx + self.c * ty + self.tx,
            self.b * tx + self.d * ty + self.ty,
        )

    @staticmethod
    def identity() -> "Transform":
        return Transform()

    @staticmethod
    def translation(x: float, y: float) -> "Transform":
        return Transform(tx=x, ty=y)

    @staticmethod
    def scaling(s: float) -> "Transform":
        return Transform(a=s, d=s)

    @staticmethod
    def rotation(t: float) -> "Transform":
        import math

        c, s = math.cos(t), math.sin(t)
        return Transform(c, s, -s, c)

    def translated(self, x: float, y: float) -> "Transform":
        return self @ Transform.translation(x, y)

    def scaled(self, s: float) -> "Transform":
        return self @ Transform.scaling(s)

    def rotated(self, t: float) -> "Transform":
        return self @ Transform.rotation(t)

    def inverted(self) -> "Transform":
        determinant = self.a * self.d - self.b * self.c
        if determinant == 0:
            raise ValueError("Singular transform cannot be inverted")
        return Transform(
            self.d / determinant,
            -self.b / determinant,
            -self.c / determinant,
            self.a / determinant,
            (self.c * self.ty - self.d * self.tx) / determinant,
            (self.b * self.tx - self.a * self.ty) / determinant,
        )


def _vertices(vertices, *, minimum: int = 2) -> np.ndarray:
    if np.asarray(vertices).size == 0:
        raise ValueError(f"expected at least {minimum} vertices")
    array = f64(vertices, ndim=2)
    if array.shape[1] != 2:
        raise ValueError("vertices must be pairs")
    if len(array) < minimum:
        raise ValueError(f"expected at least {minimum} vertices")
    return array


def moment_for_circle(mass, inner_radius, outer_radius, offset=(0, 0)) -> float:
    return lib().mp_moment_for_circle(mass, inner_radius, outer_radius, offset[0], offset[1])


def area_for_circle(inner_radius, outer_radius) -> float:
    return lib().mp_area_for_circle(inner_radius, outer_radius)


def moment_for_segment(mass, a, b, radius) -> float:
    return lib().mp_moment_for_segment(mass, a[0], a[1], b[0], b[1], radius)


def area_for_segment(a, b, radius) -> float:
    return lib().mp_area_for_segment(a[0], a[1], b[0], b[1], radius)


def moment_for_box(mass, size) -> float:
    return lib().mp_moment_for_box(mass, size[0], size[1])


def moment_for_poly(mass, vertices, offset=(0, 0), radius=0) -> float:
    array = _vertices(vertices)
    return lib().mp_moment_for_poly(mass, addr(array), len(array), offset[0], offset[1], radius)


def area_for_poly(vertices, radius=0) -> float:
    array = _vertices(vertices)
    return lib().mp_area_for_poly(addr(array), len(array), radius)


def poly_centroid(vertices) -> Vec2d:
    array = _vertices(vertices, minimum=3)
    twice_area = np.sum(
        array[:, 0] * np.roll(array[:, 1], -1)
        - array[:, 1] * np.roll(array[:, 0], -1)
    )
    if twice_area == 0:
        raise ValueError("polygon centroid is undefined for zero area")
    result = np.empty(2, dtype=np.float64)
    lib().mp_centroid_for_poly(addr(array), len(array), addr(result))
    return Vec2d(*result)


def moments_for_circles(mass, inner_radius, outer_radius, offset) -> np.ndarray:
    masses = f64(mass, ndim=1)
    inner = f64(inner_radius, ndim=1)
    outer = f64(outer_radius, ndim=1)
    offsets = f64(offset, ndim=2)
    if offsets.shape[1:] != (2,) or not (
        masses.shape == inner.shape == outer.shape == (len(offsets),)
    ):
        raise ValueError("all inputs must have the same length")
    if not len(masses):
        return np.empty(0, dtype=np.float64)
    result = np.empty_like(masses)
    lib().mp_batch_moment_for_circle(
        addr(masses), addr(inner), addr(outer), addr(offsets), addr(result), len(masses)
    )
    return result


def is_convex(points) -> bool:
    vertices = [Vec2d(*point) for point in points]
    if len(vertices) < 3:
        return False
    sign = 0
    for i, point in enumerate(vertices):
        cross = (vertices[(i + 1) % len(vertices)] - point).cross(
            vertices[(i + 2) % len(vertices)] - vertices[(i + 1) % len(vertices)]
        )
        if cross:
            new_sign = 1 if cross > 0 else -1
            if sign and sign != new_sign:
                return False
            sign = new_sign
    return bool(sign)


def convex_hull(points, tolerance=0.0):
    unique = sorted(set(map(tuple, points)))
    if len(unique) <= 1:
        return [Vec2d(*point) for point in unique]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    return [Vec2d(*point) for point in lower[:-1] + upper[:-1]]
