from __future__ import annotations

import math
import numbers
from typing import NamedTuple


class Vec2d(NamedTuple):
    x: float
    y: float

    def __repr__(self) -> str:
        return f"Vec2d({self.x}, {self.y})"

    def __add__(self, other) -> "Vec2d":
        return Vec2d(self.x + other[0], self.y + other[1])

    __radd__ = __add__

    def __sub__(self, other) -> "Vec2d":
        return Vec2d(self.x - other[0], self.y - other[1])

    def __rsub__(self, other) -> "Vec2d":
        return Vec2d(other[0] - self.x, other[1] - self.y)

    def __mul__(self, scalar: float) -> "Vec2d":
        assert isinstance(scalar, numbers.Real)
        return Vec2d(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vec2d":
        return Vec2d(self.x / scalar, self.y / scalar)

    def __floordiv__(self, scalar: float) -> "Vec2d":
        return Vec2d(self.x // scalar, self.y // scalar)

    def __neg__(self) -> "Vec2d":
        return Vec2d(-self.x, -self.y)

    def __pos__(self) -> "Vec2d":
        return self

    def __abs__(self) -> float:
        return self.length

    def __bool__(self) -> bool:
        return bool(self.x != 0 or self.y != 0)

    @property
    def length(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    @property
    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    @property
    def angle_degrees(self) -> float:
        return math.degrees(self.angle)

    @property
    def int_tuple(self) -> tuple[int, int]:
        return int(self.x), int(self.y)

    @property
    def polar_tuple(self) -> tuple[float, float]:
        return self.length, self.angle

    def dot(self, other) -> float:
        return self.x * other[0] + self.y * other[1]

    def cross(self, other) -> float:
        return self.x * other[1] - self.y * other[0]

    def get_distance(self, other) -> float:
        return math.hypot(self.x - other[0], self.y - other[1])

    def get_distance_squared(self, other) -> float:
        return (self.x - other[0]) ** 2 + (self.y - other[1]) ** 2

    get_dist_sqrd = get_distance_squared

    def get_length_sqrd(self) -> float:
        return self.length_squared

    def normalized(self) -> "Vec2d":
        length = self.length
        return self / length if length else Vec2d(0.0, 0.0)

    def normalized_and_length(self) -> tuple["Vec2d", float]:
        length = self.length
        return (self / length if length else Vec2d(0.0, 0.0), length)

    def perpendicular(self) -> "Vec2d":
        return Vec2d(-self.y, self.x)

    def perpendicular_normal(self) -> "Vec2d":
        return self.perpendicular().normalized()

    def projection(self, other) -> "Vec2d":
        vector = Vec2d(*other)
        denominator = vector.length_squared
        return vector * (self.dot(vector) / denominator) if denominator else Vec2d.zero()

    def rotated(self, angle_radians: float) -> "Vec2d":
        cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
        return Vec2d(self.x * cosine - self.y * sine, self.x * sine + self.y * cosine)

    def rotated_degrees(self, angle_degrees: float) -> "Vec2d":
        return self.rotated(math.radians(angle_degrees))

    def get_angle_between(self, other) -> float:
        return math.atan2(self.cross(other), self.dot(other))

    def get_angle_degrees_between(self, other) -> float:
        return math.degrees(self.get_angle_between(other))

    def interpolate_to(self, other, range: float) -> "Vec2d":
        return self + (Vec2d(*other) - self) * range

    def convert_to_basis(self, x_vector, y_vector) -> "Vec2d":
        return Vec2d(
            self.dot(x_vector) / Vec2d(*x_vector).length_squared,
            self.dot(y_vector) / Vec2d(*y_vector).length_squared,
        )

    def cpvrotate(self, other) -> "Vec2d":
        return Vec2d(
            self.x * other[0] - self.y * other[1],
            self.x * other[1] + self.y * other[0],
        )

    def cpvunrotate(self, other) -> "Vec2d":
        return Vec2d(
            self.x * other[0] + self.y * other[1],
            self.y * other[0] - self.x * other[1],
        )

    def scale_to_length(self, length: float) -> "Vec2d":
        return self.normalized() * length

    @staticmethod
    def from_polar(length: float, angle: float) -> "Vec2d":
        return Vec2d(length * math.cos(angle), length * math.sin(angle))

    @staticmethod
    def zero() -> "Vec2d":
        return Vec2d(0.0, 0.0)

    @staticmethod
    def unit() -> "Vec2d":
        return Vec2d(1.0, 0.0)

    @staticmethod
    def ones() -> "Vec2d":
        return Vec2d(1.0, 1.0)
