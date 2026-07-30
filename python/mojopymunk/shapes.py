from __future__ import annotations

import math
from typing import NamedTuple

from .geometry import BB, Transform, area_for_circle, area_for_poly, area_for_segment, moment_for_circle, moment_for_poly, moment_for_segment, poly_centroid
from .vec2d import Vec2d


class ShapeFilter(NamedTuple):
    group: int = 0
    categories: int = 0xFFFFFFFF
    mask: int = 0xFFFFFFFF

    @staticmethod
    def ALL_CATEGORIES() -> int:
        return 0xFFFFFFFF

    @staticmethod
    def ALL_MASKS() -> int:
        return 0xFFFFFFFF

    def rejects_collision(self, other: "ShapeFilter") -> bool:
        return (
            (self.group != 0 and self.group == other.group)
            or not (self.categories & other.mask)
            or not (other.categories & self.mask)
        )


class PointQueryInfo(NamedTuple):
    shape: "Shape"
    point: Vec2d
    distance: float
    gradient: Vec2d


class SegmentQueryInfo(NamedTuple):
    shape: "Shape"
    point: Vec2d
    normal: Vec2d
    alpha: float


class ContactPoint(NamedTuple):
    point_a: Vec2d
    point_b: Vec2d
    distance: float


class ContactPointSet(NamedTuple):
    normal: Vec2d
    points: tuple[ContactPoint, ...]


class ShapeQueryInfo(NamedTuple):
    shape: "Shape"
    contact_point_set: ContactPointSet


class Shape:
    def __init__(self, body) -> None:
        self._body = body
        if body is not None:
            body._shapes.add(self)
        self._space = None
        self._friction = 0.0
        self._elasticity = 0.0
        self._sensor = False
        self._collision_type = 0
        self._filter = ShapeFilter()
        self._surface_velocity = Vec2d.zero()
        self._mass = 0.0
        self._density = 0.0

    def _mark_query_dirty(self) -> None:
        if self._space is not None:
            self._space._query_dirty = True
            self._space._shape_dirty = True

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, value) -> None:
        if self._body is not None:
            self._body._shapes.discard(self)
        self._body = value
        if value is not None:
            value._shapes.add(self)
        self._mark_query_dirty()

    @property
    def space(self):
        return self._space

    @property
    def friction(self) -> float:
        return self._friction

    @friction.setter
    def friction(self, value: float) -> None:
        self._friction = float(value)
        self._mark_query_dirty()

    @property
    def elasticity(self) -> float:
        return self._elasticity

    @elasticity.setter
    def elasticity(self, value: float) -> None:
        self._elasticity = float(value)
        self._mark_query_dirty()

    @property
    def sensor(self) -> bool:
        return self._sensor

    @sensor.setter
    def sensor(self, value: bool) -> None:
        self._sensor = bool(value)
        self._mark_query_dirty()

    @property
    def collision_type(self) -> int:
        return self._collision_type

    @collision_type.setter
    def collision_type(self, value: int) -> None:
        self._collision_type = int(value)

    @property
    def filter(self) -> ShapeFilter:
        return self._filter

    @filter.setter
    def filter(self, value: ShapeFilter) -> None:
        self._filter = ShapeFilter(*value)
        self._mark_query_dirty()

    @property
    def surface_velocity(self) -> Vec2d:
        return self._surface_velocity

    @surface_velocity.setter
    def surface_velocity(self, value) -> None:
        self._surface_velocity = Vec2d(*value)

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        self._mass = float(value)
        self._density = self._mass / self.area if self.area else 0.0

    @property
    def density(self) -> float:
        return self._density

    @density.setter
    def density(self, value: float) -> None:
        self._density = float(value)
        self._mass = self.area * self._density

    @property
    def bb(self) -> BB:
        return self.cache_bb()

    def cache_bb(self) -> BB:
        raise NotImplementedError

    def update(self, transform: Transform) -> BB:
        return self._bb_for_transform(transform)

    def _transform(self) -> Transform:
        if self.body is None:
            return Transform.identity()
        rotation = Transform.rotation(self.body.angle)
        return Transform.translation(*self.body.position) @ rotation

    def copy(self):
        import copy

        duplicate = copy.copy(self)
        duplicate._space = None
        return duplicate


class Circle(Shape):
    def __init__(self, body, radius: float, offset=(0, 0)) -> None:
        super().__init__(body)
        self._radius = float(radius)
        self._offset = Vec2d(*offset)

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def offset(self) -> Vec2d:
        return self._offset

    @property
    def area(self) -> float:
        return area_for_circle(0, self.radius)

    @property
    def moment(self) -> float:
        return moment_for_circle(self.mass, 0, self.radius, self.offset)

    @property
    def center_of_gravity(self) -> Vec2d:
        return self.offset

    def _center(self, transform=None) -> Vec2d:
        return (transform or self._transform()) @ self.offset

    def _bb_for_transform(self, transform: Transform) -> BB:
        return BB.newForCircle(self._center(transform), self.radius)

    def cache_bb(self) -> BB:
        return self._bb_for_transform(self._transform())

    def point_query(self, p) -> PointQueryInfo:
        center = self._center()
        delta = Vec2d(*p) - center
        distance = delta.length
        gradient = delta / distance if distance else Vec2d(0.0, 1.0)
        point = center + gradient * self.radius if distance else center
        return PointQueryInfo(self, point, distance - self.radius, gradient)

    def segment_query(self, start, end, radius=0):
        return _segment_circle_query(self, start, end, radius)

    def shapes_collide(self, b: Shape) -> ContactPointSet:
        return _shapes_collide(self, b)

    def unsafe_set_radius(self, r: float) -> None:
        self._radius = float(r)
        self._mark_query_dirty()

    def unsafe_set_offset(self, o) -> None:
        self._offset = Vec2d(*o)
        self._mark_query_dirty()


class Segment(Shape):
    def __init__(self, body, a, b, radius: float) -> None:
        super().__init__(body)
        self._a, self._b, self._radius = Vec2d(*a), Vec2d(*b), float(radius)
        self._neighbors = self._a, self._b

    @property
    def a(self) -> Vec2d:
        return self._a

    @property
    def b(self) -> Vec2d:
        return self._b

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def normal(self) -> Vec2d:
        return (self.b - self.a).perpendicular_normal()

    @property
    def area(self) -> float:
        return area_for_segment(self.a, self.b, self.radius)

    @property
    def moment(self) -> float:
        return moment_for_segment(self.mass, self.a, self.b, self.radius)

    @property
    def center_of_gravity(self) -> Vec2d:
        return (self.a + self.b) * 0.5

    def _ends(self):
        transform = self._transform()
        return transform @ self.a, transform @ self.b

    def _bb_for_transform(self, transform: Transform) -> BB:
        a, b = transform @ self.a, transform @ self.b
        return BB(
            min(a.x, b.x) - self.radius,
            min(a.y, b.y) - self.radius,
            max(a.x, b.x) + self.radius,
            max(a.y, b.y) + self.radius,
        )

    def cache_bb(self) -> BB:
        return self._bb_for_transform(self._transform())

    def point_query(self, p) -> PointQueryInfo:
        a, b = self._ends()
        query = Vec2d(*p)
        closest = _closest_point(query, a, b)
        delta = query - closest
        length = delta.length
        gradient = delta / length if length else self.normal
        return PointQueryInfo(self, closest + gradient * self.radius, length - self.radius, gradient)

    def segment_query(self, start, end, radius=0):
        return _segment_capsule_query(self, start, end, radius)

    def shapes_collide(self, b: Shape) -> ContactPointSet:
        return _shapes_collide(self, b)

    def unsafe_set_radius(self, r: float) -> None:
        self._radius = float(r)
        self._mark_query_dirty()

    def unsafe_set_endpoints(self, a, b) -> None:
        self._a, self._b = Vec2d(*a), Vec2d(*b)
        self._mark_query_dirty()

    def set_neighbors(self, prev, next) -> None:
        self._neighbors = Vec2d(*prev), Vec2d(*next)


class Poly(Shape):
    def __init__(self, body, vertices, transform=None, radius=0) -> None:
        super().__init__(body)
        matrix = transform or Transform.identity()
        self._vertices = tuple(matrix @ vertex for vertex in vertices)
        if len(self._vertices) < 3:
            raise ValueError("a polygon needs at least three vertices")
        self._radius = float(radius)

    @staticmethod
    def create_box(body, size=(10, 10), radius=0):
        width, height = size
        return Poly(body, [
            (-width / 2, -height / 2), (width / 2, -height / 2),
            (width / 2, height / 2), (-width / 2, height / 2),
        ], radius=radius)

    @staticmethod
    def create_box_bb(body, bb, radius=0):
        return Poly(body, [
            (bb.left, bb.bottom), (bb.right, bb.bottom),
            (bb.right, bb.top), (bb.left, bb.top),
        ], radius=radius)

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def area(self) -> float:
        return area_for_poly(self._vertices, self.radius)

    @property
    def moment(self) -> float:
        return moment_for_poly(self.mass, self._vertices, radius=self.radius)

    @property
    def center_of_gravity(self) -> Vec2d:
        return poly_centroid(self._vertices)

    def get_vertices(self) -> list[Vec2d]:
        return list(self._vertices)

    def _world_vertices(self, transform=None):
        matrix = transform or self._transform()
        return [matrix @ vertex for vertex in self._vertices]

    def _bb_for_transform(self, transform: Transform) -> BB:
        vertices = self._world_vertices(transform)
        return BB(
            min(v.x for v in vertices) - self.radius,
            min(v.y for v in vertices) - self.radius,
            max(v.x for v in vertices) + self.radius,
            max(v.y for v in vertices) + self.radius,
        )

    def cache_bb(self) -> BB:
        return self._bb_for_transform(self._transform())

    def point_query(self, p) -> PointQueryInfo:
        query = Vec2d(*p)
        vertices = self._world_vertices()
        best_point, best_distance = vertices[0], math.inf
        winding = 0
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            candidate = _closest_point(query, a, b)
            distance = query.get_distance(candidate)
            if distance < best_distance:
                best_point, best_distance = candidate, distance
            if a.y <= query.y < b.y and (b - a).cross(query - a) > 0:
                winding += 1
            elif b.y <= query.y < a.y and (b - a).cross(query - a) < 0:
                winding -= 1
        gradient = (query - best_point).normalized()
        inside = winding != 0
        if not gradient:
            gradient = Vec2d(0.0, 1.0)
        if inside:
            gradient = -gradient
        return PointQueryInfo(
            self,
            best_point + gradient * self.radius,
            (-best_distance if inside else best_distance) - self.radius,
            gradient,
        )

    def segment_query(self, start, end, radius=0):
        hit = None
        vertices = self._world_vertices()
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            candidate = _line_intersection(start, end, a, b)
            if candidate and (hit is None or candidate[0] < hit.alpha):
                alpha, point = candidate
                normal = (b - a).perpendicular_normal()
                hit = SegmentQueryInfo(self, point, normal, alpha)
        return hit

    def shapes_collide(self, b: Shape) -> ContactPointSet:
        raise NotImplementedError("polygon collision response is not in the covered subset")

    def unsafe_set_radius(self, radius: float) -> None:
        self._radius = float(radius)
        self._mark_query_dirty()

    def unsafe_set_vertices(self, vertices, transform=None) -> None:
        matrix = transform or Transform.identity()
        self._vertices = tuple(matrix @ vertex for vertex in vertices)
        self._mark_query_dirty()


def _closest_point(point: Vec2d, a: Vec2d, b: Vec2d) -> Vec2d:
    delta = b - a
    denominator = delta.length_squared
    t = min(max((point - a).dot(delta) / denominator, 0.0), 1.0) if denominator else 0.0
    return a + delta * t


def _segment_circle_query(shape: Circle, start, end, radius=0):
    start, end = Vec2d(*start), Vec2d(*end)
    center = shape._center()
    delta, offset = end - start, start - center
    combined = shape.radius + radius
    a = delta.length_squared
    b = 2 * offset.dot(delta)
    c = offset.length_squared - combined * combined
    discriminant = b * b - 4 * a * c
    if a == 0 or discriminant < 0:
        return None
    alpha = (-b - math.sqrt(discriminant)) / (2 * a)
    if not 0 <= alpha <= 1:
        return None
    center_hit = start + delta * alpha
    normal = (center_hit - center).normalized()
    return SegmentQueryInfo(shape, center + normal * shape.radius, normal, alpha)


def _segment_capsule_query(shape: Segment, start, end, radius=0):
    best = None
    a, b = shape._ends()
    combined = shape.radius + radius
    for center in (a, b):
        proxy = Circle(None, combined, center)
        hit = _segment_circle_query(proxy, start, end, 0)
        if hit and (best is None or hit.alpha < best.alpha):
            best = SegmentQueryInfo(shape, hit.point, hit.normal, hit.alpha)
    direction = b - a
    if direction:
        normal = direction.perpendicular_normal()
        for sign in (-1, 1):
            edge_a, edge_b = a + normal * combined * sign, b + normal * combined * sign
            candidate = _line_intersection(start, end, edge_a, edge_b)
            if candidate and (best is None or candidate[0] < best.alpha):
                best = SegmentQueryInfo(shape, candidate[1], normal * sign, candidate[0])
    return best


def _line_intersection(start, end, a, b):
    start, end, a, b = Vec2d(*start), Vec2d(*end), Vec2d(*a), Vec2d(*b)
    r, s = end - start, b - a
    denominator = r.cross(s)
    if denominator == 0:
        return None
    alpha, beta = (a - start).cross(s) / denominator, (a - start).cross(r) / denominator
    if 0 <= alpha <= 1 and 0 <= beta <= 1:
        return alpha, start + r * alpha
    return None


def _shapes_collide(a: Shape, b: Shape) -> ContactPointSet:
    if isinstance(a, Circle) and isinstance(b, Circle):
        ca, cb = a._center(), b._center()
        delta = cb - ca
        normal = delta.normalized() if delta else Vec2d(1.0, 0.0)
        distance = delta.length - a.radius - b.radius
        if distance > 0:
            return ContactPointSet(Vec2d.zero(), ())
        return ContactPointSet(normal, (
            ContactPoint(ca + normal * a.radius, cb - normal * b.radius, distance),
        ))
    if isinstance(a, Circle) and isinstance(b, Segment):
        sa, sb = b._ends()
        center = a._center()
        closest = _closest_point(center, sa, sb)
        delta = closest - center
        normal = delta.normalized() if delta else -b.normal
        distance = delta.length - a.radius - b.radius
        if distance > 0:
            return ContactPointSet(Vec2d.zero(), ())
        return ContactPointSet(normal, (
            ContactPoint(center + normal * a.radius, closest - normal * b.radius, distance),
        ))
    if isinstance(a, Segment) and isinstance(b, Circle):
        result = _shapes_collide(b, a)
        return ContactPointSet(-result.normal, tuple(
            ContactPoint(point.point_b, point.point_a, point.distance) for point in result.points
        ))
    raise NotImplementedError("only circle-circle and circle-segment contacts are covered")
