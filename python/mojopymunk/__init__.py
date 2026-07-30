"""Pymunk-compatible 2D rigid-body primitives backed by Mojo kernels."""

from . import util
from ._lib import build
from .body import Body
from .geometry import (
    BB,
    Transform,
    area_for_circle,
    area_for_poly,
    area_for_segment,
    convex_hull,
    is_convex,
    moment_for_box,
    moment_for_circle,
    moment_for_poly,
    moment_for_segment,
    moments_for_circles,
    poly_centroid,
)
from .shapes import (
    Circle,
    ContactPoint,
    ContactPointSet,
    PointQueryInfo,
    Poly,
    Segment,
    SegmentQueryInfo,
    ShapeFilter,
    ShapeQueryInfo,
)
from .space import Space
from .vec2d import Vec2d

version = "0.1.0"
__version__ = version

__all__ = [
    "BB",
    "Body",
    "Circle",
    "ContactPoint",
    "ContactPointSet",
    "PointQueryInfo",
    "Poly",
    "Segment",
    "SegmentQueryInfo",
    "ShapeFilter",
    "ShapeQueryInfo",
    "Space",
    "Transform",
    "Vec2d",
    "area_for_circle",
    "area_for_poly",
    "area_for_segment",
    "build",
    "convex_hull",
    "is_convex",
    "moment_for_box",
    "moment_for_circle",
    "moment_for_poly",
    "moment_for_segment",
    "moments_for_circles",
    "poly_centroid",
    "util",
]
