import math

import numpy as np
import pymunk
import pymunk.util
import pytest

import mojopymunk as mojo


@pytest.mark.parametrize(
    "mass, inner, outer, offset",
    [(1, 0, 2, (0, 0)), (3.5, 1, 4, (2, -7)), (0.25, 3, 5, (-1, 0.5))],
)
def test_circle_mass_properties(mass, inner, outer, offset):
    assert mojo.moment_for_circle(mass, inner, outer, offset) == pytest.approx(
        pymunk.moment_for_circle(mass, inner, outer, offset)
    )
    assert mojo.area_for_circle(inner, outer) == pytest.approx(
        pymunk.area_for_circle(inner, outer)
    )


@pytest.mark.parametrize("radius", [0.0, 0.5, 2.0])
def test_segment_mass_properties(radius):
    args = (4.5, (-3.0, 2.0), (5.0, 7.0), radius)
    assert mojo.moment_for_segment(*args) == pytest.approx(pymunk.moment_for_segment(*args))
    assert mojo.area_for_segment(*args[1:]) == pytest.approx(
        pymunk.area_for_segment(*args[1:])
    )


def test_box_moment():
    assert mojo.moment_for_box(12.5, (9.0, 3.0)) == pytest.approx(
        pymunk.moment_for_box(12.5, (9.0, 3.0))
    )


@pytest.mark.parametrize("radius", [0.0, 0.25, 1.5])
def test_polygon_mass_properties(radius):
    vertices = [(-3, -1), (2, -2), (5, 1), (1, 4), (-2, 3)]
    assert mojo.moment_for_poly(7, vertices, (2, -1), radius) == pytest.approx(
        pymunk.moment_for_poly(7, vertices, (2, -1), radius), rel=1e-14
    )
    assert mojo.area_for_poly(vertices, radius) == pytest.approx(
        pymunk.area_for_poly(vertices, radius), rel=1e-14
    )


def test_polygon_centroid():
    vertices = [(-3, -1), (2, -2), (5, 1), (1, 4), (-2, 3)]
    ours = mojo.poly_centroid(vertices)
    theirs = pymunk.Poly(None, vertices).center_of_gravity
    assert ours == pytest.approx(theirs)


def test_batch_circle_moments():
    rng = np.random.default_rng(4)
    mass = rng.uniform(0.1, 10, 1000)
    inner = rng.uniform(0, 2, 1000)
    outer = inner + rng.uniform(0, 4, 1000)
    offset = rng.normal(size=(1000, 2))
    ours = mojo.moments_for_circles(mass, inner, outer, offset)
    theirs = np.array(
        [
            pymunk.moment_for_circle(m, i, o, xy)
            for m, i, o, xy in zip(mass, inner, outer, map(tuple, offset))
        ]
    )
    assert np.allclose(ours, theirs, rtol=1e-14)


def test_ffi_inputs_reject_unsafe_or_degenerate_buffers():
    assert mojo.moments_for_circles([], [], [], np.empty((0, 2))).shape == (0,)
    with pytest.raises(TypeError, match="narrow"):
        mojo.moment_for_poly(1, np.ones((3, 2), dtype=np.longdouble))
    with pytest.raises(ValueError, match="at least"):
        mojo.area_for_poly([])
    with pytest.raises(ValueError, match="zero area"):
        mojo.poly_centroid([(0, 0), (1, 0), (2, 0)])


def test_vec2d_arithmetic_and_geometry():
    ours, theirs = mojo.Vec2d(3.5, -2.0), pymunk.Vec2d(3.5, -2.0)
    other = (-4.0, 7.0)
    for operation in (
        lambda v: v + other,
        lambda v: v - other,
        lambda v: v * 2.5,
        lambda v: v / 3,
        lambda v: v.perpendicular(),
        lambda v: v.perpendicular_normal(),
        lambda v: v.rotated(0.73),
        lambda v: v.projection(other),
        lambda v: v.interpolate_to(other, 0.3),
    ):
        assert operation(ours) == pytest.approx(operation(theirs))
    assert ours.length == pytest.approx(theirs.length)
    assert ours.dot(other) == pytest.approx(theirs.dot(other))
    assert ours.cross(other) == pytest.approx(theirs.cross(other))
    assert ours.get_angle_between(other) == pytest.approx(theirs.get_angle_between(other))


def test_vec2d_constructors_and_zero_normalization():
    assert mojo.Vec2d.from_polar(4, 0.7) == pytest.approx(
        pymunk.Vec2d.from_polar(4, 0.7)
    )
    assert mojo.Vec2d.zero().normalized() == pymunk.Vec2d.zero().normalized()
    assert mojo.Vec2d.ones() == pymunk.Vec2d.ones()


def test_transform_parity():
    ours = mojo.Transform.translation(3, 4).rotated(0.4).scaled(2)
    theirs = pymunk.Transform.translation(3, 4).rotated(0.4).scaled(2)
    point = (7, -2)
    assert ours == pytest.approx(theirs)
    assert ours @ point == pytest.approx(theirs @ point)
    assert ours.inverted() == pytest.approx(theirs.inverted())


def test_bb_parity():
    ours, theirs = mojo.BB(-2, -3, 7, 11), pymunk.BB(-2, -3, 7, 11)
    other_ours, other_theirs = mojo.BB(5, -5, 9, 2), pymunk.BB(5, -5, 9, 2)
    assert ours.area() == theirs.area()
    assert ours.center() == theirs.center()
    assert ours.merge(other_ours) == other_theirs.merge(theirs)
    assert ours.clamp_vect((20, -8)) == theirs.clamp_vect((20, -8))
    for a, b in [((-10, 0), (10, 0)), ((-10, 20), (10, 20))]:
        assert ours.intersects_segment(a, b) == theirs.intersects_segment(a, b)
        assert ours.segment_query(a, b) == theirs.segment_query(a, b)


def test_convex_helpers():
    points = [(0, 0), (2, 0), (1, 0.5), (2, 2), (0, 2), (0.5, 1)]
    ours = mojo.convex_hull(points)
    theirs = pymunk.util.convex_hull(points)
    assert set(ours) == set(theirs)
    assert mojo.is_convex(ours) == pymunk.util.is_convex(theirs)
    assert not mojo.is_convex([(0, 0), (2, 0), (1, 0.5), (2, 2), (0, 2)])
