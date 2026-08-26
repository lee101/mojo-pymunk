import pymunk
import pytest

import mojopymunk as mojo


@pytest.mark.parametrize("point", [(0, 0), (5, 4), (-3, 2)])
def test_circle_point_query(point):
    ours_body, their_body = mojo.Body(1, 1), pymunk.Body(1, 1)
    ours_body.position = their_body.position = (2, -1)
    ours_body.angle = their_body.angle = 0.3
    ours = mojo.Circle(ours_body, 2.5, (1, -0.5))
    theirs = pymunk.Circle(their_body, 2.5, (1, -0.5))
    ours.cache_bb()
    theirs.cache_bb()
    a, b = ours.point_query(point), theirs.point_query(point)
    assert a.point == pytest.approx(b.point)
    assert a.distance == pytest.approx(b.distance)
    assert a.gradient == pytest.approx(b.gradient)
    assert ours.cache_bb() == pytest.approx(theirs.cache_bb())


@pytest.mark.parametrize("point", [(0, 0), (5, 4), (-3, 2)])
def test_segment_point_query(point):
    ours_body, their_body = mojo.Body(1, 1), pymunk.Body(1, 1)
    ours_body.position = their_body.position = (2, -1)
    ours_body.angle = their_body.angle = 0.3
    ours = mojo.Segment(ours_body, (-3, 0), (4, 1), 0.7)
    theirs = pymunk.Segment(their_body, (-3, 0), (4, 1), 0.7)
    ours.cache_bb()
    theirs.cache_bb()
    a, b = ours.point_query(point), theirs.point_query(point)
    assert a.point == pytest.approx(b.point)
    assert a.distance == pytest.approx(b.distance)
    assert a.gradient == pytest.approx(b.gradient)
    assert ours.cache_bb() == pytest.approx(theirs.cache_bb())


def test_poly_construction_and_point_queries():
    vertices = [(-3, -1), (2, -2), (4, 1), (1, 4), (-2, 3)]
    ours_body, their_body = mojo.Body(1, 1), pymunk.Body(1, 1)
    ours_body.position = their_body.position = (1, 2)
    ours_body.angle = their_body.angle = -0.2
    ours, theirs = mojo.Poly(ours_body, vertices, radius=0), pymunk.Poly(
        their_body, vertices, radius=0
    )
    ours.cache_bb()
    theirs.cache_bb()
    assert ours.get_vertices() == pytest.approx(theirs.get_vertices())
    assert ours.area == pytest.approx(theirs.area)
    assert ours.center_of_gravity == pytest.approx(theirs.center_of_gravity)
    assert ours.cache_bb() == pytest.approx(theirs.cache_bb())
    for point in [(1, 2), (8, 2), (-3, -4)]:
        a, b = ours.point_query(point), theirs.point_query(point)
        assert a.point == pytest.approx(b.point)
        assert a.distance == pytest.approx(b.distance)
        assert a.gradient == pytest.approx(b.gradient)


def test_circle_segment_queries():
    ours = mojo.Circle(mojo.Body(body_type=mojo.Body.STATIC), 2, (3, 1))
    theirs = pymunk.Circle(pymunk.Body(body_type=pymunk.Body.STATIC), 2, (3, 1))
    ours.cache_bb()
    theirs.cache_bb()
    a, b = ours.segment_query((-5, 1), (10, 1)), theirs.segment_query((-5, 1), (10, 1))
    assert a.point == pytest.approx(b.point)
    assert a.normal == pytest.approx(b.normal)
    assert a.alpha == pytest.approx(b.alpha)


def test_space_queries():
    ours, theirs = mojo.Space(), pymunk.Space()
    our_shapes, their_shapes = [], []
    for module, space, target in (
        (mojo, ours, our_shapes),
        (pymunk, theirs, their_shapes),
    ):
        body = module.Body(body_type=module.Body.STATIC)
        shapes = [
            module.Circle(body, 2, (-3, 0)),
            module.Circle(body, 1, (4, 1)),
            module.Segment(body, (-5, -3), (5, -3), 0.5),
        ]
        space.add(body, *shapes)
        target.extend(shapes)
    query_filter_ours, query_filter_theirs = mojo.ShapeFilter(), pymunk.ShapeFilter()
    a = ours.point_query((0, 0), 10, query_filter_ours)
    b = theirs.point_query((0, 0), 10, query_filter_theirs)
    a = sorted(a, key=lambda x: our_shapes.index(x.shape))
    b = sorted(b, key=lambda x: their_shapes.index(x.shape))
    assert [our_shapes.index(x.shape) for x in a] == [their_shapes.index(x.shape) for x in b]
    assert [x.distance for x in a] == pytest.approx([x.distance for x in b])
    a = ours.segment_query((-10, 0), (10, 0), 0, query_filter_ours)
    b = theirs.segment_query((-10, 0), (10, 0), 0, query_filter_theirs)
    assert [our_shapes.index(x.shape) for x in a] == [their_shapes.index(x.shape) for x in b]
    assert [x.alpha for x in a] == pytest.approx([x.alpha for x in b])
    assert sorted([
        our_shapes.index(shape) for shape in ours.bb_query(mojo.BB(-6, -4, 0, 2), query_filter_ours)
    ]) == sorted([
        their_shapes.index(shape)
        for shape in theirs.bb_query(pymunk.BB(-6, -4, 0, 2), query_filter_theirs)
    ])


def test_point_query_nearest_simd_tail_and_cache_invalidation():
    ours, theirs = mojo.Space(), pymunk.Space()
    our_bodies, their_bodies, our_shapes, their_shapes = [], [], [], []
    for x in (-8, -4, 1, 5, 9):
        our_body = mojo.Body(body_type=mojo.Body.STATIC)
        their_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        our_body.position = their_body.position = (x, 0)
        our_shape = mojo.Circle(our_body, 0.75)
        their_shape = pymunk.Circle(their_body, 0.75)
        our_bodies.append(our_body)
        their_bodies.append(their_body)
        our_shapes.append(our_shape)
        their_shapes.append(their_shape)
    ours.add(*our_bodies, *our_shapes)
    theirs.add(*their_bodies, *their_shapes)
    query_filter_ours = mojo.ShapeFilter()
    query_filter_theirs = pymunk.ShapeFilter()
    a = ours.point_query_nearest((0, 0), 20, query_filter_ours)
    b = theirs.point_query_nearest((0, 0), 20, query_filter_theirs)
    assert our_shapes.index(a.shape) == their_shapes.index(b.shape)
    assert a.point == pytest.approx(b.point)
    assert a.distance == pytest.approx(b.distance)
    assert a.gradient == pytest.approx(b.gradient)
    our_bodies[-1].position = their_bodies[-1].position = (0.1, 0)
    ours.reindex_shape(our_shapes[-1])
    theirs.reindex_shape(their_shapes[-1])
    a = ours.point_query_nearest((0, 0), 20, query_filter_ours)
    b = theirs.point_query_nearest((0, 0), 20, query_filter_theirs)
    assert our_shapes.index(a.shape) == their_shapes.index(b.shape)
    assert a.distance == pytest.approx(b.distance)


def test_point_query_nearest_bvh_tail_and_filters():
    ours, theirs = mojo.Space(), pymunk.Space()
    our_shapes, their_shapes = [], []
    for index in range(67):
        category = 1 << (index % 31)
        for module, space, shapes in (
            (mojo, ours, our_shapes),
            (pymunk, theirs, their_shapes),
        ):
            body = module.Body(body_type=module.Body.STATIC)
            body.position = (index - 33, (index % 5) - 2)
            shape = module.Circle(body, 0.4)
            shape.filter = module.ShapeFilter(categories=category)
            space.add(body, shape)
            shapes.append(shape)
    our_filter = mojo.ShapeFilter(mask=1 << 17)
    their_filter = pymunk.ShapeFilter(mask=1 << 17)
    a = ours.point_query_nearest((0, 0), 100, our_filter)
    b = theirs.point_query_nearest((0, 0), 100, their_filter)
    assert our_shapes.index(a.shape) == their_shapes.index(b.shape)
    assert a.point == pytest.approx(b.point)
    assert a.distance == pytest.approx(b.distance)
    a = ours.point_query_nearest((0.125, 0.25), 100, mojo.ShapeFilter())
    b = theirs.point_query_nearest((0.125, 0.25), 100, pymunk.ShapeFilter())
    assert our_shapes.index(a.shape) == their_shapes.index(b.shape)
    assert a.point == pytest.approx(b.point)
    assert a.distance == pytest.approx(b.distance)


def test_circle_circle_contact_parity():
    our_body = mojo.Body(body_type=mojo.Body.STATIC)
    their_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    our_a, our_b = mojo.Circle(our_body, 2, (0, 0)), mojo.Circle(our_body, 2, (3, 0))
    their_a, their_b = pymunk.Circle(their_body, 2, (0, 0)), pymunk.Circle(their_body, 2, (3, 0))
    for shape in (our_a, our_b, their_a, their_b):
        shape.cache_bb()
    ours, theirs = our_a.shapes_collide(our_b), their_a.shapes_collide(their_b)
    assert ours.normal == pytest.approx(theirs.normal)
    assert len(ours.points) == len(theirs.points)
    assert ours.points[0].distance == pytest.approx(theirs.points[0].distance)


def test_shape_query_does_not_swallow_unsupported_pairs():
    space = mojo.Space()
    body = mojo.Body(body_type=mojo.Body.STATIC)
    circle = mojo.Circle(body, 1)
    polygon = mojo.Poly(body, [(-1, -1), (1, -1), (0, 1)])
    space.add(body, circle, polygon)
    with pytest.raises(NotImplementedError):
        space.shape_query(circle)
