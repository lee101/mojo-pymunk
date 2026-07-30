import numpy as np
import pymunk
import pytest

import mojopymunk as mojo
from mojopymunk._lib import addr, lib, parallel_context


def paired_bodies():
    ours = mojo.Body(3.0, 7.0)
    theirs = pymunk.Body(3.0, 7.0)
    for body in (ours, theirs):
        body.center_of_gravity = 0.3, -0.2
        body.angle = 0.4
        body.position = 4, -2
        body.velocity = -1, 3
        body.angular_velocity = -0.7
    return ours, theirs


def test_body_transforms_and_point_velocity():
    ours, theirs = paired_bodies()
    for point in [(0, 0), (2, -5), (-3.5, 4.2)]:
        assert ours.local_to_world(point) == pytest.approx(theirs.local_to_world(point))
        assert ours.world_to_local(point) == pytest.approx(theirs.world_to_local(point))
        assert ours.velocity_at_world_point(point) == pytest.approx(
            theirs.velocity_at_world_point(point)
        )
        assert ours.velocity_at_local_point(point) == pytest.approx(
            theirs.velocity_at_local_point(point)
        )


def test_force_and_impulse_application():
    ours, theirs = paired_bodies()
    ours.apply_force_at_world_point((8, -3), (5, 6))
    theirs.apply_force_at_world_point((8, -3), (5, 6))
    ours.apply_force_at_local_point((-2, 4), (1, -1))
    theirs.apply_force_at_local_point((-2, 4), (1, -1))
    assert ours.force == pytest.approx(theirs.force)
    assert ours.torque == pytest.approx(theirs.torque)
    ours.apply_impulse_at_world_point((1, 5), (-2, 3))
    theirs.apply_impulse_at_world_point((1, 5), (-2, 3))
    ours.apply_impulse_at_local_point((3, -2), (0.5, 0.25))
    theirs.apply_impulse_at_local_point((3, -2), (0.5, 0.25))
    assert ours.velocity == pytest.approx(theirs.velocity)
    assert ours.angular_velocity == pytest.approx(theirs.angular_velocity)


def test_body_update_functions():
    ours, theirs = paired_bodies()
    ours.force = theirs.force = (5, -2)
    ours.torque = theirs.torque = 3
    mojo.Body.update_velocity(ours, (0, -9.8), 0.97, 0.1)
    pymunk.Body.update_velocity(theirs, (0, -9.8), 0.97, 0.1)
    assert ours.velocity == pytest.approx(theirs.velocity)
    assert ours.angular_velocity == pytest.approx(theirs.angular_velocity)
    mojo.Body.update_position(ours, 0.1)
    pymunk.Body.update_position(theirs, 0.1)
    assert ours.position == pytest.approx(theirs.position)
    assert ours.angle == pytest.approx(theirs.angle)


def test_callback_tracking_updates_after_body_is_bound():
    space = mojo.Space()
    body = mojo.Body(1, 1)
    space.add(body)
    calls = []

    def update_position(target, dt):
        calls.append(dt)
        target.position = (7, 8)

    body.position_func = update_position
    space.step(0.25)
    assert calls == [0.25]
    assert body.position == (7, 8)
    body.position_func = mojo.Body.update_position
    space.step(0.25)
    assert body.position == (7, 8)


@pytest.mark.parametrize("count", [17, 131_073])
def test_integrate_serial_and_parallel_threshold_paths(count):
    state = np.zeros((count, 16), dtype=np.float64)
    state[:, 0] = 2.0
    state[:, 1] = 4.0
    state[:, 2:4] = (1.0, 2.0)
    state[:, 4:6] = (3.0, 4.0)
    state[:, 6:8] = (5.0, 6.0)
    state[:, 8] = 0.2
    state[:, 9] = 0.3
    state[:, 10] = 0.7
    state[:, 14:16] = np.inf
    context = parallel_context() if count >= 131_072 else 0
    lib().mp_integrate(addr(state), count, 1.5, -2.0, 0.9, 0.01, context)
    assert np.allclose(state[:, 2], 1.03)
    assert np.allclose(state[:, 3], 2.04)
    assert np.allclose(state[:, 4], 2.74)
    assert np.allclose(state[:, 5], 3.61)
    assert np.allclose(state[:, 8], 0.203)
    assert np.allclose(state[:, 9], 0.27175)
    assert np.count_nonzero(state[:, 6:8]) == 0
    assert np.count_nonzero(state[:, 10]) == 0


def test_space_free_flight_parity():
    ours, theirs = mojo.Space(), pymunk.Space()
    for space in (ours, theirs):
        space.gravity = 2, -9.81
        space.damping = 0.93
    our_bodies, their_bodies = [], []
    rng = np.random.default_rng(9)
    for _ in range(50):
        a, b = mojo.Body(2, 5), pymunk.Body(2, 5)
        position = tuple(rng.normal(size=2))
        velocity = tuple(rng.normal(size=2))
        force = tuple(rng.normal(size=2))
        for body in (a, b):
            body.position = position
            body.velocity = velocity
            body.force = force
            body.angular_velocity = 0.2
            body.torque = -0.5
        ours.add(a)
        theirs.add(b)
        our_bodies.append(a)
        their_bodies.append(b)
    for _ in range(20):
        ours.step(1 / 120)
        theirs.step(1 / 120)
    assert np.allclose([b.position for b in our_bodies], [b.position for b in their_bodies])
    assert np.allclose([b.velocity for b in our_bodies], [b.velocity for b in their_bodies])
    assert np.allclose([b.angle for b in our_bodies], [b.angle for b in their_bodies])
    assert np.allclose(
        [b.angular_velocity for b in our_bodies],
        [b.angular_velocity for b in their_bodies],
    )


def test_kinematic_and_static_bodies():
    ours = mojo.Space()
    kinematic = mojo.Body(body_type=mojo.Body.KINEMATIC)
    static = mojo.Body(body_type=mojo.Body.STATIC)
    kinematic.velocity = (3, 4)
    static.velocity = (9, 9)
    ours.add(kinematic, static)
    ours.step(0.5)
    assert kinematic.position == (1.5, 2.0)
    assert static.position == (0.0, 0.0)


def test_shape_membership_and_remove():
    space = mojo.Space()
    body = mojo.Body(1, 1)
    shape = mojo.Circle(body, 2)
    space.add(body, shape)
    assert body in space.bodies and shape in space.shapes
    assert shape in body.shapes and shape.space is space and body.space is space
    space.remove(shape, body)
    assert not space.bodies and not space.shapes
    assert shape.space is None and body.space is None


def test_head_on_elastic_circle_collision_matches_velocity():
    spaces = mojo.Space(), pymunk.Space()
    velocities = []
    for module, space in zip((mojo, pymunk), spaces):
        bodies = [module.Body(1, module.moment_for_circle(1, 0, 1)) for _ in range(2)]
        bodies[0].position, bodies[1].position = (-1, 0), (1, 0)
        bodies[0].velocity, bodies[1].velocity = (2, 0), (-2, 0)
        shapes = [module.Circle(body, 1) for body in bodies]
        for shape in shapes:
            shape.elasticity = 1
        space.add(*bodies, *shapes)
        space.step(0.01)
        velocities.append([body.velocity for body in bodies])
    assert np.allclose(velocities[0], velocities[1], atol=1e-12)


def test_shape_filter_prevents_collision():
    space = mojo.Space()
    bodies = [mojo.Body(1, 1), mojo.Body(1, 1)]
    bodies[0].position, bodies[1].position = (-0.5, 0), (0.5, 0)
    bodies[0].velocity, bodies[1].velocity = (1, 0), (-1, 0)
    shapes = [mojo.Circle(body, 1) for body in bodies]
    shapes[0].filter = mojo.ShapeFilter(group=4)
    shapes[1].filter = mojo.ShapeFilter(group=4)
    space.add(*bodies, *shapes)
    space.step(0.01)
    assert bodies[0].velocity.x == 1
    assert bodies[1].velocity.x == -1


def test_sensor_prevents_collision_response():
    space = mojo.Space()
    bodies = [mojo.Body(1, 1), mojo.Body(1, 1)]
    bodies[0].position, bodies[1].position = (-0.5, 0), (0.5, 0)
    bodies[0].velocity, bodies[1].velocity = (1, 0), (-1, 0)
    shapes = [mojo.Circle(body, 1) for body in bodies]
    shapes[0].sensor = True
    space.add(*bodies, *shapes)
    space.step(0.01)
    assert bodies[0].velocity.x == 1
    assert bodies[1].velocity.x == -1


def test_circle_segment_collision_is_independent_of_add_order():
    space = mojo.Space()
    floor = mojo.Segment(None, (-5, 0), (5, 0), 0)
    body = mojo.Body(1, 1)
    body.position = (0, 0.5)
    body.velocity = (0, -1)
    circle = mojo.Circle(body, 1)
    circle.elasticity = floor.elasticity = 1
    space.add(floor, body, circle)
    space.step(0.01)
    assert body.velocity.y > 0
