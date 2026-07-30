from __future__ import annotations

import math

import numpy as np

from .vec2d import Vec2d

MASS, MOMENT, PX, PY, VX, VY, FX, FY, ANGLE, W, TORQUE, TYPE, COGX, COGY, VLIM, WLIM = range(16)


class Body:
    DYNAMIC = 0
    KINEMATIC = 1
    STATIC = 2

    def __init__(self, mass: float = 0, moment: float = 0, body_type: int = 0) -> None:
        self._state = np.zeros(16, dtype=np.float64)
        self._state[MASS] = mass
        self._state[MOMENT] = moment
        self._state[TYPE] = body_type
        self._state[VLIM] = math.inf
        self._state[WLIM] = math.inf
        self._space = None
        self._index = -1
        self._shapes: set = set()
        self._constraints: set = set()
        self._velocity_func = Body.update_velocity
        self._position_func = Body.update_position

    def _data(self) -> np.ndarray:
        if self._space is not None and self._index >= 0:
            return self._space._state[self._index]
        return self._state

    def _bind(self, space, index: int) -> None:
        self._space, self._index = space, index

    def _unbind(self) -> None:
        self._state = self._data().copy()
        self._space, self._index = None, -1

    def _mark_query_dirty(self) -> None:
        if self._space is not None:
            self._space._query_dirty = True

    @property
    def mass(self) -> float:
        return float(self._data()[MASS])

    @mass.setter
    def mass(self, value: float) -> None:
        self._data()[MASS] = value

    @property
    def moment(self) -> float:
        return float(self._data()[MOMENT])

    @moment.setter
    def moment(self, value: float) -> None:
        self._data()[MOMENT] = value

    @property
    def body_type(self) -> int:
        return int(self._data()[TYPE])

    @body_type.setter
    def body_type(self, value: int) -> None:
        self._data()[TYPE] = value

    @property
    def position(self) -> Vec2d:
        state = self._data()
        return Vec2d(state[PX], state[PY])

    @position.setter
    def position(self, value) -> None:
        self._data()[PX:PY + 1] = value
        self._mark_query_dirty()

    @property
    def velocity(self) -> Vec2d:
        state = self._data()
        return Vec2d(state[VX], state[VY])

    @velocity.setter
    def velocity(self, value) -> None:
        self._data()[VX:VY + 1] = value

    @property
    def force(self) -> Vec2d:
        state = self._data()
        return Vec2d(state[FX], state[FY])

    @force.setter
    def force(self, value) -> None:
        self._data()[FX:FY + 1] = value

    @property
    def angle(self) -> float:
        return float(self._data()[ANGLE])

    @angle.setter
    def angle(self, value: float) -> None:
        self._data()[ANGLE] = value
        self._mark_query_dirty()

    @property
    def angular_velocity(self) -> float:
        return float(self._data()[W])

    @angular_velocity.setter
    def angular_velocity(self, value: float) -> None:
        self._data()[W] = value

    @property
    def torque(self) -> float:
        return float(self._data()[TORQUE])

    @torque.setter
    def torque(self, value: float) -> None:
        self._data()[TORQUE] = value

    @property
    def center_of_gravity(self) -> Vec2d:
        state = self._data()
        return Vec2d(state[COGX], state[COGY])

    @center_of_gravity.setter
    def center_of_gravity(self, value) -> None:
        self._data()[COGX:COGY + 1] = value

    @property
    def rotation_vector(self) -> Vec2d:
        return Vec2d(math.cos(self.angle), math.sin(self.angle))

    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * self.velocity.length_squared + 0.5 * self.moment * self.angular_velocity**2

    @property
    def shapes(self):
        return set(self._shapes)

    @property
    def constraints(self):
        return set(self._constraints)

    @property
    def space(self):
        return self._space

    @property
    def is_sleeping(self) -> bool:
        return False

    @property
    def velocity_func(self):
        return self._velocity_func

    @velocity_func.setter
    def velocity_func(self, function) -> None:
        self._velocity_func = function
        if self._space is not None:
            self._space._refresh_body_callbacks(self)

    @property
    def position_func(self):
        return self._position_func

    @position_func.setter
    def position_func(self, function) -> None:
        self._position_func = function
        if self._space is not None:
            self._space._refresh_body_callbacks(self)

    @staticmethod
    def update_velocity(body: "Body", gravity, damping: float, dt: float) -> None:
        if body.body_type != Body.DYNAMIC:
            return
        if body.mass > 0:
            body.velocity = body.velocity * damping + (
                Vec2d(*gravity) + body.force / body.mass
            ) * dt
        if body.moment > 0:
            body.angular_velocity = (
                body.angular_velocity * damping + body.torque / body.moment * dt
            )

    @staticmethod
    def update_position(body: "Body", dt: float) -> None:
        if body.body_type != Body.STATIC:
            world_cog = body.local_to_world(body.center_of_gravity)
            body.angle += body.angular_velocity * dt
            body.position = (
                world_cog + body.velocity * dt
                - body.center_of_gravity.rotated(body.angle)
            )

    def local_to_world(self, v) -> Vec2d:
        return self.position + Vec2d(*v).rotated(self.angle)

    def world_to_local(self, v) -> Vec2d:
        return (Vec2d(*v) - self.position).rotated(-self.angle)

    def velocity_at_world_point(self, point) -> Vec2d:
        offset = Vec2d(*point) - self.local_to_world(self.center_of_gravity)
        return self.velocity + Vec2d(-offset.y, offset.x) * self.angular_velocity

    def velocity_at_local_point(self, point) -> Vec2d:
        return self.velocity_at_world_point(self.local_to_world(point))

    def apply_force_at_world_point(self, force, point) -> None:
        vector = Vec2d(*force)
        self.force += vector
        self.torque += (
            Vec2d(*point) - self.local_to_world(self.center_of_gravity)
        ).cross(vector)

    def apply_force_at_local_point(self, force, point=(0, 0)) -> None:
        world_force = Vec2d(*force).rotated(self.angle)
        self.apply_force_at_world_point(world_force, self.local_to_world(point))

    def apply_impulse_at_world_point(self, impulse, point) -> None:
        if self.body_type != Body.DYNAMIC:
            return
        vector = Vec2d(*impulse)
        if self.mass > 0:
            self.velocity += vector / self.mass
        if self.moment > 0:
            self.angular_velocity += (
                Vec2d(*point) - self.local_to_world(self.center_of_gravity)
            ).cross(vector) / self.moment

    def apply_impulse_at_local_point(self, impulse, point=(0, 0)) -> None:
        world_impulse = Vec2d(*impulse).rotated(self.angle)
        self.apply_impulse_at_world_point(world_impulse, self.local_to_world(point))

    def activate(self) -> None:
        return None

    def sleep(self) -> None:
        return None

    def sleep_with_group(self, body: "Body") -> None:
        return None

    def copy(self):
        duplicate = Body(self.mass, self.moment, self.body_type)
        duplicate._state = self._data().copy()
        return duplicate
