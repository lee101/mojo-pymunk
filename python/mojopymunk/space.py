from __future__ import annotations

import math

import numpy as np

from ._lib import addr, lib, parallel_context
from .body import Body, FX, FY, TORQUE
from .geometry import BB
from .shapes import Circle, PointQueryInfo, Segment, SegmentQueryInfo, Shape, ShapeFilter, ShapeQueryInfo
from .vec2d import Vec2d


class Space:
    def __init__(self, threaded: bool = False) -> None:
        self._bodies: list[Body] = []
        self._shapes: list[Shape] = []
        self._custom_bodies: set[Body] = set()
        self._shape_dirty = True
        self._shape_data = np.empty((0, 16), dtype=np.float64)
        self._query_dirty = True
        self._circle_query_shapes: list[Circle] = []
        self._noncircle_query_shapes: list[Shape] = []
        self._circle_query_data = np.empty((6, 0), dtype=np.float64)
        self._circle_query_nodes = None
        self._point_query_result = np.empty(5, dtype=np.float64)
        self._state = np.zeros((1, 16), dtype=np.float64)
        self.static_body = Body(body_type=Body.STATIC)
        self._state[0] = self.static_body._state
        self.static_body._bind(self, 0)
        self._gravity = Vec2d.zero()
        self.damping = 1.0
        self.iterations = 10
        self.collision_slop = 0.1
        self.collision_bias = (1.0 - 0.1) ** 60
        self.collision_persistence = 3
        self.sleep_time_threshold = math.inf
        self.idle_speed_threshold = 0.0
        self.threads = 1
        self.current_time_step = 0.0
        self._threaded = threaded

    @property
    def bodies(self):
        return list(self._bodies)

    @property
    def gravity(self) -> Vec2d:
        return self._gravity

    @gravity.setter
    def gravity(self, value) -> None:
        self._gravity = Vec2d(*value)

    @property
    def shapes(self):
        return list(self._shapes)

    @property
    def constraints(self):
        return []

    def _rebuild_state(self, extra=()) -> None:
        bodies = self._bodies + list(extra)
        state = np.empty((len(bodies) + 1, 16), dtype=np.float64)
        state[0] = self.static_body._data()
        for i, body in enumerate(bodies, 1):
            state[i] = body._data()
        self._state = state
        self.static_body._bind(self, 0)
        for i, body in enumerate(bodies, 1):
            body._bind(self, i)
        self._query_dirty = True
        self._shape_dirty = True

    def _refresh_body_callbacks(self, body: Body) -> None:
        custom = (
            body.velocity_func is not Body.update_velocity
            or body.position_func is not Body.update_position
        )
        if custom:
            self._custom_bodies.add(body)
        else:
            self._custom_bodies.discard(body)

    def add(self, *objs) -> None:
        new_bodies = [obj for obj in objs if isinstance(obj, Body) and obj is not self.static_body]
        for body in new_bodies:
            if body.space is not None and body.space is not self:
                raise ValueError("body already belongs to another space")
            if body not in self._bodies:
                self._bodies.append(body)
                self._refresh_body_callbacks(body)
        if new_bodies:
            self._rebuild_state()
        for shape in (obj for obj in objs if isinstance(obj, Shape)):
            if shape in self._shapes:
                continue
            if shape.body is None:
                shape.body = self.static_body
            if shape.body is not self.static_body and shape.body not in self._bodies:
                raise ValueError("shape body must be added to the space first")
            shape._space = self
            self._shapes.append(shape)
            self._query_dirty = True
            self._shape_dirty = True
        unsupported = [obj for obj in objs if not isinstance(obj, (Body, Shape))]
        if unsupported:
            raise NotImplementedError("constraints are not in the covered subset")

    def remove(self, *objs) -> None:
        for shape in (obj for obj in objs if isinstance(obj, Shape)):
            if shape in self._shapes:
                self._shapes.remove(shape)
                shape._space = None
                self._query_dirty = True
                self._shape_dirty = True
        bodies = [obj for obj in objs if isinstance(obj, Body) and obj in self._bodies]
        for body in bodies:
            if any(shape.body is body for shape in self._shapes):
                raise ValueError("remove a body's shapes before removing the body")
            body._unbind()
            self._bodies.remove(body)
            self._custom_bodies.discard(body)
        if bodies:
            self._rebuild_state()

    def _shape_state(self) -> np.ndarray:
        if not self._shape_dirty:
            return self._shape_data
        # The Mojo solver's outer loop is specialized for circles. Keep circles
        # first so circle-segment handling is independent of insertion order.
        collision_shapes = sorted(
            self._shapes, key=lambda shape: 0 if isinstance(shape, Circle) else 1
        )
        rows = np.zeros((len(collision_shapes), 16), dtype=np.float64)
        for i, shape in enumerate(collision_shapes):
            rows[i, 0] = 0 if isinstance(shape, Circle) else 1 if isinstance(shape, Segment) else 2
            rows[i, 1] = shape.body._index
            if isinstance(shape, Circle):
                rows[i, 2:4] = shape.offset
                rows[i, 6] = shape.radius
            elif isinstance(shape, Segment):
                rows[i, 2:4] = shape.a
                rows[i, 4:6] = shape.b
                rows[i, 6] = shape.radius
            rows[i, 7] = shape.elasticity
            rows[i, 8] = shape.friction
            rows[i, 9] = shape.sensor
            rows[i, 10] = shape.filter.group
            rows[i, 11] = shape.filter.categories
            rows[i, 12] = shape.filter.mask
        self._shape_data = rows
        self._shape_dirty = False
        return self._shape_data

    def step(self, dt: float) -> None:
        self.current_time_step = float(dt)
        damping = self.damping ** dt
        if self._custom_bodies:
            for body in self._bodies:
                body.position_func(body, dt)
                body.velocity_func(body, self.gravity, damping, dt)
                body.force = Vec2d.zero()
                body.torque = 0.0
        else:
            lib().mp_integrate(
                addr(self._state),
                len(self._state),
                self.gravity.x,
                self.gravity.y,
                damping,
                dt,
                parallel_context() if len(self._state) >= 131072 else 0,
            )
        if self._shapes:
            shapes = self._shape_state()
            lib().mp_collide(
                addr(self._state), len(self._state), addr(shapes), len(shapes), self.iterations
            )
        self._query_dirty = True

    def _ensure_circle_query_cache(self) -> None:
        if not self._query_dirty:
            return
        circles = [shape for shape in self._shapes if isinstance(shape, Circle)]
        self._noncircle_query_shapes = [
            shape for shape in self._shapes if not isinstance(shape, Circle)
        ]
        count = len(circles)
        data = np.empty((6, count), dtype=np.float64)
        if count:
            indices = np.fromiter(
                (shape.body._index for shape in circles), dtype=np.intp, count=count
            )
            offset_x = np.fromiter(
                (shape.offset.x for shape in circles), dtype=np.float64, count=count
            )
            offset_y = np.fromiter(
                (shape.offset.y for shape in circles), dtype=np.float64, count=count
            )
            angle = self._state[indices, 8]
            cosine, sine = np.cos(angle), np.sin(angle)
            data[0] = self._state[indices, 2] + offset_x * cosine - offset_y * sine
            data[1] = self._state[indices, 3] + offset_x * sine + offset_y * cosine
            data[2] = np.fromiter(
                (shape.radius for shape in circles), dtype=np.float64, count=count
            )
            data[3] = np.fromiter(
                (shape.filter.group for shape in circles), dtype=np.float64, count=count
            )
            data[4] = np.fromiter(
                (shape.filter.categories for shape in circles),
                dtype=np.float64,
                count=count,
            )
            data[5] = np.fromiter(
                (shape.filter.mask for shape in circles), dtype=np.float64, count=count
            )
        self._circle_query_nodes = None
        if count >= 64:
            nodes = []
            order = []

            def build_node(indices):
                node_index = len(nodes)
                centers_x, centers_y = data[0, indices], data[1, indices]
                radii = data[2, indices]
                nodes.append([
                    float(np.min(centers_x)),
                    float(np.min(centers_y)),
                    float(np.max(centers_x)),
                    float(np.max(centers_y)),
                    float(np.max(radii)),
                    -1,
                    -1,
                    0,
                    0,
                ])
                if len(indices) <= 8:
                    nodes[node_index][7] = len(order)
                    nodes[node_index][8] = len(indices)
                    order.extend(indices.tolist())
                    return node_index
                axis = 0 if np.ptp(centers_x) >= np.ptp(centers_y) else 1
                sorted_indices = indices[
                    np.argsort(data[axis, indices], kind="stable")
                ]
                middle = len(sorted_indices) // 2
                nodes[node_index][5] = build_node(sorted_indices[:middle])
                nodes[node_index][6] = build_node(sorted_indices[middle:])
                return node_index

            build_node(np.arange(count, dtype=np.intp))
            data = np.ascontiguousarray(data[:, order])
            circles = [circles[index] for index in order]
            self._circle_query_nodes = np.ascontiguousarray(
                np.asarray(nodes, dtype=np.float64).T
            )
        self._circle_query_shapes = circles
        self._circle_query_data = data
        self._query_dirty = False

    def point_query(self, point, max_distance: float, shape_filter: ShapeFilter):
        result = []
        for shape in self._shapes:
            if shape.filter.rejects_collision(shape_filter):
                continue
            info = shape.point_query(point)
            if info.distance <= max_distance:
                result.append(info)
        return result

    def point_query_nearest(self, point, max_distance: float, shape_filter: ShapeFilter):
        query = Vec2d(*point)
        self._ensure_circle_query_cache()
        best_shape = None
        best_values = None
        best_distance = float(max_distance)
        if self._circle_query_shapes:
            result = self._point_query_result
            if self._circle_query_nodes is not None:
                index = lib().mp_point_query_nearest_circle_bvh(
                    addr(self._circle_query_data),
                    len(self._circle_query_shapes),
                    addr(self._circle_query_nodes),
                    self._circle_query_nodes.shape[1],
                    query.x,
                    query.y,
                    best_distance,
                    shape_filter.group,
                    shape_filter.categories,
                    shape_filter.mask,
                    addr(result),
                )
            else:
                index = lib().mp_point_query_nearest_circle(
                    addr(self._circle_query_data),
                    len(self._circle_query_shapes),
                    query.x,
                    query.y,
                    best_distance,
                    shape_filter.group,
                    shape_filter.categories,
                    shape_filter.mask,
                    addr(result),
                )
            if index >= 0:
                best_shape = self._circle_query_shapes[index]
                best_values = tuple(result)
                best_distance = float(result[2])
        best = None
        if best_shape is not None:
            best = PointQueryInfo(
                best_shape,
                Vec2d(best_values[0], best_values[1]),
                best_distance,
                Vec2d(best_values[3], best_values[4]),
            )
        for shape in self._noncircle_query_shapes:
            if shape.filter.rejects_collision(shape_filter):
                continue
            info = shape.point_query(query)
            if info.distance <= max_distance and (
                best is None or info.distance < best.distance
            ):
                best = info
        return best

    def segment_query(self, start, end, radius: float, shape_filter: ShapeFilter):
        result = []
        for shape in self._shapes:
            if shape.filter.rejects_collision(shape_filter):
                continue
            info = shape.segment_query(start, end, radius)
            if info is not None:
                result.append(info)
        return sorted(result, key=lambda info: info.alpha)

    def segment_query_first(self, start, end, radius: float, shape_filter: ShapeFilter):
        result = self.segment_query(start, end, radius, shape_filter)
        return result[0] if result else None

    def bb_query(self, bb: BB, shape_filter: ShapeFilter):
        return [
            shape
            for shape in self._shapes
            if not shape.filter.rejects_collision(shape_filter) and shape.cache_bb().intersects(bb)
        ]

    def shape_query(self, shape: Shape):
        result = []
        for other in self._shapes:
            if other is shape or shape.filter.rejects_collision(other.filter):
                continue
            points = shape.shapes_collide(other)
            if points.points:
                result.append(ShapeQueryInfo(other, points))
        return result

    def reindex_shape(self, shape: Shape) -> None:
        self._query_dirty = True

    def reindex_shapes_for_body(self, body: Body) -> None:
        self._query_dirty = True

    def reindex_static(self) -> None:
        self._query_dirty = True

    def use_spatial_hash(self, dim: float, count: int) -> None:
        raise NotImplementedError("the covered solver uses direct pair generation")

    def on_collision(self, *args, **kwargs) -> None:
        raise NotImplementedError("collision callbacks are not in the covered subset")

    def add_post_step_callback(self, callback_function, key, *args, **kwargs) -> bool:
        callback_function(self, key, *args, **kwargs)
        return True
