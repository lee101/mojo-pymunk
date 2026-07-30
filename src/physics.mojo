"""Numerical kernels and the C ABI used by the Python package."""

from std.algorithm import parallelize
from std.gpu.host import DeviceContext
from std.math import cos, sin, sqrt
from std.memory import alloc, stack_allocation
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime ContextPtr = UnsafePointer[DeviceContext, AnyOrigin[mut=True]]
comptime STRIDE = 16


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def clamp(value: Float64, lo: Float64, hi: Float64) -> Float64:
    return min(max(value, lo), hi)


@export("mp_create_cpu_context")
def mp_create_cpu_context() abi("C") -> Int:
    try:
        var ctx = DeviceContext(api="cpu")
        var holder = alloc[DeviceContext](1)
        holder.unsafe_write(ctx)
        return Int(holder)
    except:
        return 0


@export("mp_destroy_cpu_context")
def mp_destroy_cpu_context(context_addr: Int) abi("C"):
    if context_addr == 0:
        return
    var holder = ContextPtr(unsafe_from_address=context_addr)
    holder.unsafe_deinit_pointee()
    holder.free()


@export("mp_moment_for_circle")
def mp_moment_for_circle(
    mass: Float64,
    inner_radius: Float64,
    outer_radius: Float64,
    offset_x: Float64,
    offset_y: Float64,
) abi("C") -> Float64:
    return mass * (
        0.5 * (inner_radius * inner_radius + outer_radius * outer_radius)
        + offset_x * offset_x
        + offset_y * offset_y
    )


@export("mp_area_for_circle")
def mp_area_for_circle(inner_radius: Float64, outer_radius: Float64) abi("C") -> Float64:
    return 3.141592653589793 * abs(
        inner_radius * inner_radius - outer_radius * outer_radius
    )


@export("mp_moment_for_segment")
def mp_moment_for_segment(
    mass: Float64,
    ax: Float64,
    ay: Float64,
    bx: Float64,
    by: Float64,
    radius: Float64,
) abi("C") -> Float64:
    var dx = bx - ax
    var dy = by - ay
    var length = sqrt(dx * dx + dy * dy) + 2.0 * radius
    var ox = 0.5 * (ax + bx)
    var oy = 0.5 * (ay + by)
    return mass * (
        (length * length + 4.0 * radius * radius) / 12.0 + ox * ox + oy * oy
    )


@export("mp_area_for_segment")
def mp_area_for_segment(
    ax: Float64, ay: Float64, bx: Float64, by: Float64, radius: Float64
) abi("C") -> Float64:
    var dx = bx - ax
    var dy = by - ay
    return radius * (
        3.141592653589793 * radius + 2.0 * sqrt(dx * dx + dy * dy)
    )


@export("mp_moment_for_box")
def mp_moment_for_box(mass: Float64, width: Float64, height: Float64) abi("C") -> Float64:
    return mass * (width * width + height * height) / 12.0


@export("mp_moment_for_poly")
def mp_moment_for_poly(
    mass: Float64,
    vertices_addr: Int,
    count: Int,
    offset_x: Float64,
    offset_y: Float64,
    radius: Float64,
) abi("C") -> Float64:
    var vertices = p(vertices_addr)
    if count == 2:
        var dx = vertices[2] - vertices[0]
        var dy = vertices[3] - vertices[1]
        var length = sqrt(dx * dx + dy * dy)
        var ox = 0.5 * (vertices[0] + vertices[2])
        var oy = 0.5 * (vertices[1] + vertices[3])
        return mass * (
            length * length / 12.0 + ox * ox + oy * oy
        )
    var sum1 = 0.0
    var sum2 = 0.0
    for i in range(count):
        var j = (i + 1) % count
        var x1 = vertices[2 * i] + offset_x
        var y1 = vertices[2 * i + 1] + offset_y
        var x2 = vertices[2 * j] + offset_x
        var y2 = vertices[2 * j + 1] + offset_y
        var a = x2 * y1 - y2 * x1
        var b = (
            x1 * x1 + y1 * y1 + x1 * x2 + y1 * y2 + x2 * x2 + y2 * y2
        )
        sum1 += a * b
        sum2 += a
    return mass * sum1 / (6.0 * sum2)


@export("mp_area_for_poly")
def mp_area_for_poly(vertices_addr: Int, count: Int, radius: Float64) abi("C") -> Float64:
    var vertices = p(vertices_addr)
    var area = 0.0
    var perimeter = 0.0
    for i in range(count):
        var j = (i + 1) % count
        var x1 = vertices[2 * i]
        var y1 = vertices[2 * i + 1]
        var x2 = vertices[2 * j]
        var y2 = vertices[2 * j + 1]
        area += x1 * y2 - y1 * x2
        var dx = x2 - x1
        var dy = y2 - y1
        perimeter += sqrt(dx * dx + dy * dy)
    return radius * (
        3.141592653589793 * abs(radius) + perimeter
    ) + 0.5 * area


@export("mp_centroid_for_poly")
def mp_centroid_for_poly(vertices_addr: Int, count: Int, result_addr: Int) abi("C"):
    var vertices = p(vertices_addr)
    var result = p(result_addr)
    var area_sum = 0.0
    var x_sum = 0.0
    var y_sum = 0.0
    for i in range(count):
        var j = (i + 1) % count
        var x1 = vertices[2 * i]
        var y1 = vertices[2 * i + 1]
        var x2 = vertices[2 * j]
        var y2 = vertices[2 * j + 1]
        var cross = x1 * y2 - y1 * x2
        area_sum += cross
        x_sum += (x1 + x2) * cross
        y_sum += (y1 + y2) * cross
    result[0] = x_sum / (3.0 * area_sum)
    result[1] = y_sum / (3.0 * area_sum)


@export("mp_batch_moment_for_circle")
def mp_batch_moment_for_circle(
    mass_addr: Int,
    inner_addr: Int,
    outer_addr: Int,
    offset_addr: Int,
    result_addr: Int,
    count: Int,
) abi("C"):
    var mass = p(mass_addr)
    var inner = p(inner_addr)
    var outer = p(outer_addr)
    var offset = p(offset_addr)
    var result = p(result_addr)
    for i in range(count):
        var ox = offset[2 * i]
        var oy = offset[2 * i + 1]
        result[i] = mass[i] * (
            0.5 * (inner[i] * inner[i] + outer[i] * outer[i])
            + ox * ox
            + oy * oy
        )


@export("mp_point_query_nearest_circle")
def mp_point_query_nearest_circle(
    data_addr: Int,
    count: Int,
    query_x: Float64,
    query_y: Float64,
    max_distance: Float64,
    query_group: Int,
    query_categories: Int,
    query_mask: Int,
    result_addr: Int,
) abi("C") -> Int:
    var data = p(data_addr)
    var result = p(result_addr)
    comptime W = simd_width_of[DType.float64]()
    var best_index = -1
    var best_distance = max_distance
    var i = 0
    while i + W <= count:
        var dx = SIMD[DType.float64, W](query_x) - data.load[width=W](i)
        var dy = SIMD[DType.float64, W](query_y) - data.load[width=W](count + i)
        var radii = data.load[width=W](2 * count + i)
        var distances = sqrt(dx * dx + dy * dy) - radii
        for lane in range(W):
            var index = i + lane
            var group = Int(data[3 * count + index])
            var categories = Int(data[4 * count + index])
            var mask = Int(data[5 * count + index])
            var rejected = (
                (query_group != 0 and query_group == group)
                or (categories & query_mask) == 0
                or (query_categories & mask) == 0
            )
            var distance = distances[lane]
            if (
                not rejected
                and distance <= max_distance
                and (best_index < 0 or distance < best_distance)
            ):
                best_index = index
                best_distance = distance
        i += W
    while i < count:
        var group = Int(data[3 * count + i])
        var categories = Int(data[4 * count + i])
        var mask = Int(data[5 * count + i])
        var rejected = (
            (query_group != 0 and query_group == group)
            or (categories & query_mask) == 0
            or (query_categories & mask) == 0
        )
        var dx = query_x - data[i]
        var dy = query_y - data[count + i]
        var distance = sqrt(dx * dx + dy * dy) - data[2 * count + i]
        if (
            not rejected
            and distance <= max_distance
            and (best_index < 0 or distance < best_distance)
        ):
            best_index = i
            best_distance = distance
        i += 1
    if best_index < 0:
        return -1
    var center_x = data[best_index]
    var center_y = data[count + best_index]
    var radius = data[2 * count + best_index]
    var best_dx = query_x - center_x
    var best_dy = query_y - center_y
    var center_distance = sqrt(best_dx * best_dx + best_dy * best_dy)
    var gradient_x = 0.0
    var gradient_y = 1.0
    var point_x = center_x
    var point_y = center_y
    if center_distance > 0.0:
        gradient_x = best_dx / center_distance
        gradient_y = best_dy / center_distance
        point_x += gradient_x * radius
        point_y += gradient_y * radius
    result.store(
        0,
        SIMD[DType.float64, 5](
            point_x, point_y, best_distance, gradient_x, gradient_y
        ),
    )
    return best_index


@export("mp_point_query_nearest_circle_bvh")
def mp_point_query_nearest_circle_bvh(
    data_addr: Int,
    count: Int,
    nodes_addr: Int,
    node_count: Int,
    query_x: Float64,
    query_y: Float64,
    max_distance: Float64,
    query_group: Int,
    query_categories: Int,
    query_mask: Int,
    result_addr: Int,
) abi("C") -> Int:
    var data = p(data_addr)
    var nodes = p(nodes_addr)
    var result = p(result_addr)
    comptime W = simd_width_of[DType.float64]()
    comptime STACK_SIZE = 128
    var stack = stack_allocation[STACK_SIZE, Int]()
    var stack_size = 1
    stack[0] = 0
    var best_index = -1
    var best_distance = max_distance
    while stack_size > 0:
        stack_size -= 1
        var node = stack[stack_size]
        var node_dx = max(
            max(nodes[node] - query_x, 0.0),
            query_x - nodes[2 * node_count + node],
        )
        var node_dy = max(
            max(nodes[node_count + node] - query_y, 0.0),
            query_y - nodes[3 * node_count + node],
        )
        var lower_bound = (
            sqrt(node_dx * node_dx + node_dy * node_dy)
            - nodes[4 * node_count + node]
        )
        if lower_bound > best_distance:
            continue
        var left = Int(nodes[5 * node_count + node])
        var right = Int(nodes[6 * node_count + node])
        if left >= 0:
            if stack_size + 2 > STACK_SIZE:
                return mp_point_query_nearest_circle(
                    data_addr,
                    count,
                    query_x,
                    query_y,
                    max_distance,
                    query_group,
                    query_categories,
                    query_mask,
                    result_addr,
                )
            stack[stack_size] = right
            stack[stack_size + 1] = left
            stack_size += 2
            continue
        var start = Int(nodes[7 * node_count + node])
        var end = start + Int(nodes[8 * node_count + node])
        var i = start
        while i + W <= end:
            var circle_dx = (
                SIMD[DType.float64, W](query_x) - data.load[width=W](i)
            )
            var circle_dy = (
                SIMD[DType.float64, W](query_y)
                - data.load[width=W](count + i)
            )
            var radii = data.load[width=W](2 * count + i)
            var distances = sqrt(circle_dx * circle_dx + circle_dy * circle_dy) - radii
            for lane in range(W):
                var index = i + lane
                var group = Int(data[3 * count + index])
                var categories = Int(data[4 * count + index])
                var mask = Int(data[5 * count + index])
                var rejected = (
                    (query_group != 0 and query_group == group)
                    or (categories & query_mask) == 0
                    or (query_categories & mask) == 0
                )
                var distance = distances[lane]
                if (
                    not rejected
                    and distance <= max_distance
                    and (best_index < 0 or distance < best_distance)
                ):
                    best_index = index
                    best_distance = distance
            i += W
        while i < end:
            var group = Int(data[3 * count + i])
            var categories = Int(data[4 * count + i])
            var mask = Int(data[5 * count + i])
            var rejected = (
                (query_group != 0 and query_group == group)
                or (categories & query_mask) == 0
                or (query_categories & mask) == 0
            )
            var circle_dx = query_x - data[i]
            var circle_dy = query_y - data[count + i]
            var distance = (
                sqrt(circle_dx * circle_dx + circle_dy * circle_dy)
                - data[2 * count + i]
            )
            if (
                not rejected
                and distance <= max_distance
                and (best_index < 0 or distance < best_distance)
            ):
                best_index = i
                best_distance = distance
            i += 1
    if best_index < 0:
        return -1
    var center_x = data[best_index]
    var center_y = data[count + best_index]
    var radius = data[2 * count + best_index]
    var best_dx = query_x - center_x
    var best_dy = query_y - center_y
    var center_distance = sqrt(best_dx * best_dx + best_dy * best_dy)
    var gradient_x = 0.0
    var gradient_y = 1.0
    var point_x = center_x
    var point_y = center_y
    if center_distance > 0.0:
        gradient_x = best_dx / center_distance
        gradient_y = best_dy / center_distance
        point_x += gradient_x * radius
        point_y += gradient_y * radius
    result.store(
        0,
        SIMD[DType.float64, 5](
            point_x, point_y, best_distance, gradient_x, gradient_y
        ),
    )
    return best_index


def integrate_range(
    state: Ptr,
    start: Int,
    end: Int,
    gravity_x: Float64,
    gravity_y: Float64,
    damping: Float64,
    dt: Float64,
):
    for i in range(start, end):
        var base = i * STRIDE
        var body_type = Int(state[base + 11])
        if body_type == 2:
            continue
        var old_angle = state[base + 8]
        var cog_x = state[base + 12]
        var cog_y = state[base + 13]
        state[base + 8] += state[base + 9] * dt
        var new_angle = state[base + 8]
        if cog_x == 0.0 and cog_y == 0.0:
            state[base + 2] += state[base + 4] * dt
            state[base + 3] += state[base + 5] * dt
        else:
            var old_cos = cos(old_angle)
            var old_sin = sin(old_angle)
            var world_cog_x = (
                state[base + 2] + cog_x * old_cos - cog_y * old_sin
            )
            var world_cog_y = (
                state[base + 3] + cog_x * old_sin + cog_y * old_cos
            )
            var new_cos = cos(new_angle)
            var new_sin = sin(new_angle)
            state[base + 2] = (
                world_cog_x + state[base + 4] * dt
                - cog_x * new_cos + cog_y * new_sin
            )
            state[base + 3] = (
                world_cog_y + state[base + 5] * dt
                - cog_x * new_sin - cog_y * new_cos
            )
        if body_type == 0:
            var mass = state[base]
            var moment = state[base + 1]
            if mass > 0.0:
                state[base + 4] = (
                    state[base + 4] * damping
                    + (gravity_x + state[base + 6] / mass) * dt
                )
                state[base + 5] = (
                    state[base + 5] * damping
                    + (gravity_y + state[base + 7] / mass) * dt
                )
            if moment > 0.0:
                state[base + 9] = (
                    state[base + 9] * damping
                    + state[base + 10] / moment * dt
                )
            var speed = sqrt(
                state[base + 4] * state[base + 4]
                + state[base + 5] * state[base + 5]
            )
            var speed_limit = state[base + 14]
            if speed > speed_limit:
                var scale = speed_limit / speed
                state[base + 4] *= scale
                state[base + 5] *= scale
            state[base + 9] = clamp(
                state[base + 9], -state[base + 15], state[base + 15]
            )
        state[base + 6] = 0.0
        state[base + 7] = 0.0
        state[base + 10] = 0.0


@export("mp_integrate")
def mp_integrate(
    state_addr: Int,
    count: Int,
    gravity_x: Float64,
    gravity_y: Float64,
    damping: Float64,
    dt: Float64,
    context_addr: Int,
) abi("C"):
    var state = p(state_addr)
    comptime PARALLEL_THRESHOLD = 131072
    comptime CHUNK_SIZE = 2048
    if count < PARALLEL_THRESHOLD or context_addr == 0:
        integrate_range(
            state, 0, count, gravity_x, gravity_y, damping, dt
        )
        return
    var chunk_count = (count + CHUNK_SIZE - 1) // CHUNK_SIZE

    @parameter
    def integrate_chunk(chunk: Int):
        var start = chunk * CHUNK_SIZE
        var end = min(start + CHUNK_SIZE, count)
        integrate_range(
            state, start, end, gravity_x, gravity_y, damping, dt
        )

    var context = ContextPtr(unsafe_from_address=context_addr)
    parallelize[integrate_chunk](chunk_count, context[])


def body_point(
    state: Ptr, body_index: Int, local_x: Float64, local_y: Float64
) -> SIMD[DType.float64, 2]:
    var base = body_index * STRIDE
    var angle = state[base + 8]
    var c = cos(angle)
    var s = sin(angle)
    return SIMD[DType.float64, 2](
        state[base + 2] + local_x * c - local_y * s,
        state[base + 3] + local_x * s + local_y * c,
    )


def inverse_mass(state: Ptr, body_index: Int) -> Float64:
    var base = body_index * STRIDE
    if Int(state[base + 11]) != 0 or state[base] <= 0.0:
        return 0.0
    return 1.0 / state[base]


def resolve_contact(
    state: Ptr,
    body_a: Int,
    body_b: Int,
    nx: Float64,
    ny: Float64,
    penetration: Float64,
    elasticity: Float64,
    friction: Float64,
):
    var inv_a = inverse_mass(state, body_a)
    var inv_b = inverse_mass(state, body_b)
    var inv_sum = inv_a + inv_b
    if inv_sum == 0.0:
        return
    var base_a = body_a * STRIDE
    var base_b = body_b * STRIDE
    var correction = 0.8 * max(penetration - 0.001, 0.0) / inv_sum
    state[base_a + 2] -= nx * correction * inv_a
    state[base_a + 3] -= ny * correction * inv_a
    state[base_b + 2] += nx * correction * inv_b
    state[base_b + 3] += ny * correction * inv_b
    var rvx = state[base_b + 4] - state[base_a + 4]
    var rvy = state[base_b + 5] - state[base_a + 5]
    var normal_speed = rvx * nx + rvy * ny
    if normal_speed >= 0.0:
        return
    var impulse = -(1.0 + elasticity) * normal_speed / inv_sum
    var ix = impulse * nx
    var iy = impulse * ny
    state[base_a + 4] -= ix * inv_a
    state[base_a + 5] -= iy * inv_a
    state[base_b + 4] += ix * inv_b
    state[base_b + 5] += iy * inv_b
    rvx = state[base_b + 4] - state[base_a + 4]
    rvy = state[base_b + 5] - state[base_a + 5]
    var tangent_speed = rvx * (-ny) + rvy * nx
    var tangent_impulse = -tangent_speed / inv_sum
    tangent_impulse = clamp(tangent_impulse, -friction * impulse, friction * impulse)
    var tx = tangent_impulse * (-ny)
    var ty = tangent_impulse * nx
    state[base_a + 4] -= tx * inv_a
    state[base_a + 5] -= ty * inv_a
    state[base_b + 4] += tx * inv_b
    state[base_b + 5] += ty * inv_b


def filters_reject(shapes: Ptr, a: Int, b: Int) -> Bool:
    var aa = a * STRIDE
    var bb = b * STRIDE
    var group_a = Int(shapes[aa + 10])
    var group_b = Int(shapes[bb + 10])
    if group_a != 0 and group_a == group_b:
        return True
    var categories_a = UInt32(shapes[aa + 11])
    var categories_b = UInt32(shapes[bb + 11])
    var mask_a = UInt32(shapes[aa + 12])
    var mask_b = UInt32(shapes[bb + 12])
    return (categories_a & mask_b) == 0 or (categories_b & mask_a) == 0


@export("mp_collide")
def mp_collide(
    state_addr: Int,
    body_count: Int,
    shape_addr: Int,
    shape_count: Int,
    iterations: Int,
) abi("C"):
    var state = p(state_addr)
    var shapes = p(shape_addr)
    for iteration in range(iterations):
        for i in range(shape_count):
            var ib = i * STRIDE
            if Int(shapes[ib]) != 0:
                continue
            for j in range(i + 1, shape_count):
                var jb = j * STRIDE
                if filters_reject(shapes, i, j):
                    continue
                var type_j = Int(shapes[jb])
                if type_j != 0 and type_j != 1:
                    continue
                var body_i = Int(shapes[ib + 1])
                var body_j = Int(shapes[jb + 1])
                if body_i == body_j:
                    continue
                var ci = body_point(
                    state, body_i, shapes[ib + 2], shapes[ib + 3]
                )
                var radius_i = shapes[ib + 6]
                var nx = 0.0
                var ny = 1.0
                var distance = 0.0
                var penetration = 0.0
                if type_j == 0:
                    var cj = body_point(
                        state, body_j, shapes[jb + 2], shapes[jb + 3]
                    )
                    var dx = cj[0] - ci[0]
                    var dy = cj[1] - ci[1]
                    distance = sqrt(dx * dx + dy * dy)
                    var radius_sum = radius_i + shapes[jb + 6]
                    if distance >= radius_sum:
                        continue
                    if distance > 1.0e-12:
                        nx = dx / distance
                        ny = dy / distance
                    penetration = radius_sum - distance
                else:
                    var a = body_point(
                        state, body_j, shapes[jb + 2], shapes[jb + 3]
                    )
                    var b = body_point(
                        state, body_j, shapes[jb + 4], shapes[jb + 5]
                    )
                    var abx = b[0] - a[0]
                    var aby = b[1] - a[1]
                    var denom = abx * abx + aby * aby
                    var t = 0.0
                    if denom > 0.0:
                        t = clamp(
                            ((ci[0] - a[0]) * abx + (ci[1] - a[1]) * aby) / denom,
                            0.0,
                            1.0,
                        )
                    var qx = a[0] + t * abx
                    var qy = a[1] + t * aby
                    var dx = ci[0] - qx
                    var dy = ci[1] - qy
                    distance = sqrt(dx * dx + dy * dy)
                    var radius_sum = radius_i + shapes[jb + 6]
                    if distance >= radius_sum:
                        continue
                    if distance > 1.0e-12:
                        nx = -dx / distance
                        ny = -dy / distance
                    penetration = radius_sum - distance
                if shapes[ib + 9] != 0.0 or shapes[jb + 9] != 0.0:
                    continue
                resolve_contact(
                    state,
                    body_i,
                    body_j,
                    nx,
                    ny,
                    penetration,
                    shapes[ib + 7] * shapes[jb + 7],
                    shapes[ib + 8] * shapes[jb + 8],
                )
