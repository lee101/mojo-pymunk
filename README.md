# mojo-pymunk

`mojo-pymunk` is a standalone Mojo implementation of useful, compute-heavy
parts of [Pymunk](https://www.pymunk.org/), exposed to Python with Pymunk's
names and signatures for the covered subset.

The strongest use case is bulk 2D mass-property and geometry work. The package
also includes bodies, circle/segment dynamics, shape queries, and a compact
rigid-body `Space` so the covered API can be used as an actual simulation
instead of as a collection of disconnected kernels.

```python
import mojopymunk as pymunk

space = pymunk.Space()
space.gravity = (0, -900)

body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 10))
body.position = (0, 100)
ball = pymunk.Circle(body, 10)
ball.elasticity = 0.8

floor = pymunk.Segment(None, (-100, 0), (100, 0), 2)
floor.elasticity = 0.8
floor.friction = 0.7
space.add(body, ball, floor)

for _ in range(240):
    space.step(1 / 120)

print(body.position, body.velocity)
```

Changing `import pymunk` to `import mojopymunk as pymunk` is sufficient for
code that stays inside the subset below.

## Coverage

| Pymunk area | Covered API |
| --- | --- |
| Mass properties | `moment_for_circle`, `moment_for_segment`, `moment_for_box`, `moment_for_poly`, `area_for_circle`, `area_for_segment`, `area_for_poly` |
| Bulk extension | `moments_for_circles` performs one FFI call for an array of circles |
| Math types | `Vec2d`, `BB`, `Transform` and their common arithmetic, geometry, and construction methods |
| Bodies | Dynamic, kinematic, and static `Body`; transforms; point velocity; forces and impulses; default and custom velocity/position callbacks |
| Shapes | `Circle`, `Segment`, and convex `Poly`; mass properties, bounding boxes, point queries, segment queries, and circle-circle/circle-segment contacts |
| World | `Space.add`, `remove`, `step`, gravity, damping, circle-circle and circle-segment response, filters, sensors, and spatial query methods |
| Geometry helpers | `poly_centroid`, `is_convex`, `convex_hull` |

The tests compare this API directly with Pymunk 7.3.0. They cover numerical
mass properties, geometry types, body updates, force and impulse application,
free-flight trajectories, contact velocity, shape filters, and point,
segment, bounding-box, and nearest-shape queries.

This is not a complete replacement for Chipmunk:

- Constraints and joints are not implemented.
- Polygon collision response is not implemented; polygons support geometry
  and queries.
- Collision callbacks, sleeping, Chipmunk-compatible thread controls, cached
  impulses, and Chipmunk's arbiter API are not implemented.
- The covered contact solver uses direct pair generation, so collision
  stepping is intended for small circle/segment scenes. It is not a replacement
  for Chipmunk's BBTree and mature iterative solver.
- Rounded polygon segment queries and shape-mass-driven automatic body mass
  accumulation are outside the covered subset.

Unsupported operations raise `NotImplementedError` instead of silently
returning an approximate result.

## Install and run

The repository pins the tested Mojo nightly and Pymunk reference version.

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` creates `dist/libmojo-pymunk.so`. Imports rebuild a missing or
stale library when a Mojo compiler is available. A packaged deployment can set
`MOJOPYMUNK_LIB=/absolute/path/to/libmojo-pymunk.so`.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14, and Pymunk 7.3.0:

| case | mojo-pymunk | pymunk | result |
| --- | ---: | ---: | ---: |
| `Space.step`, 50k bodies | 0.55 ms | 1.50 ms | 2.72x faster |
| `moments_for_circles`, 1m items | 3.72 ms | 1823.96 ms | 490.25x faster |
| `moment_for_poly`, 100k vertices | 0.60 ms | 9.02 ms | 15.14x faster |
| `point_query_nearest`, 20k circles | 0.02 ms | 0.01 ms | 1.61x slower |

These numbers include Python boundary costs. Bulk geometry wins because a
contiguous array crosses into one Mojo loop instead of making a Python/CFFI
conversion per item or vertex. Body stepping keeps callback bookkeeping out of
the hot path, skips trigonometry for the common zero-center-of-gravity case,
and uses chunked CPU parallelism only above 131,072 bodies. Nearest-circle
queries traverse a cached balanced BVH and scan its structure-of-arrays leaves
with native-width SIMD plus a scalar remainder.

No GPU path is included. The benchmarked integration, query, and mass-property
kernels are below roughly two floating-point operations per byte moved. The
collision loop has dependent writes to shared body state, so a light GPU path
would add transfer and synchronization overhead without an independent,
arithmetic-heavy kernel to amortize it.

## How it works

All compiled code lives in one compilation unit, `src/physics.mojo`. Exported
functions use `@export("name")` with the C ABI. Python loads the resulting
shared library with `ctypes`; NumPy buffers cross the boundary as integer
addresses and Mojo reconstructs `UnsafePointer[Float64,
AnyOrigin[mut=True]]` values inside each non-parametric export.

A world uses a persistent C-contiguous `float64` matrix with one 16-value row
per body:

```text
mass, moment, position x/y, velocity x/y, force x/y,
angle, angular velocity, torque, type, center of gravity x/y,
velocity limit, angular velocity limit
```

Each `Body` property reads and writes its row directly. `Space.step()` makes
one FFI call for all positions and velocities, then one call for the covered
contact solver when shapes exist. Python owns all numerical state and cached
query buffers, and the Mojo library retains no NumPy pointers. A CPU runtime
context is created lazily for very large parallel steps and released at Python
shutdown.

## License

MIT
