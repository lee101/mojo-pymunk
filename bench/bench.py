"""Reproducible mojo-pymunk benchmarks against Pymunk 7.3."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojopymunk as mojo  # noqa: E402
import pymunk  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


CASES = []


def case(name):
    def register(builder):
        CASES.append((name, builder))
        return builder

    return register


@case("Space.step, 50k bodies")
def integration_case():
    rng = np.random.default_rng(1)
    positions = rng.normal(size=(50_000, 2))
    velocities = rng.normal(size=(50_000, 2))

    def make(module):
        space = module.Space()
        space.gravity = (0, -9.81)
        space.damping = 0.99
        bodies = [module.Body(1, 1) for _ in range(len(positions))]
        for body, position, velocity in zip(bodies, positions, velocities):
            body.position = tuple(position)
            body.velocity = tuple(velocity)
        space.add(*bodies)
        return space

    ours, theirs = make(mojo), make(pymunk)
    return lambda: ours.step(1 / 120), lambda: theirs.step(1 / 120)


@case("moments_for_circles, 1m items")
def batch_circle_case():
    rng = np.random.default_rng(2)
    count = 1_000_000
    mass = rng.uniform(0.1, 5, count)
    inner = rng.uniform(0, 1, count)
    outer = inner + rng.uniform(0, 3, count)
    offsets = rng.normal(size=(count, 2))

    def upstream():
        return np.fromiter(
            (
                pymunk.moment_for_circle(m, ri, ro, tuple(offset))
                for m, ri, ro, offset in zip(mass, inner, outer, offsets)
            ),
            dtype=np.float64,
            count=count,
        )

    return (
        lambda: mojo.moments_for_circles(mass, inner, outer, offsets),
        upstream,
    )


@case("moment_for_poly, 100k vertices")
def polygon_case():
    angles = np.linspace(0, 2 * np.pi, 100_000, endpoint=False)
    vertices_array = np.column_stack((np.cos(angles), np.sin(angles)))
    vertices_list = list(map(tuple, vertices_array))
    return (
        lambda: mojo.moment_for_poly(10, vertices_array),
        lambda: pymunk.moment_for_poly(10, vertices_list),
    )


@case("point_query_nearest, 20k circles")
def query_case():
    rng = np.random.default_rng(3)
    positions = rng.uniform(-1000, 1000, size=(20_000, 2))

    def make(module):
        space = module.Space()
        bodies = [module.Body(body_type=module.Body.STATIC) for _ in positions]
        shapes = [module.Circle(body, 2) for body in bodies]
        for body, position in zip(bodies, positions):
            body.position = tuple(position)
        space.add(*bodies, *shapes)
        return space

    ours, theirs = make(mojo), make(pymunk)
    our_filter, their_filter = mojo.ShapeFilter(), pymunk.ShapeFilter()
    return (
        lambda: ours.point_query_nearest((0, 0), 100, our_filter),
        lambda: theirs.point_query_nearest((0, 0), 100, their_filter),
    )


def main():
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(f"Python {platform.python_version()}, Pymunk {pymunk.version}")
    print()
    print("| case | mojo-pymunk | pymunk | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, builder in CASES:
        ours, theirs = builder()
        ours()
        theirs()
        our_time = timeit(ours)
        their_time = timeit(theirs)
        ratio = their_time / our_time
        result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {our_time * 1e3:.2f} ms | "
            f"{their_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
