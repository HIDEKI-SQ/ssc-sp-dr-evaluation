"""Baseline 2D layout generators for synthetic experiments."""

from __future__ import annotations

import numpy as np


def grid_layout(n_side: int) -> np.ndarray:
    """n_side x n_side grid on [0, 1]^2 (N = n_side**2)."""
    x = np.linspace(0, 1, n_side)
    y = np.linspace(0, 1, n_side)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def line_layout(n: int) -> np.ndarray:
    """N points equally spaced on a line."""
    x = np.linspace(0, 1, n)
    return np.column_stack([x, np.zeros_like(x)])


def circle_layout(n: int) -> np.ndarray:
    """N points equally spaced on a unit circle (equal angular spacing)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def random_layout(n: int, rng: np.random.Generator) -> np.ndarray:
    """N points uniform on [0, 1]^2."""
    return rng.uniform(0, 1, size=(n, 2))


def make_layout(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "grid":
        side = int(round(np.sqrt(n)))
        if side * side != n:
            raise ValueError(f"grid requires square N; got N={n}")
        return grid_layout(side)
    if kind == "line":
        return line_layout(n)
    if kind == "circle":
        return circle_layout(n)
    if kind == "random":
        return random_layout(n, rng)
    raise ValueError(f"unknown layout kind: {kind}")
