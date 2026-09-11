"""Tier 1: the person's own expression basis.

A vendor's fifty-two named blendshapes are a coarse, symmetric vocabulary
decided in advance. The basis here is fitted to *this* face: principal
components of the pose-normalized mesh residual, so a component is whatever
set of vertices actually moves together on this person — including the
asymmetric, sub-threshold deformations that nobody named. That is what lets a
latent expression be represented at all, and it is also what makes a
478-point mesh affordable: forty-eight coefficients stand in for 1,434 values
per frame, with the reconstruction error measured rather than assumed.

``describe_component`` turns a component into words by geometry: which face
regions it displaces, and which way. No model is asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.anubis.utils.motion.landmarks import (
    FACE_POINT_COUNT,
    FACE_VALUES_PER_POINT,
    LandmarkSet,
)

DEFAULT_COMPONENT_COUNT = 48
_MIN_FIT_FRAMES = 8


@dataclass
class MotionBasis:
    """A fitted basis: mean residual, components, and how well they reconstruct."""

    basis_id: str | None
    landmark_set_version: str
    mean: np.ndarray  # [values]
    components: np.ndarray  # [component_count, values]
    explained_variance_ratio: list[float] = field(default_factory=list)
    reconstruction_error: float = 0.0
    fitted_frames: int = 0
    fitted_seconds: float = 0.0

    @property
    def component_count(self) -> int:
        """Return how many components the basis holds."""
        return int(self.components.shape[0])

    def encode(self, residuals: np.ndarray) -> np.ndarray:
        """Project ``[frames, values]`` residuals onto the basis → ``[frames, K]``."""
        array = np.asarray(residuals, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return (array - self.mean[None, :]) @ self.components.T

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """Rebuild ``[frames, values]`` residuals from ``[frames, K]`` coefficients."""
        array = np.asarray(coefficients, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array @ self.components + self.mean[None, :]

    def to_record(self) -> dict[str, Any]:
        """Return the storable record for this object."""
        return {
            "basis_id": self.basis_id,
            "landmark_set_version": self.landmark_set_version,
            "component_count": self.component_count,
            "mean": np.ascontiguousarray(self.mean, dtype=np.float32).tobytes(),
            "components": np.ascontiguousarray(self.components, dtype=np.float32).tobytes(),
            "explained_variance": list(self.explained_variance_ratio),
            "reconstruction_error": float(self.reconstruction_error),
            "fitted_frames": int(self.fitted_frames),
            "fitted_seconds": float(self.fitted_seconds),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> MotionBasis:
        """Rebuild the object from a stored record."""
        mean = np.frombuffer(record["mean"], dtype=np.float32).copy()
        count = int(record["component_count"])
        components = (
            np.frombuffer(record["components"], dtype=np.float32).copy().reshape(count, -1)
        )
        return cls(
            basis_id=(str(record["basis_id"]) if record.get("basis_id") else None),
            landmark_set_version=str(record["landmark_set_version"]),
            mean=mean,
            components=components,
            explained_variance_ratio=[float(value) for value in (record.get("explained_variance") or [])],
            reconstruction_error=float(record.get("reconstruction_error") or 0.0),
            fitted_frames=int(record.get("fitted_frames") or 0),
            fitted_seconds=float(record.get("fitted_seconds") or 0.0),
        )


def fit_basis(
    residuals: np.ndarray,
    *,
    landmark_set_version: str,
    component_count: int = DEFAULT_COMPONENT_COUNT,
    fitted_seconds: float = 0.0,
) -> MotionBasis:
    """Fit a basis to ``[frames, values]`` pose-normalized residuals.

    Uses ``IncrementalPCA`` so a long golden set is fitted in batches rather
    than held as one matrix; the component count is capped by what the data
    can support.
    """
    from sklearn.decomposition import IncrementalPCA

    array = np.asarray(residuals, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    frames, values = array.shape
    if frames < _MIN_FIT_FRAMES:
        raise ValueError(f"At least {_MIN_FIT_FRAMES} frames are needed to fit a basis.")
    count = max(1, min(int(component_count), frames - 1, values))
    batch = max(count * 2, min(frames, 512))
    model = IncrementalPCA(n_components=count, batch_size=batch)
    model.fit(array)
    components = np.asarray(model.components_, dtype=np.float32)
    mean = np.asarray(model.mean_, dtype=np.float32)
    basis = MotionBasis(
        basis_id=None,
        landmark_set_version=landmark_set_version,
        mean=mean,
        components=components,
        explained_variance_ratio=[float(value) for value in model.explained_variance_ratio_],
        fitted_frames=frames,
        fitted_seconds=float(fitted_seconds),
    )
    rebuilt = basis.decode(basis.encode(array))
    basis.reconstruction_error = float(np.sqrt(np.mean((rebuilt - array) ** 2)))
    return basis


def _direction_words(vector: np.ndarray) -> str:
    """Name the dominant direction of a 3-vector in image space, for the subject."""
    x, y, z = (float(value) for value in vector)
    magnitude = np.linalg.norm([x, y, z])
    if magnitude < 1e-9:
        return "in place"
    axis = int(np.argmax(np.abs([x, y, z])))
    if axis == 0:
        # The residual's x axis points toward the subject's own left.
        return "toward the person's own left" if x > 0 else "toward the person's own right"
    if axis == 1:
        # Image y grows downward.
        return "downward" if y > 0 else "upward"
    return "forward" if z < 0 else "back"


def describe_component(
    component: np.ndarray,
    landmark_set: LandmarkSet,
    *,
    region_limit: int = 3,
) -> str:
    """Say what one component moves, by region and direction.

    The mesh regions come from the landmark set (subject's own left and right).
    The description names the few regions that carry most of the component's
    displacement, largest first.
    """
    vector = np.asarray(component, dtype=np.float32).reshape(
        FACE_POINT_COUNT, FACE_VALUES_PER_POINT
    )
    regions = landmark_set.groups.get("face") or {}
    scored: list[tuple[float, str, np.ndarray]] = []
    for region, indices in regions.items():
        index_array = np.fromiter(indices, dtype=np.int64)
        index_array = index_array[index_array < FACE_POINT_COUNT]
        if index_array.size == 0:
            continue
        displacement = vector[index_array]
        mean_vector = displacement.mean(axis=0)
        magnitude = float(np.linalg.norm(displacement, axis=1).mean())
        scored.append((magnitude, region, mean_vector))
    if not scored:
        return "a movement across the face"
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[0][0]
    phrases: list[str] = []
    for magnitude, region, mean_vector in scored[:region_limit]:
        if magnitude < top * 0.35:
            break
        phrases.append(f"the {region.replace('_', ' ')} {_direction_words(mean_vector)}")
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]
