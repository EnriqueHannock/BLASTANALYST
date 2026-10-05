"""FragAnalyst - digital rock fragmentation analysis from blast photographs.

A single-file Streamlit app: upload blast or muckpile photographs, calibrate
scale against a reference object, segment particles, review and correct the
result, and get particle-size distribution statistics, oversize and fines,
a Rosin-Rammler fit, blast comparisons, and CSV/Excel/PDF export.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Everything (analysis code, storage, and the UI) lives in this one file by
design, so it is easy to read top to bottom and easy to deploy as a single
file on Streamlit Community Cloud.
"""

from __future__ import annotations

import io
import json
import math
import os
import sqlite3
import statistics
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image as PILImage
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
from streamlit_image_coordinates import streamlit_image_coordinates



# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
DEFAULT_COV_WARNING_THRESHOLD = 0.05


class CalibrationMethod(str, Enum):
    TWO_POINT_REFERENCE = "two_point_reference"
    MANUAL_SCALE_ENTRY = "manual_scale_entry"


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class ReferenceMeasurement:
    point_a: Point2D
    point_b: Point2D
    known_length_mm: float
    label: str = ""

    def pixel_length(self) -> float:
        dx = self.point_b.x - self.point_a.x
        dy = self.point_b.y - self.point_a.y
        return math.hypot(dx, dy)

    def scale_mm_per_px(self) -> float:
        pixel_length = self.pixel_length()
        if pixel_length <= 0:
            raise CalibrationError("Reference points are coincident; can't compute a scale from a zero-length line.")
        if self.known_length_mm <= 0:
            raise CalibrationError("known_length_mm must be greater than zero.")
        return self.known_length_mm / pixel_length


@dataclass
class CalibrationResult:
    method: CalibrationMethod
    scale_mm_per_px: float
    is_valid: bool
    measurements: list[ReferenceMeasurement] = field(default_factory=list)
    scale_uncertainty_mm_per_px: float | None = None
    coefficient_of_variation: float | None = None
    consistency_warning: str | None = None

    def to_dict(self) -> dict:
        return {
            "method": self.method.value,
            "scale_mm_per_px": self.scale_mm_per_px,
            "is_valid": self.is_valid,
            "num_measurements": len(self.measurements),
            "scale_uncertainty_mm_per_px": self.scale_uncertainty_mm_per_px,
            "coefficient_of_variation": self.coefficient_of_variation,
            "consistency_warning": self.consistency_warning,
        }


def calibrate_from_reference_points(
    measurements: list[ReferenceMeasurement],
    cov_warning_threshold: float = DEFAULT_COV_WARNING_THRESHOLD,
) -> CalibrationResult:
    if not measurements:
        raise CalibrationError("At least one reference measurement is required.")

    per_measurement_scale = [m.scale_mm_per_px() for m in measurements]
    mean_scale = statistics.fmean(per_measurement_scale)

    uncertainty = None
    cov = None
    warning = None
    if len(per_measurement_scale) > 1:
        uncertainty = statistics.stdev(per_measurement_scale)
        cov = uncertainty / mean_scale if mean_scale else None
        if cov is not None and cov > cov_warning_threshold:
            warning = (
                f"Reference measurements disagree by {cov:.1%} (coefficient of "
                f"variation), above the {cov_warning_threshold:.0%} threshold. "
                "Re-measure before trusting downstream sizes."
            )

    return CalibrationResult(
        method=CalibrationMethod.TWO_POINT_REFERENCE,
        scale_mm_per_px=mean_scale,
        is_valid=True,
        measurements=measurements,
        scale_uncertainty_mm_per_px=uncertainty,
        coefficient_of_variation=cov,
        consistency_warning=warning,
    )


def calibrate_from_manual_entry(
    *,
    mm_per_pixel: float | None = None,
    pixels_per_mm: float | None = None,
) -> CalibrationResult:
    if (mm_per_pixel is None) == (pixels_per_mm is None):
        raise CalibrationError("Provide exactly one of mm_per_pixel or pixels_per_mm.")

    if mm_per_pixel is not None:
        if mm_per_pixel <= 0:
            raise CalibrationError("mm_per_pixel must be greater than zero.")
        scale = mm_per_pixel
    else:
        if pixels_per_mm <= 0:  # type: ignore[operator]
            raise CalibrationError("pixels_per_mm must be greater than zero.")
        scale = 1.0 / pixels_per_mm  # type: ignore[operator]

    return CalibrationResult(
        method=CalibrationMethod.MANUAL_SCALE_ENTRY,
        scale_mm_per_px=scale,
        is_valid=True,
    )


def pixels_to_mm(value_px: float, calibration: CalibrationResult) -> float:
    if not calibration.is_valid:
        raise CalibrationError("Calibration is not valid; can't convert to physical units.")
    return value_px * calibration.scale_mm_per_px


def area_px_to_mm2(value_px2: float, calibration: CalibrationResult) -> float:
    if not calibration.is_valid:
        raise CalibrationError("Calibration is not valid; can't convert to physical units.")
    return value_px2 * (calibration.scale_mm_per_px ** 2)


# ---------------------------------------------------------------------------
# Image quality checks
# ---------------------------------------------------------------------------
@dataclass
class QualityThresholds:
    min_width_px: int = 800
    min_height_px: int = 600
    blur_variance_min: float = 100.0
    clipped_pixel_fraction_max: float = 0.02


@dataclass
class ImageQualityReport:
    warnings: list[str]
    blur_variance: float
    clipped_fraction: float
    width_px: int
    height_px: int


def assess_image_quality(
    image_bgr: np.ndarray,
    thresholds: QualityThresholds | None = None,
) -> ImageQualityReport:
    thresholds = thresholds or QualityThresholds()
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= 1) | (gray >= 254)))

    warnings: list[str] = []
    if width < thresholds.min_width_px or height < thresholds.min_height_px:
        warnings.append(
            f"Resolution {width}x{height} is low - fine particle sizes may not be reliable."
        )
    if blur_variance < thresholds.blur_variance_min:
        warnings.append(
            f"Image looks blurry (focus score {blur_variance:.1f}). Segmentation boundaries may be unreliable."
        )
    if clipped > thresholds.clipped_pixel_fraction_max:
        warnings.append(
            f"{clipped:.1%} of pixels are over/under-exposed. Consider reshooting under more even light."
        )

    return ImageQualityReport(
        warnings=warnings,
        blur_variance=blur_variance,
        clipped_fraction=clipped,
        width_px=width,
        height_px=height,
    )


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
class ThresholdMode(str, Enum):
    OTSU = "otsu"
    ADAPTIVE = "adaptive"


@dataclass
class SegmentationConfig:
    threshold_mode: ThresholdMode = ThresholdMode.OTSU
    adaptive_block_size: int = 51
    adaptive_c: int = 2
    blur_kernel: int = 5
    morph_kernel: int = 3
    morph_open_iterations: int = 2
    min_distance_between_peaks: int = 15
    invert: bool = True  # rocks are usually darker/lighter than gaps depending on lighting


def _threshold(gray: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (config.blur_kernel, config.blur_kernel), 0)
    flag = cv2.THRESH_BINARY_INV if config.invert else cv2.THRESH_BINARY

    if config.threshold_mode is ThresholdMode.OTSU:
        _, thresh = cv2.threshold(blurred, 0, 255, flag + cv2.THRESH_OTSU)
    else:
        adaptive_flag = cv2.THRESH_BINARY_INV if config.invert else cv2.THRESH_BINARY
        block_size = config.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, adaptive_flag, block_size, config.adaptive_c
        )
    return thresh


def segment(image_bgr: np.ndarray, config: SegmentationConfig | None = None) -> np.ndarray:
    """Run marker-controlled watershed segmentation on a BGR image.

    Returns a labeled image the same height/width as the input, with 0 for
    background and a unique positive integer per detected particle.
    """
    config = config or SegmentationConfig()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    binary = _threshold(gray, config)

    kernel = np.ones((config.morph_kernel, config.morph_kernel), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iterations)
    opened_bool = opened > 0

    distance = ndi.distance_transform_edt(opened_bool)

    coords = peak_local_max(
        distance,
        min_distance=config.min_distance_between_peaks,
        labels=opened_bool,
    )
    markers = np.zeros(distance.shape, dtype=bool)
    if len(coords) > 0:
        markers[tuple(coords.T)] = True
    marker_labels, _ = ndi.label(markers)

    labels = watershed(-distance, marker_labels, mask=opened_bool)
    return labels.astype(np.int32)


def label_confidences(labels: np.ndarray) -> dict[int, float]:
    """Classical watershed has no learned confidence score, so every
    detected particle gets 1.0. This keeps the interface consistent with
    a future ML-based segmenter, which would return real per-mask scores.
    """
    unique = [int(v) for v in np.unique(labels) if v != 0]
    return {label_id: 1.0 for label_id in unique}


# ---------------------------------------------------------------------------
# Particle measurement
# ---------------------------------------------------------------------------
class ParticleFlag(str, Enum):
    TOUCHES_BORDER = "touches_border"
    LOW_CONFIDENCE = "low_confidence"
    SIZE_OUTLIER = "size_outlier"
    UNUSUAL_ASPECT_RATIO = "unusual_aspect_ratio"
    OVERLAP_SUSPECTED = "overlap_suspected"


class ParticleProvenance(str, Enum):
    AUTO_DETECTED = "auto_detected"
    MANUAL_ADDED = "manual_added"


@dataclass
class ParticleMeasurement:
    particle_id: int
    area_px: float
    perimeter_px: float
    centroid_xy: tuple[float, float]
    bbox: tuple[int, int, int, int]
    width_px: float
    height_px: float
    aspect_ratio: float
    circularity: float
    equiv_diameter_px: float
    feret_diameter_px: float | None
    orientation_deg: float
    confidence: float
    provenance: ParticleProvenance
    flags: list[ParticleFlag] = field(default_factory=list)
    kept: bool = True


@dataclass
class QCThresholds:
    min_area_px: float = 16.0
    max_area_fraction_of_image: float = 0.5
    max_aspect_ratio: float = 6.0
    min_confidence: float = 0.5
    border_margin_px: int = 1


def equivalent_circular_diameter(area_px: float) -> float:
    if area_px < 0:
        raise ValueError("area_px must be >= 0")
    return math.sqrt(4.0 * area_px / math.pi)


def circularity(area_px: float, perimeter_px: float) -> float:
    if perimeter_px <= 0:
        return 0.0
    return (4.0 * math.pi * area_px) / (perimeter_px ** 2)


def measure_particles_from_labels(
    label_image: np.ndarray,
    confidences: dict[int, float] | None = None,
    qc: QCThresholds | None = None,
) -> list[ParticleMeasurement]:
    qc = qc or QCThresholds()
    confidences = confidences or {}
    image_area = label_image.shape[0] * label_image.shape[1]
    max_row, max_col = label_image.shape[0] - 1, label_image.shape[1] - 1

    results: list[ParticleMeasurement] = []
    for prop in regionprops(label_image):
        area = float(prop.area)
        perimeter = float(prop.perimeter)
        min_row, min_col, max_r, max_c = prop.bbox
        width = float(max_c - min_col)
        height = float(max_r - min_row)
        major = float(prop.axis_major_length) or 1e-9
        minor = float(prop.axis_minor_length) or 1e-9
        aspect = major / minor if minor else float("inf")

        try:
            feret = float(prop.feret_diameter_max)
        except (AttributeError, NotImplementedError):
            feret = None

        confidence = float(confidences.get(prop.label, 1.0))

        flags: list[ParticleFlag] = []
        touches_border = (
            min_row <= qc.border_margin_px
            or min_col <= qc.border_margin_px
            or max_r >= max_row - qc.border_margin_px
            or max_c >= max_col - qc.border_margin_px
        )
        if touches_border:
            flags.append(ParticleFlag.TOUCHES_BORDER)
        if area < qc.min_area_px or area > qc.max_area_fraction_of_image * image_area:
            flags.append(ParticleFlag.SIZE_OUTLIER)
        if aspect > qc.max_aspect_ratio:
            flags.append(ParticleFlag.UNUSUAL_ASPECT_RATIO)
        if confidence < qc.min_confidence:
            flags.append(ParticleFlag.LOW_CONFIDENCE)
        solidity = float(prop.solidity) if prop.solidity else 1.0
        if solidity < 0.75:
            flags.append(ParticleFlag.OVERLAP_SUSPECTED)

        results.append(
            ParticleMeasurement(
                particle_id=int(prop.label),
                area_px=area,
                perimeter_px=perimeter,
                centroid_xy=(float(prop.centroid[1]), float(prop.centroid[0])),
                bbox=(int(min_row), int(min_col), int(max_r), int(max_c)),
                width_px=width,
                height_px=height,
                aspect_ratio=aspect,
                circularity=circularity(area, perimeter),
                equiv_diameter_px=equivalent_circular_diameter(area),
                feret_diameter_px=feret,
                orientation_deg=math.degrees(float(prop.orientation)),
                confidence=confidence,
                provenance=ParticleProvenance.AUTO_DETECTED,
                flags=flags,
            )
        )
    return results


def add_manual_particle(
    particles: list[ParticleMeasurement],
    center_xy: tuple[float, float],
    radius_px: float,
) -> ParticleMeasurement:
    """Add a particle as an approximate circle. Manual boundary tracing is
    out of scope here - this is enough to record a rock the automatic
    segmentation missed entirely, at a size the user estimates by eye.
    """
    next_id = max([p.particle_id for p in particles], default=0) + 1
    area = math.pi * radius_px ** 2
    perimeter = 2 * math.pi * radius_px
    particle = ParticleMeasurement(
        particle_id=next_id,
        area_px=area,
        perimeter_px=perimeter,
        centroid_xy=center_xy,
        bbox=(
            int(center_xy[1] - radius_px),
            int(center_xy[0] - radius_px),
            int(center_xy[1] + radius_px),
            int(center_xy[0] + radius_px),
        ),
        width_px=radius_px * 2,
        height_px=radius_px * 2,
        aspect_ratio=1.0,
        circularity=1.0,
        equiv_diameter_px=radius_px * 2,
        feret_diameter_px=radius_px * 2,
        orientation_deg=0.0,
        confidence=1.0,
        provenance=ParticleProvenance.MANUAL_ADDED,
        flags=[],
    )
    particles.append(particle)
    return particle


def merge_particles(
    particles: list[ParticleMeasurement],
    ids_to_merge: list[int],
) -> ParticleMeasurement:
    """Combine two or more particles into one, summing area and recomputing
    the derived quantities from the combined bounding region. Used to fix a
    single rock that got split into pieces by segmentation.
    """
    to_merge = [p for p in particles if p.particle_id in ids_to_merge]
    if len(to_merge) < 2:
        raise ValueError("Need at least two particles to merge.")

    total_area = sum(p.area_px for p in to_merge)
    min_row = min(p.bbox[0] for p in to_merge)
    min_col = min(p.bbox[1] for p in to_merge)
    max_row = max(p.bbox[2] for p in to_merge)
    max_col = max(p.bbox[3] for p in to_merge)
    centroid_x = sum(p.centroid_xy[0] * p.area_px for p in to_merge) / total_area
    centroid_y = sum(p.centroid_xy[1] * p.area_px for p in to_merge) / total_area
    # Perimeter of a merge is not simply additive; approximate with the
    # equivalent-circle perimeter, flagged so it's not mistaken for a
    # directly measured value.
    approx_diameter = equivalent_circular_diameter(total_area)
    approx_perimeter = math.pi * approx_diameter

    for p in to_merge:
        particles.remove(p)

    next_id = max([p.particle_id for p in particles], default=0) + 1
    merged = ParticleMeasurement(
        particle_id=next_id,
        area_px=total_area,
        perimeter_px=approx_perimeter,
        centroid_xy=(centroid_x, centroid_y),
        bbox=(min_row, min_col, max_row, max_col),
        width_px=float(max_col - min_col),
        height_px=float(max_row - min_row),
        aspect_ratio=(max_col - min_col) / max(1.0, (max_row - min_row)),
        circularity=circularity(total_area, approx_perimeter),
        equiv_diameter_px=approx_diameter,
        feret_diameter_px=approx_diameter,
        orientation_deg=0.0,
        confidence=min(p.confidence for p in to_merge),
        provenance=ParticleProvenance.MANUAL_ADDED,
        flags=[],
    )
    particles.append(merged)
    return merged


# ---------------------------------------------------------------------------
# Fragmentation statistics
# ---------------------------------------------------------------------------
MIN_PARTICLES_FOR_FIT = 30


class SizeBasis(str, Enum):
    """What each particle contributes to the distribution.

    NUMBER weights every particle equally. AREA weights by projected area,
    which is the closer image-derived analogue to a mass-based sieve curve.
    There is deliberately no MASS or VOLUME basis - 2D area does not convert
    to 3D mass/volume without an extra, unvalidated assumption, and this
    tool won't silently make that assumption for you.
    """

    NUMBER = "number"
    AREA = "area"


@dataclass
class CumulativeDistribution:
    sizes_sorted: list[float]
    cumulative_passing_pct: list[float]
    basis: SizeBasis


def build_cumulative_distribution(
    sizes: list[float],
    weights: list[float] | None,
    basis: SizeBasis,
) -> CumulativeDistribution:
    if not sizes:
        raise ValueError("sizes must be non-empty")

    order = np.argsort(sizes)
    sizes_sorted = np.asarray(sizes)[order]

    if basis is SizeBasis.NUMBER:
        w = np.ones_like(sizes_sorted, dtype=float)
    else:
        if weights is None:
            raise ValueError("weights (areas) required for AREA basis")
        w = np.asarray(weights)[order]

    cumulative = np.cumsum(w)
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("total weight must be > 0")
    cumulative_pct = 100.0 * cumulative / total

    return CumulativeDistribution(
        sizes_sorted=sizes_sorted.tolist(),
        cumulative_passing_pct=cumulative_pct.tolist(),
        basis=basis,
    )


def percentile_size(dist: CumulativeDistribution, percentile: float) -> float:
    if not 0 < percentile < 100:
        raise ValueError("percentile must be in (0, 100)")
    return float(np.interp(percentile, dist.cumulative_passing_pct, dist.sizes_sorted))


def d_values(
    dist: CumulativeDistribution,
    percentiles: tuple[float, ...] = (10, 20, 30, 50, 60, 80, 90),
) -> dict[str, float]:
    return {f"D{int(p)}": percentile_size(dist, p) for p in percentiles}


def oversize_percentage(
    sizes: list[float],
    weights: list[float] | None,
    threshold: float,
    basis: SizeBasis,
) -> float:
    sizes_arr = np.asarray(sizes)
    if basis is SizeBasis.NUMBER:
        total = len(sizes_arr)
        if total == 0:
            raise ValueError("sizes must be non-empty")
        return 100.0 * int(np.sum(sizes_arr > threshold)) / total
    if weights is None:
        raise ValueError("weights (areas) required for AREA basis")
    weights_arr = np.asarray(weights)
    total_w = weights_arr.sum()
    if total_w <= 0:
        raise ValueError("total weight must be > 0")
    over_w = weights_arr[sizes_arr > threshold].sum()
    return 100.0 * float(over_w) / float(total_w)


def fines_percentage(
    sizes: list[float],
    weights: list[float] | None,
    threshold: float,
    basis: SizeBasis,
) -> float:
    sizes_arr = np.asarray(sizes)
    if basis is SizeBasis.NUMBER:
        total = len(sizes_arr)
        if total == 0:
            raise ValueError("sizes must be non-empty")
        return 100.0 * int(np.sum(sizes_arr < threshold)) / total
    if weights is None:
        raise ValueError("weights (areas) required for AREA basis")
    weights_arr = np.asarray(weights)
    total_w = weights_arr.sum()
    if total_w <= 0:
        raise ValueError("total weight must be > 0")
    under_w = weights_arr[sizes_arr < threshold].sum()
    return 100.0 * float(under_w) / float(total_w)


@dataclass
class RosinRammlerFit:
    xc: float
    n: float
    r_squared: float
    xc_stderr: float | None
    n_stderr: float | None


def _rosin_rammler_p(x: np.ndarray, xc: float, n: float) -> np.ndarray:
    return 1.0 - np.exp(-((x / xc) ** n))


def fit_rosin_rammler(
    dist: CumulativeDistribution,
    min_particles: int = MIN_PARTICLES_FOR_FIT,
) -> RosinRammlerFit | None:
    """Fit P(x) = 1 - exp[-(x/Xc)^n]. Returns None below min_particles
    rather than a fit that will look precise but isn't backed by enough
    data - see docs/algorithms/statistics.md.
    """
    n_points = len(dist.sizes_sorted)
    if n_points < min_particles:
        return None

    x = np.asarray(dist.sizes_sorted, dtype=float)
    p = np.asarray(dist.cumulative_passing_pct, dtype=float) / 100.0
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    valid = x > 0

    y_lin = np.log(np.log(1.0 / (1.0 - p_clipped[valid])))
    x_lin = np.log(x[valid])
    n_seed, intercept = np.polyfit(x_lin, y_lin, 1)
    xc_seed = math.exp(-intercept / n_seed) if n_seed != 0 else float(np.median(x))
    n_seed = abs(n_seed) if n_seed != 0 else 1.0

    try:
        popt, pcov = curve_fit(
            _rosin_rammler_p, x, p, p0=[xc_seed, n_seed],
            bounds=([1e-9, 1e-9], [np.inf, np.inf]), maxfev=10000,
        )
    except RuntimeError:
        return None

    xc_fit, n_fit = popt
    predicted = _rosin_rammler_p(x, xc_fit, n_fit)
    ss_res = float(np.sum((p - predicted) ** 2))
    ss_tot = float(np.sum((p - np.mean(p)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    stderr = np.sqrt(np.diag(pcov)) if pcov is not None else (None, None)

    return RosinRammlerFit(
        xc=float(xc_fit),
        n=float(n_fit),
        r_squared=r_squared,
        xc_stderr=float(stderr[0]) if stderr[0] is not None else None,
        n_stderr=float(stderr[1]) if stderr[1] is not None else None,
    )


# ---------------------------------------------------------------------------
# Reporting and export
# ---------------------------------------------------------------------------
BOUNDARY_COLOR = (235, 206, 135)  # BGR - sky blue, matches app theme
ID_COLOR = (255, 255, 255)
FLAG_COLOR = (60, 60, 220)


def draw_annotations(
    image_bgr: np.ndarray,
    label_image: np.ndarray,
    particles: list[ParticleMeasurement],
    show_boundaries: bool = True,
    show_ids: bool = False,
    show_flagged: bool = True,
) -> np.ndarray:
    output = image_bgr.copy()
    kept_ids = {p.particle_id for p in particles if p.kept}

    if show_boundaries:
        for particle_id in kept_ids:
            mask = (label_image == particle_id).astype(np.uint8)
            if mask.sum() == 0:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            particle = next((p for p in particles if p.particle_id == particle_id), None)
            color = BOUNDARY_COLOR
            if particle and particle.flags and show_flagged:
                color = FLAG_COLOR
            cv2.drawContours(output, contours, -1, color, 2)

    if show_ids:
        for particle in particles:
            if not particle.kept:
                continue
            cx, cy = int(particle.centroid_xy[0]), int(particle.centroid_xy[1])
            cv2.putText(
                output, str(particle.particle_id), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ID_COLOR, 1, cv2.LINE_AA,
            )

    return output


def particles_to_dataframe(
    particles: list[ParticleMeasurement],
    calibration: CalibrationResult | None,
) -> pd.DataFrame:
    rows = []
    for p in particles:
        row = {
            "particle_id": p.particle_id,
            "kept": p.kept,
            "provenance": p.provenance.value,
            "area_px": p.area_px,
            "equiv_diameter_px": p.equiv_diameter_px,
            "feret_diameter_px": p.feret_diameter_px,
            "circularity": round(p.circularity, 3),
            "aspect_ratio": round(p.aspect_ratio, 3),
            "confidence": p.confidence,
            "flags": ", ".join(f.value for f in p.flags),
        }
        if calibration is not None and calibration.is_valid:
            row["area_mm2"] = area_px_to_mm2(p.area_px, calibration)
            row["equiv_diameter_mm"] = pixels_to_mm(p.equiv_diameter_px, calibration)
            if p.feret_diameter_px is not None:
                row["feret_diameter_mm"] = pixels_to_mm(p.feret_diameter_px, calibration)
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="particles")
    return buffer.getvalue()


def build_pdf_report(
    blast_name: str,
    image_name: str,
    summary: dict,
    warnings: list[str],
    annotated_image_bgr: np.ndarray | None = None,
) -> bytes:
    """One-page PDF summary: blast/image identity, key stats, warnings, and
    the annotated image if available. Uses reportlab directly rather than a
    templating layer, since a single page doesn't need one.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 25 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, y, "Fragmentation Analysis Report")
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, f"Blast: {blast_name}")
    y -= 5 * mm
    c.drawString(20 * mm, y, f"Image: {image_name}")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Summary statistics")
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    for key, value in summary.items():
        if isinstance(value, float):
            value = f"{value:.2f}"
        c.drawString(22 * mm, y, f"{key}: {value}")
        y -= 5 * mm

    if warnings:
        y -= 5 * mm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(20 * mm, y, "Warnings")
        y -= 6 * mm
        c.setFont("Helvetica", 9)
        for w in warnings:
            for line in _wrap(w, 95):
                c.drawString(22 * mm, y, line)
                y -= 4.5 * mm

    if annotated_image_bgr is not None:
        rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        image_reader = ImageReader(io.BytesIO(cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()))
        max_width = width - 40 * mm
        aspect = rgb.shape[0] / rgb.shape[1]
        draw_width = max_width
        draw_height = draw_width * aspect
        if draw_height > y - 20 * mm:
            draw_height = y - 20 * mm
            draw_width = draw_height / aspect
        c.drawImage(
            image_reader, 20 * mm, y - draw_height - 5 * mm,
            width=draw_width, height=draw_height, preserveAspectRatio=True,
        )

    c.showPage()
    c.save()
    return buffer.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}".strip()
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Upload storage
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class UploadValidationError(ValueError):
    pass


@dataclass
class StoredImage:
    stored_filename: str
    stored_path: str
    original_filename: str
    width_px: int
    height_px: int
    quality_warnings: list[str]


def save_uploaded_image(raw_bytes: bytes, original_filename: str, upload_dir: str) -> StoredImage:
    if len(raw_bytes) == 0:
        raise UploadValidationError("Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise UploadValidationError(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise UploadValidationError("Could not decode this as an image - it may be corrupted or an unsupported format.")

    height, width = image_bgr.shape[:2]
    _, ext = os.path.splitext(original_filename)
    ext = ext.lower() if ext.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"} else ".bin"
    stored_filename = f"{uuid.uuid4()}{ext}"

    os.makedirs(upload_dir, exist_ok=True)
    stored_path = os.path.join(upload_dir, stored_filename)
    with open(stored_path, "wb") as f:
        f.write(raw_bytes)

    quality = assess_image_quality(image_bgr)

    return StoredImage(
        stored_filename=stored_filename,
        stored_path=stored_path,
        original_filename=os.path.basename(original_filename),
        width_px=width,
        height_px=height,
        quality_warnings=quality.warnings,
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get("FRAGANALYST_DB_PATH", "data/fraganalyst.db")
UPLOAD_DIR = os.environ.get("FRAGANALYST_UPLOAD_DIR", "data/uploads")

SCHEMA = """
CREATE TABLE IF NOT EXISTS blasts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    mine_or_quarry TEXT,
    bench TEXT,
    blast_date TEXT,
    rock_type TEXT,
    ucs_mpa REAL,
    burden_m REAL,
    spacing_m REAL,
    bench_height_m REAL,
    stemming_length_m REAL,
    subdrill_m REAL,
    dummy_hole_depth_m REAL,
    dummy_hole_spacing_m REAL,
    hole_diameter_mm REAL,
    explosive_type TEXT,
    explosive_density_g_cm3 REAL,
    charge_length_m REAL,
    powder_factor_kg_m3 REAL,
    num_holes INTEGER,
    stemming_material TEXT,
    weather_notes TEXT,
    operator_notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    blast_id TEXT NOT NULL REFERENCES blasts(id),
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    width_px INTEGER NOT NULL,
    height_px INTEGER NOT NULL,
    quality_warnings_json TEXT,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calibrations (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id),
    method TEXT NOT NULL,
    scale_mm_per_px REAL NOT NULL,
    is_valid INTEGER NOT NULL,
    coefficient_of_variation REAL,
    consistency_warning TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id),
    blast_id TEXT NOT NULL REFERENCES blasts(id),
    calibration_id TEXT REFERENCES calibrations(id),
    segmentation_config_json TEXT,
    particles_json TEXT NOT NULL,
    summary_json TEXT,
    oversize_threshold_mm REAL,
    size_basis TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def create_blast(fields: dict) -> str:
    blast_id = _new_id()
    fields = {**fields, "id": blast_id, "created_at": _now()}
    columns = ", ".join(fields.keys())
    placeholders = ", ".join(f":{k}" for k in fields.keys())
    with get_connection() as conn:
        conn.execute(f"INSERT INTO blasts ({columns}) VALUES ({placeholders})", fields)
    return blast_id


def list_blasts() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM blasts ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_blast(blast_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM blasts WHERE id = ?", (blast_id,)).fetchone()
    return dict(row) if row else None


def add_image(
    blast_id: str,
    original_filename: str,
    stored_filename: str,
    width_px: int,
    height_px: int,
    quality_warnings: list[str],
) -> str:
    image_id = _new_id()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO images
               (id, blast_id, original_filename, stored_filename, width_px, height_px,
                quality_warnings_json, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                image_id, blast_id, original_filename, stored_filename,
                width_px, height_px, json.dumps(quality_warnings), _now(),
            ),
        )
    return image_id


def list_images(blast_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM images WHERE blast_id = ? ORDER BY uploaded_at DESC", (blast_id,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["quality_warnings"] = json.loads(d.pop("quality_warnings_json") or "[]")
        result.append(d)
    return result


def get_image(image_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["quality_warnings"] = json.loads(d.pop("quality_warnings_json") or "[]")
    return d


def save_calibration(image_id: str, result) -> str:
    calibration_id = _new_id()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO calibrations
               (id, image_id, method, scale_mm_per_px, is_valid,
                coefficient_of_variation, consistency_warning, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                calibration_id, image_id, result.method.value, result.scale_mm_per_px,
                int(result.is_valid), result.coefficient_of_variation,
                result.consistency_warning, _now(),
            ),
        )
    return calibration_id


def get_latest_calibration(image_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM calibrations WHERE image_id = ? ORDER BY created_at DESC LIMIT 1",
            (image_id,),
        ).fetchone()
    return dict(row) if row else None


def save_analysis(
    image_id: str,
    blast_id: str,
    calibration_id: str | None,
    particles: list,
    summary: dict,
    oversize_threshold_mm: float | None,
    size_basis: str,
    segmentation_config: dict | None = None,
) -> str:
    analysis_id = _new_id()
    particles_json = json.dumps([_particle_to_dict(p) for p in particles])
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO analyses
               (id, image_id, blast_id, calibration_id, segmentation_config_json,
                particles_json, summary_json, oversize_threshold_mm, size_basis, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis_id, image_id, blast_id, calibration_id,
                json.dumps(segmentation_config) if segmentation_config else None,
                particles_json, json.dumps(summary), oversize_threshold_mm, size_basis, _now(),
            ),
        )
    return analysis_id


def list_analyses(blast_id: str | None = None) -> list[dict]:
    with get_connection() as conn:
        if blast_id:
            rows = conn.execute(
                "SELECT * FROM analyses WHERE blast_id = ? ORDER BY created_at DESC", (blast_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM analyses ORDER BY created_at DESC").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        d.pop("particles_json", None)
        result.append(d)
    return result


def get_analysis(analysis_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["summary"] = json.loads(d.pop("summary_json") or "{}")
    d["particles"] = json.loads(d.pop("particles_json") or "[]")
    d["segmentation_config"] = json.loads(d.pop("segmentation_config_json") or "null")
    return d


def get_latest_analysis_for_image(image_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM analyses WHERE image_id = ? ORDER BY created_at DESC LIMIT 1",
            (image_id,),
        ).fetchone()
    return get_analysis(row["id"]) if row else None


def get_latest_analyses_for_blast(blast_id: str) -> dict:
    """One analysis per image: the most recently saved one. Images with no
    saved analysis at all are simply absent from the result, since there is
    nothing to roll up for them yet.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, image_id FROM analyses WHERE blast_id = ? ORDER BY created_at DESC",
            (blast_id,),
        ).fetchall()
    latest_by_image: dict[str, dict] = {}
    for row in rows:
        if row["image_id"] in latest_by_image:
            continue
        latest_by_image[row["image_id"]] = get_analysis(row["id"])
    return latest_by_image


def get_calibration(calibration_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM calibrations WHERE id = ?", (calibration_id,)).fetchone()
    return dict(row) if row else None


def _particle_to_dict(p) -> dict:
    d = asdict(p)
    d["provenance"] = p.provenance.value
    d["flags"] = [f.value for f in p.flags]
    return d


# ---------------------------------------------------------------------------
# Blast-level aggregation (multi-image rollup)
# ---------------------------------------------------------------------------
@dataclass
class ImageContribution:
    image_id: str
    original_filename: str
    stored_filename: str
    particle_count: int
    area_mm2: float
    mean_circularity: float


@dataclass
class BlastAggregate:
    combined_df: pd.DataFrame
    included: list[ImageContribution]
    skipped: list[tuple[str, str]]  # (filename, reason)

    @property
    def is_empty(self) -> bool:
        return self.combined_df.empty


def aggregate_blast_particles(blast_id: str, image_ids: list[str] | None = None) -> BlastAggregate:
    """Pool kept particles from every analyzed image in a blast into one
    dataframe, in physical units. One row per particle, so this is the
    thing that scales to thousands of rows across many images - everything
    downstream (concat, column math) is pandas/numpy vectorized rather than
    a python loop over particles.
    """
    images_by_id = {img["id"]: img for img in list_images(blast_id)}
    latest_analyses = get_latest_analyses_for_blast(blast_id)

    candidate_ids = image_ids if image_ids is not None else list(images_by_id.keys())

    frames: list[pd.DataFrame] = []
    included: list[ImageContribution] = []
    skipped: list[tuple[str, str]] = []

    for image_id in candidate_ids:
        image = images_by_id.get(image_id)
        if image is None:
            continue
        analysis = latest_analyses.get(image_id)
        if analysis is None:
            skipped.append((image["original_filename"], "not analyzed yet"))
            continue

        calibration = get_calibration(analysis["calibration_id"]) if analysis.get("calibration_id") else None
        if not calibration or not calibration["is_valid"]:
            skipped.append((image["original_filename"], "no valid calibration"))
            continue
        scale = calibration["scale_mm_per_px"]

        df = pd.DataFrame(analysis["particles"])
        if df.empty:
            skipped.append((image["original_filename"], "no particles recorded"))
            continue
        df = df[df["kept"]].copy()
        if df.empty:
            skipped.append((image["original_filename"], "all particles removed"))
            continue

        df["equiv_diameter_mm"] = df["equiv_diameter_px"] * scale
        df["area_mm2"] = df["area_px"] * (scale ** 2)
        df["source_image_id"] = image_id
        df["source_image"] = image["original_filename"]
        frames.append(df)

        included.append(ImageContribution(
            image_id=image_id,
            original_filename=image["original_filename"],
            stored_filename=image["stored_filename"],
            particle_count=int(len(df)),
            area_mm2=float(df["area_mm2"].sum()),
            mean_circularity=float(df["circularity"].mean()),
        ))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["equiv_diameter_mm", "area_mm2", "circularity", "source_image"]
    )
    return BlastAggregate(combined_df=combined, included=included, skipped=skipped)


@dataclass
class AggregateSummary:
    particle_count: int
    total_area_mm2: float
    mean_circularity: float
    dist: CumulativeDistribution
    d_values: dict[str, float]
    rosin_rammler: RosinRammlerFit | None


def compute_aggregate_summary(aggregate: BlastAggregate, basis: SizeBasis) -> AggregateSummary | None:
    if aggregate.is_empty:
        return None
    df = aggregate.combined_df
    sizes = df["equiv_diameter_mm"].to_numpy()
    areas = df["area_mm2"].to_numpy()
    weights = areas if basis is SizeBasis.AREA else None

    dist = build_cumulative_distribution(sizes.tolist(), weights.tolist() if weights is not None else None, basis)
    dv = d_values(dist)
    fit = fit_rosin_rammler(dist)

    return AggregateSummary(
        particle_count=int(len(df)),
        total_area_mm2=float(areas.sum()),
        mean_circularity=float(df["circularity"].mean()),
        dist=dist,
        d_values=dv,
        rosin_rammler=fit,
    )


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
DEFAULT_SEG_CONFIG_UI = {
    "threshold_mode": "otsu",
    "min_distance": 15,
    "morph_kernel": 3,
    "invert": True,
    "min_area": 16.0,
    "max_aspect": 6.0,
    "min_confidence": 0.5,
}

DEFAULTS = {
    "current_blast_id": None,
    "current_image_id": None,
    "selected_image_ids": [],
    "particles_by_image": {},
    "labels_by_image": {},
    "seg_config_by_image": {},
    "calibration_points": [],
}


def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = dict(value) if isinstance(value, dict) else (list(value) if isinstance(value, list) else value)


def load_image_bgr(stored_filename: str) -> np.ndarray | None:
    path = os.path.join(UPLOAD_DIR, stored_filename)
    if not os.path.isfile(path):
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_image_rgb(stored_filename: str) -> np.ndarray | None:
    bgr = load_image_bgr(stored_filename)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_current_blast() -> dict | None:
    if not st.session_state.get("current_blast_id"):
        return None
    return get_blast(st.session_state["current_blast_id"])


def get_current_image() -> dict | None:
    if not st.session_state.get("current_image_id"):
        return None
    return get_image(st.session_state["current_image_id"])


def get_particles_for_image(image_id: str) -> list | None:
    return st.session_state["particles_by_image"].get(image_id)


def set_particles_for_image(image_id: str, particles: list) -> None:
    st.session_state["particles_by_image"][image_id] = particles


def get_label_image(image_id: str) -> np.ndarray | None:
    return st.session_state["labels_by_image"].get(image_id)


def set_label_image(image_id: str, labels: np.ndarray) -> None:
    st.session_state["labels_by_image"][image_id] = labels


def get_seg_config_for_image(image_id: str) -> dict:
    return st.session_state["seg_config_by_image"].get(image_id, DEFAULT_SEG_CONFIG_UI.copy())


def set_seg_config_for_image(image_id: str, config: dict) -> None:
    st.session_state["seg_config_by_image"][image_id] = config


def reset_image_working_state(image_id: str) -> None:
    """Clear cached (unsaved) segmentation results for one image - used
    when switching to a freshly selected image so a previous image's
    particles don't linger on screen before segmentation has run for this
    one. Nothing is deleted from the database.
    """
    st.session_state["particles_by_image"].pop(image_id, None)
    st.session_state["labels_by_image"].pop(image_id, None)


# ---------------------------------------------------------------------------
# Page: Blasts
# ---------------------------------------------------------------------------
def render_blasts() -> None:
    st.title("Blasts")
    st.caption("A blast can hold several photographs. Engineering parameters below are stored as metadata only - they are never used in size calculations.")

    with st.expander("New blast", expanded=len(list_blasts()) == 0):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Blast name", placeholder="e.g. Chikowa Bench 3, Blast A")
            mine_or_quarry = st.text_input("Mine / quarry", placeholder="e.g. Shayona Chikowa")
            bench = st.text_input("Bench")
            rock_type = st.text_input("Rock type", placeholder="e.g. limestone")
        with col2:
            burden_m = st.number_input("Burden (m)", min_value=0.0, value=0.0, step=0.1)
            spacing_m = st.number_input("Spacing (m)", min_value=0.0, value=0.0, step=0.1)
            dummy_hole_depth_m = st.number_input("Dummy-hole depth (m)", min_value=0.0, value=0.0, step=0.1)
            dummy_hole_spacing_m = st.number_input("Dummy-hole spacing (m)", min_value=0.0, value=0.0, step=0.1)

        with st.expander("More blast parameters (optional)"):
            c1, c2, c3 = st.columns(3)
            with c1:
                ucs_mpa = st.number_input("UCS (MPa)", min_value=0.0, value=0.0, step=1.0)
                bench_height_m = st.number_input("Bench height (m)", min_value=0.0, value=0.0, step=0.1)
                stemming_length_m = st.number_input("Stemming length (m)", min_value=0.0, value=0.0, step=0.1)
            with c2:
                subdrill_m = st.number_input("Subdrill (m)", min_value=0.0, value=0.0, step=0.1)
                hole_diameter_mm = st.number_input("Hole diameter (mm)", min_value=0.0, value=0.0, step=1.0)
                num_holes = st.number_input("Number of holes", min_value=0, value=0, step=1)
            with c3:
                explosive_type = st.text_input("Explosive type")
                powder_factor = st.number_input("Powder factor (kg/m3)", min_value=0.0, value=0.0, step=0.01)
                stemming_material = st.text_input("Stemming material")
            operator_notes = st.text_area("Operator notes")

        if st.button("Create blast", type="primary", disabled=not name.strip()):
            fields = {
                "name": name.strip(),
                "mine_or_quarry": mine_or_quarry or None,
                "bench": bench or None,
                "rock_type": rock_type or None,
                "ucs_mpa": ucs_mpa or None,
                "burden_m": burden_m or None,
                "spacing_m": spacing_m or None,
                "bench_height_m": bench_height_m or None,
                "stemming_length_m": stemming_length_m or None,
                "subdrill_m": subdrill_m or None,
                "dummy_hole_depth_m": dummy_hole_depth_m or None,
                "dummy_hole_spacing_m": dummy_hole_spacing_m or None,
                "hole_diameter_mm": hole_diameter_mm or None,
                "explosive_type": explosive_type or None,
                "explosive_density_g_cm3": None,
                "charge_length_m": None,
                "powder_factor_kg_m3": powder_factor or None,
                "num_holes": int(num_holes) or None,
                "stemming_material": stemming_material or None,
                "weather_notes": None,
                "operator_notes": operator_notes or None,
                "blast_date": None,
            }
            blast_id = create_blast(fields)
            st.session_state["current_blast_id"] = blast_id
            st.session_state["current_image_id"] = None
            st.session_state["selected_image_ids"] = []
            st.success(f"Created blast '{name}'.")
            st.rerun()

    st.divider()
    st.subheader("Existing blasts")
    blasts = list_blasts()
    if not blasts:
        st.info("No blasts yet - create one above.")
        return

    for blast in blasts:
        images = list_images(blast["id"])
        is_current = blast["id"] == st.session_state.get("current_blast_id")
        cols = st.columns([5, 2, 2])
        with cols[0]:
            label = f"**{blast['name']}**"
            if blast.get("rock_type"):
                label += f"  ·  {blast['rock_type']}"
            st.markdown(label)
            st.caption(f"{len(images)} image(s)")
        with cols[1]:
            if st.button("Select" if not is_current else "Selected", key=f"select_{blast['id']}", disabled=is_current):
                st.session_state["current_blast_id"] = blast["id"]
                st.session_state["current_image_id"] = None
                st.session_state["selected_image_ids"] = [img["id"] for img in images]
                st.rerun()
        with cols[2]:
            if is_current:
                st.markdown(":large_blue_circle: active")


# ---------------------------------------------------------------------------
# Page: Upload
# ---------------------------------------------------------------------------
def render_upload() -> None:
    st.title("Image Upload")
    blast = get_current_blast()
    if blast is None:
        st.warning("Select or create a blast first, on the Blasts page.")
        return

    st.caption(f"Blast: **{blast['name']}**")

    files = st.file_uploader(
        "Blast / muckpile photographs",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
    )

    if files and st.button("Upload", type="primary"):
        uploaded_count = 0
        for file in files:
            try:
                stored = save_uploaded_image(file.getvalue(), file.name, UPLOAD_DIR)
            except UploadValidationError as exc:
                st.error(f"{file.name}: {exc}")
                continue
            new_image_id = add_image(
                blast["id"], stored.original_filename, stored.stored_filename,
                stored.width_px, stored.height_px, stored.quality_warnings,
            )
            st.session_state["selected_image_ids"].append(new_image_id)
            uploaded_count += 1
        if uploaded_count:
            st.success(f"Uploaded {uploaded_count} image(s).")
            st.rerun()

    st.divider()
    images = list_images(blast["id"])
    if not images:
        st.info("No images uploaded yet.")
        return

    st.subheader(f"Images ({len(images)})")
    cols = st.columns(4)
    for i, image in enumerate(images):
        with cols[i % 4]:
            thumb_path = f"{UPLOAD_DIR}/{image['stored_filename']}"
            st.image(thumb_path, width="stretch")
            caption = f"{image['width_px']}x{image['height_px']}"
            if image["quality_warnings"]:
                caption += f" - {len(image['quality_warnings'])} warning(s)"
            st.caption(caption)
            is_current = image["id"] == st.session_state.get("current_image_id")
            if st.button("Selected" if is_current else "Select", key=f"select_img_{image['id']}", disabled=is_current, width="stretch"):
                st.session_state["current_image_id"] = image["id"]
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Calibrate
# ---------------------------------------------------------------------------
DISPLAY_WIDTH = 700


def _reliability_line(image: dict, calibration: dict | None) -> None:
    cols = st.columns(2)
    with cols[0]:
        if calibration and calibration["is_valid"]:
            st.success(f"Calibrated: {calibration['scale_mm_per_px']:.4f} mm/px")
        else:
            st.error("Not calibrated - sizes will be shown in pixels only.")
    with cols[1]:
        warnings = image.get("quality_warnings") or []
        if warnings:
            st.warning(f"Image quality: {len(warnings)} warning(s)")
        else:
            st.success("Image quality: no warnings")
    if calibration and calibration.get("consistency_warning"):
        st.warning(calibration["consistency_warning"])


def render_calibrate() -> None:
    st.title("Scale Calibration")
    image = get_current_image()
    if image is None:
        st.warning("Select an image on the Upload page first.")
        return

    latest_calibration = get_latest_calibration(image["id"])
    _reliability_line(image, latest_calibration)

    for warning in image.get("quality_warnings") or []:
        st.info(warning)

    rgb = load_image_rgb(image["stored_filename"])
    if rgb is None:
        st.error("Stored image file is missing.")
        return

    scale = min(1.0, DISPLAY_WIDTH / image["width_px"])
    display_w = int(image["width_px"] * scale)
    display_h = int(image["height_px"] * scale)

    pil_img = PILImage.fromarray(rgb).resize((display_w, display_h))

    st.markdown("**Method A - reference object.** Click one end of a reference object, then the other.")
    coords = streamlit_image_coordinates(pil_img, key="calibration_click")

    points = st.session_state["calibration_points"]
    if coords is not None:
        real_point = Point2D(coords["x"] / scale, coords["y"] / scale)
        if len(points) < 2:
            already_have = points and points[-1] == real_point
            if not already_have:
                points.append(real_point)
                st.session_state["calibration_points"] = points
                st.rerun()

    if points:
        st.caption(f"{len(points)}/2 points placed.")
        if st.button("Clear points"):
            st.session_state["calibration_points"] = []
            st.rerun()

    known_length_mm = st.number_input("Known physical length of that reference (mm)", min_value=0.0, value=0.0, step=1.0)

    if st.button("Save calibration from points", type="primary", disabled=len(points) != 2 or known_length_mm <= 0):
        try:
            result = calibrate_from_reference_points([
                ReferenceMeasurement(points[0], points[1], known_length_mm, "user reference")
            ])
        except CalibrationError as exc:
            st.error(str(exc))
        else:
            save_calibration(image["id"], result)
            st.session_state["calibration_points"] = []
            st.success(f"Saved: {result.scale_mm_per_px:.4f} mm/px")
            st.rerun()

    st.divider()
    st.markdown("**Method C - manual scale entry.**")
    mm_per_px = st.number_input("Millimetres per pixel", min_value=0.0, value=0.0, step=0.01, format="%.4f")
    if st.button("Save manual scale", disabled=mm_per_px <= 0):
        result = calibrate_from_manual_entry(mm_per_pixel=mm_per_px)
        save_calibration(image["id"], result)
        st.success(f"Saved: {result.scale_mm_per_px:.4f} mm/px")
        st.rerun()


# ---------------------------------------------------------------------------
# Page: Segmentation & review
# ---------------------------------------------------------------------------
DISPLAY_WIDTH = 700


def render_segment() -> None:
    st.title("Segmentation & Particle Review")
    image = get_current_image()
    if image is None:
        st.warning("Select an image on the Upload page first.")
        return

    image_id = image["id"]
    image_bgr = load_image_bgr(image["stored_filename"])
    if image_bgr is None:
        st.error("Stored image file is missing.")
        return

    st.caption(f"Editing **{image['original_filename']}** - settings and particles below belong only to this image.")

    saved_config = get_seg_config_for_image(image_id)

    with st.sidebar:
        st.subheader("Segmentation settings")
        threshold_mode = st.selectbox(
            "Threshold mode", ["otsu", "adaptive"],
            index=["otsu", "adaptive"].index(saved_config["threshold_mode"]),
            key=f"threshold_mode_{image_id}",
        )
        min_distance = st.slider(
            "Minimum distance between particle centers (px)", 3, 60,
            saved_config["min_distance"], key=f"min_distance_{image_id}",
        )
        morph_kernel = st.slider(
            "Morphological kernel size", 1, 9, saved_config["morph_kernel"],
            step=2, key=f"morph_kernel_{image_id}",
        )
        invert = st.checkbox(
            "Invert threshold (rocks lighter than background)",
            value=saved_config["invert"], key=f"invert_{image_id}",
        )

        st.subheader("Quality-control thresholds")
        min_area = st.number_input(
            "Minimum particle area (px)", min_value=1.0,
            value=saved_config["min_area"], key=f"min_area_{image_id}",
        )
        max_aspect = st.number_input(
            "Maximum aspect ratio", min_value=1.0,
            value=saved_config["max_aspect"], key=f"max_aspect_{image_id}",
        )
        min_confidence = st.slider(
            "Minimum confidence", 0.0, 1.0, saved_config["min_confidence"],
            key=f"min_confidence_{image_id}",
        )

    current_config = {
        "threshold_mode": threshold_mode,
        "min_distance": min_distance,
        "morph_kernel": morph_kernel,
        "invert": invert,
        "min_area": min_area,
        "max_aspect": max_aspect,
        "min_confidence": min_confidence,
    }

    if st.button("Run segmentation", type="primary"):
        config = SegmentationConfig(
            threshold_mode=ThresholdMode(threshold_mode),
            min_distance_between_peaks=min_distance,
            morph_kernel=morph_kernel,
            invert=invert,
        )
        labels = segment(image_bgr, config)
        confidences = label_confidences(labels)
        qc = QCThresholds(min_area_px=min_area, max_aspect_ratio=max_aspect, min_confidence=min_confidence)
        particles = measure_particles_from_labels(labels, confidences, qc)

        set_label_image(image_id, labels)
        set_particles_for_image(image_id, particles)
        set_seg_config_for_image(image_id, current_config)
        st.success(f"Found {len(particles)} particle(s) in {image['original_filename']}.")

    particles = get_particles_for_image(image_id)
    labels = get_label_image(image_id)

    if particles is None or labels is None:
        st.info("Run segmentation to see results here.")
        return

    st.divider()
    col_display, col_table = st.columns([3, 2])

    with col_display:
        show_boundaries = st.checkbox("Show boundaries", value=True, key=f"show_boundaries_{image_id}")
        show_ids = st.checkbox("Show particle IDs", value=False, key=f"show_ids_{image_id}")
        annotated = draw_annotations(image_bgr, labels, particles, show_boundaries, show_ids)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, width="stretch", caption="Detected particles (red outline = flagged)")

        st.markdown("**Add a missed particle** - click its center, then set a radius and confirm.")

        scale = min(1.0, DISPLAY_WIDTH / image["width_px"])
        display_img = PILImage.fromarray(annotated_rgb).resize(
            (int(image["width_px"] * scale), int(image["height_px"] * scale))
        )
        click = streamlit_image_coordinates(display_img, key=f"add_particle_click_{image_id}")
        radius = st.slider("Estimated radius (px)", 3, 200, 20, key=f"add_radius_{image_id}")
        if click is not None and st.button("Add particle at last click"):
            center = (click["x"] / scale, click["y"] / scale)
            add_manual_particle(particles, center, float(radius))
            set_particles_for_image(image_id, particles)
            st.rerun()

    with col_table:
        df = particles_to_dataframe(particles, calibration=None)
        edited = st.data_editor(
            df[["particle_id", "kept", "equiv_diameter_px", "circularity", "flags"]],
            hide_index=True,
            disabled=["particle_id", "equiv_diameter_px", "circularity", "flags"],
            width="stretch",
            key=f"particle_editor_{image_id}",
        )
        if st.button("Apply keep/remove changes"):
            kept_map = dict(zip(edited["particle_id"], edited["kept"]))
            for p in particles:
                if p.particle_id in kept_map:
                    p.kept = bool(kept_map[p.particle_id])
            set_particles_for_image(image_id, particles)
            st.rerun()

        st.markdown("**Merge particles**")
        ids = [p.particle_id for p in particles]
        to_merge = st.multiselect("Select two or more IDs to merge", ids, key=f"merge_select_{image_id}")
        if st.button("Merge selected", disabled=len(to_merge) < 2):
            merge_particles(particles, to_merge)
            set_particles_for_image(image_id, particles)
            st.rerun()

        removed = sum(1 for p in particles if not p.kept)
        added = sum(1 for p in particles if p.provenance.value == "manual_added")
        st.caption(f"{len(particles)} total - {removed} removed - {added} manually added")


# ---------------------------------------------------------------------------
# Page: Blast overview
# ---------------------------------------------------------------------------
def render_overview() -> None:
    st.title("Blast Overview")
    blast = get_current_blast()
    if blast is None:
        st.warning("Select a blast first, on the Blasts page.")
        return

    all_images = list_images(blast["id"])
    if not all_images:
        st.info("No images uploaded to this blast yet.")
        return

    selected_ids = [
        i for i in st.session_state["selected_image_ids"]
        if i in {img["id"] for img in all_images}
    ]
    st.caption(
        f"Rolling up {len(selected_ids)} of {len(all_images)} image(s) in **{blast['name']}** - "
        "change which images are included from the sidebar checklist."
    )
    if not selected_ids:
        st.info("No images selected for this overview. Check some in the sidebar.")
        return

    basis_label = st.radio("Distribution weighting", ["Area (recommended)", "Number"], horizontal=True, key="overview_basis")
    basis = SizeBasis.AREA if basis_label.startswith("Area") else SizeBasis.NUMBER

    aggregate = aggregate_blast_particles(blast["id"], selected_ids)

    if aggregate.skipped:
        with st.expander(f"{len(aggregate.skipped)} image(s) excluded from this rollup"):
            for filename, reason in aggregate.skipped:
                st.write(f"- **{filename}**: {reason}")

    if not aggregate.included:
        st.warning("None of the selected images have a saved, calibrated analysis to roll up yet.")
        return

    summary = compute_aggregate_summary(aggregate, basis)

    st.subheader("Combined metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images included", len(aggregate.included))
    c2.metric("Total particle count", summary.particle_count)
    c3.metric("Total area analyzed (mm2)", f"{summary.total_area_mm2:,.0f}")
    c4.metric("Mean circularity", f"{summary.mean_circularity:.3f}")

    st.subheader(f"Combined D-values (mm, {basis.value}-weighted)")
    dv_cols = st.columns(len(summary.d_values))
    for col, (key, value) in zip(dv_cols, summary.d_values.items()):
        col.metric(key, f"{value:.1f}")

    st.subheader("Combined cumulative passing curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary.dist.sizes_sorted, y=summary.dist.cumulative_passing_pct,
        mode="lines", name="Combined (empirical)", line=dict(color="#38BDF8", width=2),
    ))
    if summary.rosin_rammler is not None:
        fit = summary.rosin_rammler
        x_fit = np.linspace(min(summary.dist.sizes_sorted), max(summary.dist.sizes_sorted), 200)
        y_fit = 100 * (1 - np.exp(-((x_fit / fit.xc) ** fit.n)))
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit, mode="lines", name="Rosin-Rammler fit",
            line=dict(color="#FFFFFF", width=2, dash="dash"),
        ))
        st.caption(f"Rosin-Rammler: Xc = {fit.xc:.1f} mm, n = {fit.n:.2f}, R2 = {fit.r_squared:.3f}")
    fig.update_layout(
        xaxis_title="Particle size (mm)", yaxis_title="Cumulative passing (%)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The combined curve pools particles from every included image, weighted by the basis "
        "above, so each image contributes in proportion to its own area or particle count. This "
        "assumes each photo samples the muckpile without systematic bias - treat it as an "
        "estimate, not a sieve result."
    )

    st.subheader("Images in this overview")
    images_by_id = {img["id"]: img for img in all_images}
    contrib_by_id = {c.image_id: c for c in aggregate.included}
    cols = st.columns(4)
    for idx, image_id in enumerate(selected_ids):
        image = images_by_id.get(image_id)
        if image is None:
            continue
        with cols[idx % 4]:
            thumb_path = f"{UPLOAD_DIR}/{image['stored_filename']}"
            st.image(thumb_path, width="stretch")
            contribution = contrib_by_id.get(image_id)
            if contribution:
                st.caption(
                    f"{image['original_filename']}\n\n"
                    f"{contribution.particle_count} particles - {contribution.area_mm2:,.0f} mm2"
                )
            else:
                st.caption(f"{image['original_filename']} - not in rollup")


# ---------------------------------------------------------------------------
# Page: Results
# ---------------------------------------------------------------------------
def _calibration_from_record(record: dict) -> CalibrationResult:
    return CalibrationResult(
        method=CalibrationMethod(record["method"]),
        scale_mm_per_px=record["scale_mm_per_px"],
        is_valid=bool(record["is_valid"]),
        coefficient_of_variation=record.get("coefficient_of_variation"),
        consistency_warning=record.get("consistency_warning"),
    )


def render_results() -> None:
    st.title("Fragmentation Results")
    blast = get_current_blast()
    image = get_current_image()
    particles = get_particles_for_image(image["id"]) if image else None

    if blast is None or image is None:
        st.warning("Select a blast and image first.")
        return
    if not particles:
        st.info("Run segmentation on the Segmentation & Review page first.")
        return

    kept = [p for p in particles if p.kept]
    if len(kept) == 0:
        st.warning("No particles are currently kept - nothing to compute statistics from.")
        return

    calibration_record = get_latest_calibration(image["id"])
    calibration = _calibration_from_record(calibration_record) if calibration_record else None
    has_scale = calibration is not None and calibration.is_valid

    if not has_scale:
        st.error("No valid calibration for this image - sizes below are in pixels, not millimetres.")

    unit = "mm" if has_scale else "px"
    scale = calibration.scale_mm_per_px if has_scale else 1.0

    sizes = [p.equiv_diameter_px * scale for p in kept]
    areas = [p.area_px * (scale ** 2) for p in kept]

    basis_label = st.radio("Distribution weighting", ["Area (recommended)", "Number"], horizontal=True)
    basis = SizeBasis.AREA if basis_label.startswith("Area") else SizeBasis.NUMBER
    weights = areas if basis is SizeBasis.AREA else None

    dist = build_cumulative_distribution(sizes, weights, basis)
    dv = d_values(dist)

    st.subheader(f"D-values ({unit}, {basis.value}-weighted)")
    cols = st.columns(len(dv))
    for col, (key, value) in zip(cols, dv.items()):
        col.metric(key, f"{value:.1f}")

    st.subheader("Particle-size distribution")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist.sizes_sorted, y=dist.cumulative_passing_pct,
        mode="lines", name="Measured (empirical)",
        line=dict(color="#38BDF8", width=2),
    ))

    fit = fit_rosin_rammler(dist)
    if fit is not None:
        x_fit = np.linspace(min(dist.sizes_sorted), max(dist.sizes_sorted), 200)
        y_fit = 100 * (1 - np.exp(-((x_fit / fit.xc) ** fit.n)))
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit, mode="lines", name="Rosin-Rammler fit",
            line=dict(color="#FFFFFF", width=2, dash="dash"),
        ))
        st.caption(
            f"Rosin-Rammler: Xc = {fit.xc:.1f} {unit}, n = {fit.n:.2f}, R2 = {fit.r_squared:.3f}"
        )
    else:
        st.caption(f"Fewer than the minimum particle count for a reliable Rosin-Rammler fit ({len(kept)} kept).")

    fig.update_layout(
        xaxis_title=f"Particle size ({unit})",
        yaxis_title="Cumulative passing (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Oversize / fines")
    col1, col2 = st.columns(2)
    with col1:
        oversize_threshold = st.number_input(f"Oversize threshold ({unit})", min_value=0.0, value=float(dv["D80"]))
        oversize_pct = oversize_percentage(sizes, weights, oversize_threshold, basis)
        st.metric(f"Oversize (> {oversize_threshold:.0f} {unit}, {basis.value} basis)", f"{oversize_pct:.1f}%")
    with col2:
        fines_threshold = st.number_input(f"Fines threshold ({unit})", min_value=0.0, value=float(dv["D10"]))
        fines_pct = fines_percentage(sizes, weights, fines_threshold, basis)
        st.metric(f"Fines (< {fines_threshold:.0f} {unit}, {basis.value} basis)", f"{fines_pct:.1f}%")

    st.caption(
        "Oversize/fines percentages are reported on a NUMBER or AREA basis, never as a mass or "
        "volume percentage - 2D image area does not translate to 3D mass without an extra, "
        "unvalidated conversion assumption. See docs/algorithms/statistics.md."
    )

    st.divider()
    summary = {
        "particle_count": len(kept),
        "basis": basis.value,
        "unit": unit,
        **dv,
        "oversize_pct": oversize_pct,
        "oversize_threshold": oversize_threshold,
        "fines_pct": fines_pct,
        "fines_threshold": fines_threshold,
    }
    if fit is not None:
        summary["rosin_rammler_xc"] = fit.xc
        summary["rosin_rammler_n"] = fit.n
        summary["rosin_rammler_r2"] = fit.r_squared

    if st.button("Save this analysis to the blast", type="primary"):
        save_analysis(
            image["id"], blast["id"],
            calibration_record["id"] if calibration_record else None,
            particles, summary, oversize_threshold, basis.value,
            segmentation_config=get_seg_config_for_image(image["id"]),
        )
        st.success("Analysis saved. It now feeds Blast Overview and Blast Comparison.")


# ---------------------------------------------------------------------------
# Page: Blast comparison
# ---------------------------------------------------------------------------
def render_comparison() -> None:
    st.title("Blast Comparison")
    st.caption("Compares the full, multi-image fragmentation profile of one blast against another.")

    blasts = list_blasts()
    blasts_with_data = [b for b in blasts if get_latest_analyses_for_blast(b["id"])]
    if len(blasts_with_data) < 2:
        st.info("Save at least one analysis in each of two different blasts first.")
        return

    names = {b["id"]: b["name"] for b in blasts_with_data}
    col_a, col_b = st.columns(2)
    with col_a:
        blast_a_id = st.selectbox("Blast A", list(names.keys()), format_func=lambda i: names[i], key="compare_blast_a")
    with col_b:
        remaining = [bid for bid in names if bid != blast_a_id]
        blast_b_id = st.selectbox("Blast B", remaining, format_func=lambda i: names[i], key="compare_blast_b")

    basis_label = st.radio("Distribution weighting", ["Area (recommended)", "Number"], horizontal=True, key="compare_basis")
    basis = SizeBasis.AREA if basis_label.startswith("Area") else SizeBasis.NUMBER

    agg_a = aggregate_blast_particles(blast_a_id)
    agg_b = aggregate_blast_particles(blast_b_id)
    summary_a = compute_aggregate_summary(agg_a, basis)
    summary_b = compute_aggregate_summary(agg_b, basis)

    if summary_a is None or summary_b is None:
        st.warning("One of the selected blasts has no analyzed, calibrated images to compare.")
        return

    for name, agg in [(names[blast_a_id], agg_a), (names[blast_b_id], agg_b)]:
        if agg.skipped:
            reasons = ", ".join(f"{fname} ({reason})" for fname, reason in agg.skipped)
            st.caption(f"{name}: excluded from the rollup - {reasons}")

    st.subheader("Cumulative passing curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary_a.dist.sizes_sorted, y=summary_a.dist.cumulative_passing_pct,
        mode="lines", name=names[blast_a_id], line=dict(color="#38BDF8", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=summary_b.dist.sizes_sorted, y=summary_b.dist.cumulative_passing_pct,
        mode="lines", name=names[blast_b_id], line=dict(color="#F5F7FA", width=2, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Particle size (mm)", xaxis_type="log",
        yaxis_title="Cumulative passing (%)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Thresholds for the two metrics below")
    col1, col2 = st.columns(2)
    with col1:
        min_target_mm = st.number_input(
            "Minimum target size (mm) - over-fragmentation is material finer than this",
            min_value=0.0, value=float(min(summary_a.d_values["D10"], summary_b.d_values["D10"])),
        )
    with col2:
        crusher_gape_mm = st.number_input(
            "Crusher jaw clearance (mm) - boulders are material coarser than this",
            min_value=0.0, value=float(max(summary_a.d_values["D90"], summary_b.d_values["D90"])),
        )

    def kpi_row(name: str, agg: BlastAggregate, summary: AggregateSummary) -> dict:
        df = agg.combined_df
        sizes = df["equiv_diameter_mm"].to_numpy()
        areas = df["area_mm2"].to_numpy()
        weights = areas if basis is SizeBasis.AREA else None
        return {
            "Blast": name,
            "Images included": len(agg.included),
            "Particle count": summary.particle_count,
            "P20 (mm)": summary.d_values["D20"],
            "P50 (mm)": summary.d_values["D50"],
            "P80 (mm)": summary.d_values["D80"],
            "Uniformity index n": summary.rosin_rammler.n if summary.rosin_rammler else None,
            "Over-fragmentation %": fines_percentage(sizes.tolist(), weights.tolist() if weights is not None else None, min_target_mm, basis),
            "Boulder / oversize %": oversize_percentage(sizes.tolist(), weights.tolist() if weights is not None else None, crusher_gape_mm, basis),
        }

    kpi_df = pd.DataFrame([
        kpi_row(names[blast_a_id], agg_a, summary_a),
        kpi_row(names[blast_b_id], agg_b, summary_b),
    ])
    st.subheader("KPI comparison")
    st.dataframe(kpi_df.set_index("Blast").T, width="stretch")

    st.caption(
        "Uniformity index n is the Rosin-Rammler shape parameter: higher means a narrower, more "
        "uniform size spread. It shows as blank/NaN when a blast has fewer than 30 pooled "
        "particles - not enough for a reliable fit. Over-fragmentation and boulder percentages "
        "use the basis selected above and are not mass or volume fractions. This page reports "
        "the measured differences without judging which blast is 'better' - that call is yours."
    )


# ---------------------------------------------------------------------------
# Page: Export
# ---------------------------------------------------------------------------
def render_export() -> None:
    st.title("Export")
    blast = get_current_blast()
    image = get_current_image()
    particles = get_particles_for_image(image["id"]) if image else None
    labels = get_label_image(image["id"]) if image else None

    if blast is None or image is None or not particles:
        st.info("Run segmentation and load the Results page first, then come back here to export.")
        return

    calibration_record = get_latest_calibration(image["id"])
    calibration = _calibration_from_record(calibration_record) if calibration_record else None

    df = particles_to_dataframe(particles, calibration)
    st.dataframe(df, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV", dataframe_to_csv_bytes(df),
            file_name=f"{blast['name']}_particles.csv", mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download Excel", dataframe_to_excel_bytes(df),
            file_name=f"{blast['name']}_particles.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    image_bgr = load_image_bgr(image["stored_filename"])
    annotated = None
    if image_bgr is not None and labels is not None:
        annotated = draw_annotations(image_bgr, labels, particles, show_boundaries=True, show_ids=True)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Annotated image", width="stretch")
        success, encoded = cv2.imencode(".png", annotated)
        if success:
            st.download_button(
                "Download annotated image (PNG)", encoded.tobytes(),
                file_name=f"{blast['name']}_annotated.png", mime="image/png",
            )

    st.divider()
    st.subheader("PDF report")
    kept = [p for p in particles if p.kept]
    warnings = list(image.get("quality_warnings") or [])
    if calibration is None or not calibration.is_valid:
        warnings.append("No valid scale calibration - all sizes in this report are in pixels, not millimetres.")
    if calibration and calibration.consistency_warning:
        warnings.append(calibration.consistency_warning)

    summary = {
        "particles_detected": len(particles),
        "particles_kept": len(kept),
        "particles_removed": len(particles) - len(kept),
    }

    if st.button("Generate PDF report"):
        pdf_bytes = build_pdf_report(
            blast_name=blast["name"],
            image_name=image["original_filename"],
            summary=summary,
            warnings=warnings,
            annotated_image_bgr=annotated,
        )
        st.download_button(
            "Download PDF", pdf_bytes,
            file_name=f"{blast['name']}_report.pdf", mime="application/pdf",
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="FragAnalyst", page_icon=":black_circle:", layout="wide")

CUSTOM_CSS = """
<style>
[data-testid="stSidebar"] {
    border-right: 1px solid #1c2028;
}
.frag-brand {
    font-family: monospace;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    color: #7DD3FC;
    padding: 0.25rem 0 1rem 0;
}
.frag-brand strong {
    color: #FFFFFF;
}
[data-testid="stMetricValue"] {
    color: #38BDF8;
}
hr {
    border-color: #1c2028;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

init_db()
init_state()

PAGES = {
    "Blasts": render_blasts,
    "Upload": render_upload,
    "Calibrate": render_calibrate,
    "Segment & Review": render_segment,
    "Results": render_results,
    "Blast Overview": render_overview,
    "Blast Comparison": render_comparison,
    "Export": render_export,
}

with st.sidebar:
    st.markdown('<div class="frag-brand">FRAG<strong>ANALYST</strong></div>', unsafe_allow_html=True)
    choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed", key="nav_radio")

    active_blast = get_blast(st.session_state["current_blast_id"]) if st.session_state.get("current_blast_id") else None
    st.divider()
    st.caption("Active blast")
    st.write(active_blast["name"] if active_blast else "none selected")

    if active_blast:
        blast_images = list_images(active_blast["id"])
        if blast_images:
            st.caption("Images in Blast Overview")
            selected = set(st.session_state["selected_image_ids"])
            for img in blast_images:
                checked = st.checkbox(
                    img["original_filename"], value=img["id"] in selected,
                    key=f"sidebar_include_{img['id']}",
                )
                if checked:
                    selected.add(img["id"])
                else:
                    selected.discard(img["id"])
            st.session_state["selected_image_ids"] = list(selected)

    current_image = get_current_image()
    if current_image:
        st.divider()
        st.caption("Editing")
        st.write(current_image["original_filename"])

PAGES[choice]()
