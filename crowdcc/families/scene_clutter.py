"""Person, object and texture distractors."""
from __future__ import annotations
import cv2
import numpy as np
from crowdcc.base import Context, GTTransform, LevelParams
from crowdcc.crowd_spacing import SPACING_OVER_HEAD_DIAMETER, resolve_spacing_px
from crowdcc.image_ops import (
    assert_shape_preserved,
    clip01,
    gaussian_blur,
    linear_to_srgb,
    srgb_to_linear,
    to_float01,
)
from crowdcc.registry import register
from crowdcc.scene_ops import (
    OCCLUSION_ALPHA,
    _fill_polygons,
    person_silhouette,
    points_inside_mask,
)


# crowd_like_distractors
DISTRACTOR_MIN_HEIGHT_PX = 12.0
DISTRACTOR_MAX_HEIGHT_FRAC = 0.5
DISTRACTOR_PLACEMENT_TRIES = 60
DISTRACTOR_ASPECT = 0.55
DECOY_MAX_PAIR_OVERLAP = 0.15
PERSON_CROP_HEIGHT_HEADS = 3.5
PERSON_CROP_HEAD_FROM_TOP = 0.18
PERSON_CROP_TRIES = 12
COUNT_REFERENCE_MEGAPIXELS = 0.7
COUNT_SCALE_RANGE = (1.0, 2.5)
DEPICTION_CUES = ("poster", "led_screen", "reflection", "pictogram")
POSTER_CONTRAST = 0.55
POSTER_SATURATION = 0.35
POSTER_BORDER_FRAC = 0.06
LED_SCANLINE_PERIOD_PX = 4
LED_SCANLINE_DEPTH = 0.35
REFLECTION_CONTRAST = 0.40
REFLECTION_BLUR_FRAC = 0.004
PICTOGRAM_COLOR = np.array([0.05, 0.06, 0.09], dtype=np.float32)


# object_distractors, texture_distractors
OBJECT_LIGHT_DIR = np.array([-0.42, -0.55, 0.72], dtype=np.float32)
OBJECT_AMBIENT = 0.34
OBJECT_SPECULAR_EXPONENT = 48.0
OBJECT_EDGE_FEATHER_PX = 0.7
OBJECT_REFERENCE_LUMA = 0.25
OBJECT_LUMA_GAIN_RANGE = (0.40, 1.60)
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# object_distractors
OBJECT_CUES = ("balloon", "produce", "helmet", "ball")
OBJECT_ASPECT = 1.0
OBJECT_MIN_DIAMETER_PX = 6.0
OBJECT_MAX_DIAMETER_FRAC = 0.12
OBJECT_SCATTER_SPACINGS = 1.6
OBJECT_MIN_COUNT = 20
OBJECT_MAX_COUNT = 600
BALLOON_ALBEDOS = np.array([
    [0.62, 0.09, 0.10], [0.10, 0.22, 0.58], [0.72, 0.55, 0.06],
    [0.09, 0.36, 0.16], [0.63, 0.30, 0.55], [0.78, 0.78, 0.80],
], dtype=np.float32)
BALLOON_ASPECT = 0.80
BALLOON_BODY_FRAC = 0.82
BALLOON_GLOSS = 0.55
PRODUCE_ALBEDOS = np.array([
    [0.20, 0.32, 0.09], [0.64, 0.29, 0.04], [0.55, 0.10, 0.08],
    [0.35, 0.44, 0.14], [0.70, 0.62, 0.20], [0.26, 0.20, 0.11],
], dtype=np.float32)
PRODUCE_GLOSS = 0.12
PRODUCE_MOTTLE = 0.16
HELMET_ALBEDOS = np.array([
    [0.72, 0.66, 0.10], [0.70, 0.72, 0.74], [0.10, 0.12, 0.16],
    [0.55, 0.16, 0.07], [0.12, 0.30, 0.50],
], dtype=np.float32)
HELMET_GLOSS = 0.70
HELMET_VISOR_DARKEN = 0.45
BALL_ALBEDOS = np.array([
    [0.78, 0.78, 0.79], [0.66, 0.32, 0.06], [0.16, 0.17, 0.20],
    [0.70, 0.60, 0.10],
], dtype=np.float32)
BALL_GLOSS = 0.40
BALL_SEAM_DARKEN = 0.55


# texture_distractors
TEXTURE_CUES = ("gravel", "foliage", "window_grid")
TEXTURE_MIN_PITCH_PX = 3.5
TEXTURE_PATCH_RADIUS_PITCHES = (2.5, 7.0)
TEXTURE_PLACEMENT_TRIES = 600
TEXTURE_EDGE_FEATHER_FRAC = 0.18
TEXTURE_BLOB_VARIANTS = 24
TEXTURE_TILE_PITCHES = 24.0
TEXTURE_TILE_MAX_PX = 384
TEXTURE_GT_MARGIN_HEAD_RADII = 2.0
TEXTURE_BLOB_WOBBLE = 0.28
TEXTURE_OPACITY = 0.92
GRAVEL_ALBEDOS = np.array([
    [0.30, 0.28, 0.25], [0.42, 0.39, 0.35], [0.22, 0.21, 0.20],
    [0.36, 0.32, 0.27], [0.50, 0.47, 0.44],
], dtype=np.float32)
FOLIAGE_ALBEDOS = np.array([
    [0.10, 0.21, 0.07], [0.16, 0.30, 0.10], [0.07, 0.14, 0.05],
    [0.21, 0.33, 0.12], [0.13, 0.24, 0.14],
], dtype=np.float32)
WINDOW_MULLION = np.array([0.34, 0.33, 0.31], dtype=np.float32)
WINDOW_GLASS = np.array([0.06, 0.08, 0.11], dtype=np.float32)
WINDOW_MULLION_FRAC = 0.22


def person_patch(linear: np.ndarray, points: np.ndarray | None, spacing: float,
                 box_hw: tuple[int, int], rng: np.random.Generator) -> np.ndarray | None:
    """Crop a labelled person patch; return None if unavailable."""
    if points is None or len(points) == 0:
        return None
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    head = spacing / SPACING_OVER_HEAD_DIAMETER
    crop_h = max(PERSON_CROP_HEIGHT_HEADS * head, 4.0)
    crop_w = max(crop_h * DISTRACTOR_ASPECT, 3.0)
    height, width = linear.shape[:2]
    for _ in range(PERSON_CROP_TRIES):
        cx, cy = pts[int(rng.integers(len(pts)))]
        x0 = int(round(cx - crop_w / 2.0))
        y0 = int(round(cy - PERSON_CROP_HEAD_FROM_TOP * crop_h))
        x1, y1 = x0 + int(round(crop_w)), y0 + int(round(crop_h))
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue
        crop = linear[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        return cv2.resize(crop, (box_hw[1], box_hw[0]), interpolation=cv2.INTER_LINEAR)
    return None


def _apply_depiction_cue(
    patch: np.ndarray, alpha: np.ndarray, cue: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply depiction styling to a person patch."""
    if cue == "pictogram":
        flat = np.broadcast_to(PICTOGRAM_COLOR, patch.shape).copy()
        return flat, alpha
    if cue == "poster":
        grey = patch.mean(axis=2, keepdims=True)
        flat = grey + POSTER_SATURATION * (patch - grey)
        flat = 0.5 + POSTER_CONTRAST * (flat - 0.5)
        board = np.ones(alpha.shape, dtype=np.float32)
        border = max(int(POSTER_BORDER_FRAC * min(alpha.shape)), 1)
        board[:border] = board[-border:] = board[:, :border] = board[:, -border:] = 0.0
        printed = np.where(alpha[..., None] > 0.5, flat, 0.5 * np.ones_like(flat))
        return printed, np.maximum(alpha, board)
    if cue == "led_screen":
        rows = np.arange(patch.shape[0])[:, None, None]
        scan = 1.0 - LED_SCANLINE_DEPTH * (rows % LED_SCANLINE_PERIOD_PX
                                           < LED_SCANLINE_PERIOD_PX // 2)
        panel = np.where(alpha[..., None] > 0.5, patch, 0.02 * np.ones_like(patch))
        return panel * scan, np.ones(alpha.shape, dtype=np.float32)
    if cue == "reflection":
        mirrored = patch[:, ::-1]
        faded = 0.5 + REFLECTION_CONTRAST * (mirrored - 0.5)
        blur = REFLECTION_BLUR_FRAC * min(patch.shape[:2]) * (1.0 + rng.random())
        return gaussian_blur(faded, max(blur, 0.6)), alpha
    raise ValueError(f"unknown depiction cue {cue!r}; known: {DEPICTION_CUES}")


def _distractor_box(centre_xy: tuple[float, float], height_px: float,
                    aspect: float = DISTRACTOR_ASPECT) -> tuple[int, int, int, int]:
    """Return a bottom-anchored distractor box."""
    box_h = int(round(height_px))
    box_w = int(round(height_px * aspect))
    x0 = int(round(centre_xy[0] - box_w / 2))
    y0 = int(round(centre_xy[1] - box_h))
    return x0, y0, x0 + box_w, y0 + box_h


def clearance_map(shape_hw: tuple[int, int], points: np.ndarray | None,
                  margin: float) -> np.ndarray:
    """Return distance to the nearest head minus margin."""
    occupied = np.ones(shape_hw, dtype=np.uint8)
    if points is not None and len(points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, shape_hw[1] - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, shape_hw[0] - 1)
        occupied[ys, xs] = 0
    return cv2.distanceTransform(occupied, cv2.DIST_L2, 5) - float(margin)


def _too_close(box: tuple[int, int, int, int],
               placed: list[tuple[int, int, int, int]],
               max_overlap: float = DECOY_MAX_PAIR_OVERLAP) -> bool:
    """Check overlap relative to the smaller box area."""
    ax0, ay0, ax1, ay1 = box
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1)
    for bx0, by0, bx1, by1 in placed:
        w = min(ax1, bx1) - max(ax0, bx0)
        h = min(ay1, by1) - max(ay0, by0)
        if w <= 0 or h <= 0:
            continue
        if (w * h) / min(area_a, max((bx1 - bx0) * (by1 - by0), 1)) > max_overlap:
            return True
    return False


def _paste_distractor(
    canvas: np.ndarray, centre_xy: tuple[float, float], height_px: float,
    cue: str, rng: np.random.Generator, texture: np.ndarray | None = None,
    coverage: np.ndarray | None = None,
) -> bool:
    """Composite a person decoy and update coverage."""
    box_h = int(round(height_px))
    box_w = int(round(height_px * DISTRACTOR_ASPECT))
    if box_h < 4 or box_w < 3:
        return False
    x0, y0, x1, y1 = _distractor_box(centre_xy, height_px)
    if x0 < 0 or y0 < 0 or x1 > canvas.shape[1] or y1 > canvas.shape[0]:
        return False
    alpha = np.zeros((box_h, box_w), dtype=np.float32)
    _fill_polygons(alpha, person_silhouette(box_h * 0.98, rng),
                   (box_w / 2.0, float(box_h) - 1.0))
    if float(alpha.sum()) <= 0:
        return False
# Fall back to underlying pixels.
    patch = (canvas[y0:y1, x0:x1].copy() if texture is None
             else np.ascontiguousarray(texture[:box_h, :box_w], dtype=np.float32))
    filled, alpha = _apply_depiction_cue(patch, alpha, cue, rng)
    alpha = clip01(gaussian_blur(alpha, 0.7))[..., None]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - alpha) + filled * alpha
    if coverage is not None:
        np.maximum(coverage[y0:y1, x0:x1], alpha[..., 0],
                   out=coverage[y0:y1, x0:x1])
    return True


def _size_scaled_count(base: int, height: int, width: int) -> int:
    """Scale count by image area with bounded square-root scaling."""
    megapixels = (height * width) / 1e6
    scale = float(np.clip(np.sqrt(megapixels / COUNT_REFERENCE_MEGAPIXELS),
                          *COUNT_SCALE_RANGE))
    return max(1, int(round(base * scale)))


def _verify_crowd_like_distractors(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Check decoy placement and report GT coverage."""
    assert_shape_preserved(src, dst, "crowd_like_distractors")
    touched = np.abs(to_float01(dst) - to_float01(src)).max(axis=2) > 2 / 255
    changed = float(touched.mean())
    if changed <= 0.0:
        raise AssertionError("crowd_like_distractors: no decoys placed")
    n_points = 0 if ctx.gt_points is None else len(ctx.gt_points)
    covered = int(points_inside_mask(ctx.gt_points, touched).sum())
    return {"distractor_area_fraction": changed,
            "n_gt_heads_covered": float(covered),
            "gt_heads_covered_fraction": float(covered) / n_points if n_points else 0.0}


@register(
    name="crowd_like_distractors",
    family="scene_clutter",
    gt_impact="altering",
    needs=["gt_points"],
    levels={
        "light": LevelParams(count=5, height_spacing_frac=3.5,
                             cue_mix=DEPICTION_CUES),
        "moderate": LevelParams(count=15, height_spacing_frac=3.0,
                                cue_mix=DEPICTION_CUES),
        "high": LevelParams(count=30, height_spacing_frac=2.5,
                            cue_mix=DEPICTION_CUES),
    },
    verify=_verify_crowd_like_distractors,
)
def crowd_like_distractors(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, GTTransform]:
    """Add person depictions and mask covered GT heads."""
    linear = srgb_to_linear(to_float01(img))
    height, width = img.shape[:2]
    spacing = resolve_spacing_px(ctx.gt_points, (height, width))
    target_h = float(np.clip(float(p.height_spacing_frac) * spacing,
                             DISTRACTOR_MIN_HEIGHT_PX,
                             DISTRACTOR_MAX_HEIGHT_FRAC * min(height, width)))
    rng = ctx.rng_image
    cues = [c for c in p.cue_mix if c in DEPICTION_CUES]
    canvas = linear.copy()
    coverage = np.zeros((height, width), dtype=np.float32)
    placed, used = 0, {c: 0 for c in DEPICTION_CUES}
    footprints: list[tuple[int, int, int, int]] = []
    n_decoys = _size_scaled_count(int(p.count),
                                  height, width)
    for _ in range(n_decoys):
        cue = cues[int(rng.integers(len(cues)))]
        this_h = target_h * (0.7 + 0.6 * rng.random())
        centre = None
        for _try in range(DISTRACTOR_PLACEMENT_TRIES):
            candidate = _object_placement((height, width), ctx.gt_points, this_h,
                                          spacing, rng, aspect=DISTRACTOR_ASPECT)
            if candidate is None:
                break
            if not _too_close(_distractor_box(candidate, this_h), footprints):
                centre = candidate
                break
        if centre is None:
            continue
        box_h = int(round(this_h))
        texture = person_patch(linear, ctx.gt_points, spacing,
                               (box_h, max(int(round(this_h * DISTRACTOR_ASPECT)), 1)), rng)
        if _paste_distractor(canvas, centre, this_h, cue, rng, texture, coverage):
            placed += 1
            used[cue] += 1
            footprints.append(_distractor_box(centre, this_h))
    if placed == 0:
        raise AssertionError(
            f"crowd_like_distractors: 0/{n_decoys} decoys placed ({width}x{height}, "
            f"spacing {spacing:.1f} px, height {target_h:.1f} px)"
        )
    return linear_to_srgb(clip01(canvas)), GTTransform(
        kind="occluded_mask", output_hw=(height, width),
        occluded_mask=points_inside_mask(ctx.gt_points, coverage > OCCLUSION_ALPHA),
        drop_occluded=True,
    )


def _sphere_normals(shape_hw: tuple[int, int] | int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return ellipsoid normals and silhouette."""
    height, width = ((shape_hw, shape_hw) if isinstance(shape_hw, int)
                     else shape_hw)
    ry, rx = (height - 1) / 2.0, (width - 1) / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx - rx) / max(rx, 1e-6)
    ny = (yy - ry) / max(ry, 1e-6)
    rr = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))
    return np.stack([nx, ny, nz], axis=-1), rr <= 1.0


def _shade_sphere(normals: np.ndarray, albedo: np.ndarray, gloss: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Apply Lambert–Blinn-Phong shading in linear light."""
    light = OBJECT_LIGHT_DIR + rng.normal(0.0, 0.06, 3).astype(np.float32)
    light /= max(float(np.linalg.norm(light)), 1e-6)
    lambert = np.clip((normals * light).sum(-1), 0.0, 1.0)
# View direction is +z.
    half = light + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half /= max(float(np.linalg.norm(half)), 1e-6)
    spec = np.clip((normals * half).sum(-1), 0.0, 1.0) ** OBJECT_SPECULAR_EXPONENT
    shade = OBJECT_AMBIENT + (1.0 - OBJECT_AMBIENT) * lambert
    return albedo[None, None, :] * shade[..., None] + gloss * spec[..., None]


def _pick_albedo(table: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample albedo with brightness jitter."""
    base = table[int(rng.integers(len(table)))]
    return np.clip(base * (0.82 + 0.36 * rng.random()), 0.01, 1.0).astype(np.float32)


def _object_sprite(cue: str, size: int, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Render a shaded object sprite in linear light."""
    normals, disc = _sphere_normals(size)
    alpha = disc.astype(np.float32)
    if cue == "balloon":
        body_h = max(int(round(BALLOON_BODY_FRAC * size)), 3)
        body_w = max(int(round(BALLOON_ASPECT * size)), 3)
        b_norm, b_disc = _sphere_normals((body_h, body_w))
        body = _shade_sphere(b_norm, _pick_albedo(BALLOON_ALBEDOS, rng),
                             BALLOON_GLOSS, rng)
        rgb = np.zeros((size, size, 3), dtype=np.float32)
        alpha = np.zeros((size, size), dtype=np.float32)
        x0 = (size - body_w) // 2
        rgb[:body_h, x0:x0 + body_w] = body
        alpha[:body_h, x0:x0 + body_w] = b_disc.astype(np.float32)
        cx = size // 2
        neck_h = max(int(round(0.07 * size)), 1)
        neck = np.array([[cx - max(size // 22, 1), body_h - 1],
                         [cx + max(size // 22, 1), body_h - 1],
                         [cx, min(body_h + neck_h, size - 1)]], dtype=np.int32)
        cv2.fillPoly(alpha, [neck], 1.0, lineType=cv2.LINE_AA)
        knot = body[max(body_h - 2, 0), body_w // 2] * 0.55
        rgb[body_h - 1:, :] = knot
        cv2.line(alpha, (cx, body_h + neck_h), (cx, size - 1), 0.65,
                 max(size // 40, 1), lineType=cv2.LINE_AA)
        return rgb, alpha
    if cue == "produce":
        rgb = _shade_sphere(normals, _pick_albedo(PRODUCE_ALBEDOS, rng),
                            PRODUCE_GLOSS, rng)
        blotch = rng.random((max(size // 4, 2), max(size // 4, 2))).astype(np.float32)
        blotch = cv2.resize(blotch, (size, size), interpolation=cv2.INTER_CUBIC)
        rgb *= (1.0 - PRODUCE_MOTTLE + 2.0 * PRODUCE_MOTTLE * blotch)[..., None]
        return rgb, alpha
    if cue == "helmet":
        cut = 0.22 + 0.14 * rng.random()
        alpha = (disc & (normals[..., 1] <= cut)).astype(np.float32)
        rgb = _shade_sphere(normals, _pick_albedo(HELMET_ALBEDOS, rng),
                            HELMET_GLOSS, rng)
        brim = np.zeros((size, size), dtype=np.float32)
        row = int(round((cut + 1.0) / 2.0 * (size - 1)))
        cv2.ellipse(brim, (size // 2, row), (max(size // 2 - 1, 1), max(size // 12, 1)),
                    0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
        alpha = np.maximum(alpha, brim)
        band = (normals[..., 1] > cut - 0.45) & (normals[..., 1] <= cut)
        rgb[band] *= 1.0 - HELMET_VISOR_DARKEN
        return rgb, alpha
    if cue == "ball":
        rgb = _shade_sphere(normals, _pick_albedo(BALL_ALBEDOS, rng),
                            BALL_GLOSS, rng)
        seam = np.zeros((size, size), dtype=np.float32)
        cv2.ellipse(seam, (size // 2, size // 2),
                    (max(size // 2 - 1, 1), max(int(round(0.30 * size)), 1)),
                    float(rng.random() * 180.0), 0, 360, 1.0,
                    max(size // 16, 1), lineType=cv2.LINE_AA)
        rgb *= (1.0 - BALL_SEAM_DARKEN * seam * disc)[..., None]
        return rgb, alpha
    raise ValueError(f"unknown object cue {cue!r}; known: {OBJECT_CUES}")


def _object_placement(shape_hw: tuple[int, int], points: np.ndarray | None,
                      size_px: float, spacing: float, rng: np.random.Generator,
                      aspect: float = OBJECT_ASPECT) -> tuple[float, float] | None:
    """Sample a head-relative position within image bounds."""
    height, width = shape_hw
    box_w = float(round(size_px * aspect))
    box_h = float(round(size_px))
    if box_w + 1.0 >= width or box_h + 1.0 >= height:
        return None
    lo_x, hi_x = 0.5 * box_w + 0.5, width - 0.5 * box_w - 0.5
    lo_y, hi_y = box_h + 0.5, height - 0.5
    if points is None or len(points) == 0:
        return float(rng.uniform(lo_x, hi_x)), float(rng.uniform(lo_y, hi_y))
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    anchor = pts[int(rng.integers(len(pts)))]
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = float(rng.uniform(0.0, OBJECT_SCATTER_SPACINGS * max(spacing, 1e-6)))
    return (float(np.clip(anchor[0] + radius * np.cos(angle), lo_x, hi_x)),
            float(np.clip(anchor[1] + radius * np.sin(angle), lo_y, hi_y)))


def _paste_object(canvas: np.ndarray, centre_xy: tuple[float, float],
                  size_px: float, cue: str, rng: np.random.Generator,
                  coverage: np.ndarray | None = None) -> bool:
    """Composite an object with local brightness matching."""
    x0, y0, x1, y1 = _distractor_box(centre_xy, size_px, OBJECT_ASPECT)
    if x0 < 0 or y0 < 0 or x1 > canvas.shape[1] or y1 > canvas.shape[0]:
        return False
    size = y1 - y0
    if size < 4 or (x1 - x0) != size:
        return False
    rgb, alpha = _object_sprite(cue, size, rng)
    if float(alpha.sum()) <= 0:
        return False
    under = canvas[y0:y1, x0:x1]
    local = float((under * LUMA_WEIGHTS).sum(-1).mean())
    rgb = rgb * float(np.clip(local / OBJECT_REFERENCE_LUMA,
                              *OBJECT_LUMA_GAIN_RANGE))
    alpha = clip01(gaussian_blur(alpha, OBJECT_EDGE_FEATHER_PX))[..., None]
    canvas[y0:y1, x0:x1] = under * (1.0 - alpha) + rgb * alpha
    if coverage is not None:
        np.maximum(coverage[y0:y1, x0:x1], alpha[..., 0],
                   out=coverage[y0:y1, x0:x1])
    return True


def _verify_object_distractors(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Check object placement and report GT coverage."""
    assert_shape_preserved(src, dst, "object_distractors")
    touched = np.abs(to_float01(dst) - to_float01(src)).max(axis=2) > 2 / 255
    changed = float(touched.mean())
    if changed <= 0.0:
        raise AssertionError("object_distractors: no decoys placed")
    n_points = 0 if ctx.gt_points is None else len(ctx.gt_points)
    covered = int(points_inside_mask(ctx.gt_points, touched).sum())
    return {"object_area_fraction": changed,
            "n_gt_heads_covered": float(covered),
            "gt_heads_covered_fraction": float(covered) / n_points if n_points else 0.0}


@register(
    name="object_distractors",
    family="scene_clutter",
    gt_impact="altering",
    needs=["gt_points"],
    levels={
        "light": LevelParams(count_gt_frac=0.20, size_head_frac=(0.8, 2.0),
                             cue_mix=OBJECT_CUES),
        "moderate": LevelParams(count_gt_frac=0.40, size_head_frac=(0.8, 2.0),
                                cue_mix=OBJECT_CUES),
        "high": LevelParams(count_gt_frac=0.60, size_head_frac=(0.8, 2.0),
                            cue_mix=OBJECT_CUES),
    },
    verify=_verify_object_distractors,
)
def object_distractors(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, GTTransform]:
    """Add head-scale objects and mask covered GT heads."""
    linear = srgb_to_linear(to_float01(img))
    height, width = img.shape[:2]
    spacing = resolve_spacing_px(ctx.gt_points, (height, width))
    head_px = spacing / SPACING_OVER_HEAD_DIAMETER
    size_range = p.size_head_frac
    lo, hi = ((float(size_range[0]), float(size_range[-1]))
              if isinstance(size_range, (tuple, list))
              else (float(size_range), float(size_range)))
    max_px = OBJECT_MAX_DIAMETER_FRAC * min(height, width)
    rng = ctx.rng_image
    cues = [c for c in p.cue_mix if c in OBJECT_CUES]
    if not cues:
        raise ValueError(f"cue_mix {p.cue_mix!r} has no known object cue")
    canvas = linear.copy()
    coverage = np.zeros((height, width), dtype=np.float32)
    n_gt = 0 if ctx.gt_points is None else len(ctx.gt_points)
    wanted = int(np.clip(
        round(float(p.count_gt_frac) * n_gt),
        OBJECT_MIN_COUNT, OBJECT_MAX_COUNT))
    placed, used, sizes = 0, {c: 0 for c in OBJECT_CUES}, []
    for _ in range(wanted):
        cue = cues[int(rng.integers(len(cues)))]
        this = float(np.clip((lo + (hi - lo) * rng.random()) * head_px,
                             OBJECT_MIN_DIAMETER_PX, max_px))
        centre = _object_placement((height, width), ctx.gt_points, this, spacing, rng)
        if centre is None:
            continue
        if _paste_object(canvas, centre, this, cue, rng, coverage):
            placed += 1
            used[cue] += 1
            sizes.append(this)
    if placed == 0:
        raise AssertionError(
            f"object_distractors: 0/{wanted} decoys placed ({width}x{height}, "
            f"{n_gt} heads, head {head_px:.1f} px)"
        )
    return linear_to_srgb(clip01(canvas)), GTTransform(
        kind="occluded_mask", output_hw=(height, width),
        occluded_mask=points_inside_mask(ctx.gt_points, coverage > OCCLUSION_ALPHA),
        drop_occluded=True,
    )


def _texture_tile(cue: str, size: int, pitch: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Render a linear-light texture tile."""
    if cue == "window_grid":
        tile = np.empty((size, size, 3), dtype=np.float32)
        tile[:] = WINDOW_MULLION
        step = max(int(round(pitch)), 2)
        inset = max(int(round(WINDOW_MULLION_FRAC * step)), 1)
        for y0 in range(0, size, step):
            for x0 in range(0, size, step):
                ya, yb = y0 + inset, min(y0 + step - inset, size)
                xa, xb = x0 + inset, min(x0 + step - inset, size)
                if yb <= ya or xb <= xa:
                    continue
                lit = 1.0 + 2.4 * float(rng.random() < 0.18) * float(rng.random())
                tile[ya:yb, xa:xb] = WINDOW_GLASS * lit
        return tile
    if cue not in ("gravel", "foliage"):
        raise ValueError(f"unknown texture cue {cue!r}; known: {TEXTURE_CUES}")
    table = GRAVEL_ALBEDOS if cue == "gravel" else FOLIAGE_ALBEDOS
    tile = np.empty((size, size, 3), dtype=np.float32)
    tile[:] = table[0] * 0.55
    bank: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(TEXTURE_BLOB_VARIANTS):
        scale = 0.65 + 0.7 * rng.random()
        if cue == "gravel":
            bh = bw = max(int(round(pitch * scale)), 2)
        else:
            bh = max(int(round(pitch * scale)), 2)
            bw = max(int(round(pitch * scale * (0.35 + 0.35 * rng.random()))), 2)
        norm, mask = _sphere_normals((bh, bw))
        rgb = _shade_sphere(norm, _pick_albedo(table, rng),
                            0.06 if cue == "gravel" else 0.10, rng)
        if cue == "foliage":
            rot = cv2.getRotationMatrix2D((bw / 2.0, bh / 2.0),
                                          float(rng.random() * 180.0), 1.0)
            rgb = cv2.warpAffine(rgb, rot, (bw, bh), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask.astype(np.float32), rot, (bw, bh),
                                  flags=cv2.INTER_LINEAR) > 0.5
        bank.append((rgb, mask))
    radius = max(pitch / 2.0, 1.5)
# Overlap texture blobs.
    count = int(np.clip(3.0 * size * size / (np.pi * radius ** 2), 8, 20000))
    picks = rng.integers(0, len(bank), size=count)
    cys = rng.integers(0, size, size=count)
    cxs = rng.integers(0, size, size=count)
    for i in range(count):
        rgb, mask = bank[int(picks[i])]
        bh, bw = mask.shape
        y0, x0 = int(cys[i]) - bh // 2, int(cxs[i]) - bw // 2
        ya, xa = max(y0, 0), max(x0, 0)
        yb, xb = min(y0 + bh, size), min(x0 + bw, size)
        if yb <= ya or xb <= xa:
            continue
        sub = mask[ya - y0:yb - y0, xa - x0:xb - x0]
        tile[ya:yb, xa:xb][sub] = rgb[ya - y0:yb - y0, xa - x0:xb - x0][sub]
    return tile


def _texture_field(cue: str, shape_hw: tuple[int, int], pitch: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Tile texture with random flips except for window grids."""
    height, width = shape_hw
    size = int(np.clip(round(TEXTURE_TILE_PITCHES * pitch), 64, TEXTURE_TILE_MAX_PX))
    tile = _texture_tile(cue, size, pitch, rng)
    field = np.empty((height, width, 3), dtype=np.float32)
    for y0 in range(0, height, size):
        for x0 in range(0, width, size):
            block = tile
            if cue != "window_grid":
                if rng.random() < 0.5:
                    block = block[::-1]
                if rng.random() < 0.5:
                    block = block[:, ::-1]
            yb, xb = min(y0 + size, height), min(x0 + size, width)
            field[y0:yb, x0:xb] = block[:yb - y0, :xb - x0]
    return field


def _texture_patch_mask(shape_hw: tuple[int, int], clearance: np.ndarray,
                        pitch: float, coverage: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Generate texture patches in head-free regions."""
    height, width = shape_hw
    allowed = clearance > 0.0
    if not allowed.any():
        return np.zeros(shape_hw, dtype=np.float32), 0
    ys, xs = np.nonzero(allowed)
    mask = np.zeros(shape_hw, dtype=np.float32)
    lo, hi = TEXTURE_PATCH_RADIUS_PITCHES
    target_px = coverage * height * width
    covered, placed = 0.0, 0
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    for _ in range(TEXTURE_PLACEMENT_TRIES):
        if covered >= target_px:
            break
        pick = int(rng.integers(len(ys)))
        cy, cx = int(ys[pick]), int(xs[pick])
        radius = pitch * (lo + (hi - lo) * rng.random())
        radius = min(radius, float(np.sqrt(max(target_px - covered, 0.0) / np.pi)))
        if radius < 1.0:
            break
        wobble = 1.0 + TEXTURE_BLOB_WOBBLE * rng.normal(0.0, 1.0, angles.shape)
        span = int(np.ceil(radius * (1.0 + 3.0 * TEXTURE_BLOB_WOBBLE))) + 2
        y0, x0 = max(cy - span, 0), max(cx - span, 0)
        y1, x1 = min(cy + span, height), min(cx + span, width)
        if y1 <= y0 or x1 <= x0:
            continue
        poly = np.column_stack([cx - x0 + radius * wobble * np.cos(angles),
                                cy - y0 + radius * wobble * np.sin(angles)])
        blob = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        cv2.fillPoly(blob, [np.round(poly).astype(np.int32)], 1.0,
                     lineType=cv2.LINE_AA)
        blob *= allowed[y0:y1, x0:x1]
        roi = mask[y0:y1, x0:x1]
        gained = float(((blob > 0) & (roi <= 0)).sum())
        if gained <= 0.0:
            continue
        np.maximum(roi, blob, out=roi)
        covered += gained
        placed += 1
    feather = max(TEXTURE_EDGE_FEATHER_FRAC * pitch, 0.8)
    return clip01(gaussian_blur(mask, feather)) * allowed, placed


def _verify_texture_distractors(
    src: np.ndarray, dst: np.ndarray, p: LevelParams, ctx: Context
) -> dict[str, float]:
    """Check texture placement without head occlusion."""
    assert_shape_preserved(src, dst, "texture_distractors")
    touched = np.abs(to_float01(dst) - to_float01(src)).max(axis=2) > 2 / 255
    changed = float(touched.mean())
    if changed <= 0.0:
        raise AssertionError("texture_distractors: no texture placed")
    covered = int(points_inside_mask(ctx.gt_points, touched).sum())
    if covered:
        raise AssertionError(f"texture_distractors: {covered} labelled head(s) covered")
    return {"texture_area_fraction": changed,
            "n_gt_heads_covered": float(covered)}


@register(
    name="texture_distractors",
    family="scene_clutter",
    gt_impact="preserving",
    needs=["gt_points"],
    levels={
        # Coverage is bounded by head clearance.
        "light": LevelParams(coverage=0.15, pitch_head_frac=1.0,
                             cue_mix=TEXTURE_CUES),
        "moderate": LevelParams(coverage=0.28, pitch_head_frac=1.0,
                                cue_mix=TEXTURE_CUES),
        "high": LevelParams(coverage=0.45, pitch_head_frac=1.0,
                            cue_mix=TEXTURE_CUES),
    },
    verify=_verify_texture_distractors,
)
def texture_distractors(
    img: np.ndarray, p: LevelParams, ctx: Context
) -> tuple[np.ndarray, dict[str, float]]:
    """Paint head-scale textures outside labelled heads."""
    linear = srgb_to_linear(to_float01(img))
    height, width = img.shape[:2]
    spacing = resolve_spacing_px(ctx.gt_points, (height, width))
    head_px = spacing / SPACING_OVER_HEAD_DIAMETER
    pitch = max(float(p.pitch_head_frac) * head_px,
                TEXTURE_MIN_PITCH_PX)
    rng = ctx.rng_image
    cues = [c for c in p.cue_mix if c in TEXTURE_CUES]
    if not cues:
        raise ValueError(f"cue_mix {p.cue_mix!r} has no known texture cue")
    margin = TEXTURE_GT_MARGIN_HEAD_RADII * (head_px / 2.0)
    clearance = clearance_map((height, width), ctx.gt_points, margin)
    coverage = float(p.coverage)
    canvas = linear.copy()
    used = {c: 0.0 for c in TEXTURE_CUES}
    total = np.zeros((height, width), dtype=np.float32)
    for cue in cues:
        mask, _ = _texture_patch_mask((height, width), clearance, pitch,
                                      coverage / len(cues), rng)
        if float(mask.max()) <= 0.0:
            continue
        field = _texture_field(cue, (height, width), pitch, rng)
        local = float((linear * LUMA_WEIGHTS).sum(-1).mean())
        field = field * float(np.clip(local / OBJECT_REFERENCE_LUMA,
                                      *OBJECT_LUMA_GAIN_RANGE))
        alpha = (mask * TEXTURE_OPACITY)[..., None]
        canvas = canvas * (1.0 - alpha) + field * alpha
        used[cue] = float((mask > 0.5).mean())
        total = np.maximum(total, mask)
    return linear_to_srgb(clip01(canvas)), {
        "texture_coverage_requested": coverage,
        "texture_coverage_achieved": float((total > 0.5).mean()),
        "head_free_fraction": float((clearance > 0.0).mean()),
        "texture_pitch_px": pitch,
        "head_diameter_px": head_px,
        "crowd_spacing_px": spacing,
        **{f"n_cue_{cue}": used[cue] for cue in TEXTURE_CUES},
    }
