


import os
import json
import math
import itertools
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np


# =========================================================
# Small helpers
# =========================================================

def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# =========================================================
# Image loading & preprocessing
# =========================================================

def _load_and_resize(image_path: str,
                     canon_w: int,
                     canon_h: int,
                     max_dim: int = 1800) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image from disk, convert to gray & color, resize while preserving aspect.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"Failed to read image with OpenCV: {image_path}")

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        gray_r = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        color_r = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        gray_r, color_r = gray, color

    _safe_print(f"[INFO] Loaded {image_path} -> resized to {gray_r.shape[1]}x{gray_r.shape[0]}")
    return gray_r, color_r


def _equalize_and_blur(gray: np.ndarray) -> np.ndarray:
    """Just blur to reduce noise. Histogram equalization not used."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


# =========================================================
# Fiducial detection & homography
# =========================================================

def _adaptive_thresh_and_morph(blur: np.ndarray, expected_r: int):
    blockSize = max(15, int(expected_r // 1.5) | 1)
    C = 7
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize, C
    )

    k = max(3, int(round(expected_r * 0.15)))
    kernel = np.ones((k, k), np.uint8)

    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th, opened, closed


def _hough_circle_detect(gray_blur: np.ndarray,
                         expected_r: int,
                         minRadius: Optional[int] = None,
                         maxRadius: Optional[int] = None):
    dp = 1.2
    minDist = max(10, int(expected_r * 1.0))

    if minRadius is None:
        minRadius = max(2, int(expected_r * 0.5))
    if maxRadius is None:
        maxRadius = max(3, int(expected_r * 1.8))

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=100,
        param2=55,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    return [(int(x), int(y), int(r)) for (x, y, r) in circles]


def _contour_circle_detect(binary_img: np.ndarray,
                           expected_r: int,
                           area_thresh: Optional[float] = None):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    if area_thresh is None:
        area_thresh = max(30, math.pi * (expected_r * 0.4) ** 2)

    for c in contours:
        area = cv2.contourArea(c)
        if area < area_thresh:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if 0.5 < circularity <= 1.2:
            (x, y), r = cv2.minEnclosingCircle(c)
            candidates.append((int(round(x)), int(round(y)), int(round(r))))

    return candidates


def _dedupe_candidates(cands, min_center_dist=10):
    if not cands:
        return []
    cands = sorted(cands, key=lambda v: v[2], reverse=True)
    keep = []
    for c in cands:
        x, y, r = c
        too_close = False
        for k in keep:
            if _dist((x, y), (k[0], k[1])) < min_center_dist:
                too_close = True
                break
        if not too_close:
            keep.append(c)
    return keep


def _select_fiducial_candidates(candidates, image_shape):
    h, w = image_shape[:2]
    img_area = w * h
    best_combo = None
    best_score = 1e9

    if len(candidates) < 4:
        _safe_print("[WARN] Not enough candidates for geometry matching.")
        return []

    for combo in itertools.combinations(candidates, 4):
        pts = np.array([(c[0], c[1]) for c in combo], dtype=np.float32)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        bbox_w, bbox_h = x_max - x_min, y_max - y_min
        area = bbox_w * bbox_h
        ratio = area / img_area
        if ratio < 0.1 or ratio > 0.95:
            continue

        diag1 = _dist(pts[0], pts[2])
        diag2 = _dist(pts[1], pts[3])
        reg = abs(diag1 - diag2)
        if reg < best_score:
            best_combo = pts
            best_score = reg

    if best_combo is None:
        _safe_print("[WARN] No suitable 4-point combination found.")
        return []
    return best_combo.tolist()


def _order_fiducials(points):
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]       # TL
    ordered[2] = pts[np.argmax(s)]       # BR
    ordered[1] = pts[np.argmin(diff)]    # TR
    ordered[3] = pts[np.argmax(diff)]    # BL
    return ordered


def _compute_homography_and_warp(img_color: np.ndarray,
                                 src_pts_ordered: np.ndarray,
                                 config: Dict[str, Any],
                                 canon_w: int,
                                 canon_h: int) -> np.ndarray:
    src = np.array(src_pts_ordered, dtype=np.float32).reshape(4, 2)

    fid = config["fiducials"]["positions_px"]
    dst_list = [
        fid["top_left"],
        fid["top_right"],
        fid["bottom_right"],
        fid["bottom_left"],
    ]
    dst = np.array(dst_list, dtype=np.float32).reshape(4, 2)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("findHomography failed. Check fiducials/config.")

    warped = cv2.warpPerspective(img_color, H, (canon_w, canon_h), flags=cv2.INTER_LINEAR)
    return warped


# =========================================================
# Question template + scoring
# =========================================================

def _build_template_positions_from_config(cfg: Dict[str, Any]):
    q_cfg = cfg["questions"]
    start_x, start_y = q_cfg["start_position_px"]
    cols = q_cfg["columns"]
    qpc = q_cfg["questions_per_column"]
    R = q_cfg["bubble_radius_px"]
    pad = q_cfg["bubble_padding_px"]
    col_gap = q_cfg["column_gap_px"]
    options = q_cfg["options"]

    H_STEP = 2 * R + pad
    V_STEP = 2 * R + pad

    template_positions = []
    qnum = 1
    x = start_x

    for col in range(cols):
        y = start_y
        for _ in range(qpc):
            bubbles = []
            for i in range(len(options)):
                cx = x + R + i * H_STEP
                cy = y + R
                bubbles.append((cx, cy))
            template_positions.append({"q": qnum, "bubbles": bubbles})
            qnum += 1
            y += V_STEP
        x += len(options) * H_STEP + col_gap

    return template_positions


ROI_R = 14
AREA_THR = 0.25  # tune as needed


def _analyze_roi(roi: np.ndarray):
    if roi.size == 0:
        return 255.0, 0.0
    mean_val = roi.mean()
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = float(np.sum(th == 0)) / float(th.size)
    return mean_val, dark_ratio


def _detect_single_question(gray: np.ndarray, bubble_centers: List[Tuple[int, int]]):
    metrics = []
    candidates = []

    for (cx, cy) in bubble_centers:
        x1, y1 = int(cx - ROI_R), int(cy - ROI_R)
        x2, y2 = int(cx + ROI_R), int(cy + ROI_R)

        roi = gray[y1:y2, x1:x2]
        mean_val, dark_ratio = _analyze_roi(roi)
        metrics.append((mean_val, dark_ratio))

        if dark_ratio >= AREA_THR:
            candidates.append(len(metrics) - 1)

    if len(candidates) == 0:
        means = [m[0] for m in metrics]
        if not means:
            return None, "blank"
        idx = int(np.argmin(means))
        if means[idx] < 230:
            return idx, "single_infer"
        return None, "blank"

    if len(candidates) == 1:
        return candidates[0], "single"

    return None, "multiple"


def _score_warped_sheet(warped: np.ndarray,
                        template_pos: List[Dict[str, Any]],
                        answer_key: Dict[int, int]):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    detected: Dict[int, Optional[int]] = {}
    status: Dict[int, str] = {}
    score = 0

    for q in template_pos:
        qid = q["q"]
        choice, st = _detect_single_question(gray, q["bubbles"])
        detected[qid] = choice
        status[qid] = st

        if answer_key.get(qid) == choice and st in ("single", "single_infer"):
            score += 1

    return detected, status, score


# =========================================================
# Answer key normalization
# =========================================================

def _normalize_single_answer(opt,
                             options_list: List[str]) -> Optional[int]:
    """
    Accepts:
      - int index (0,1,2,3)
      - string 'A','B','C','D'
      - digit string '0','1',...
    Returns index or None.
    """
    if opt is None:
        return None

    if isinstance(opt, int):
        return opt

    if isinstance(opt, str):
        s = opt.strip().upper()

        if s.isdigit():
            return int(s)

        if s in options_list:
            return options_list.index(s)

        if len(s) == 1 and "A" <= s <= "Z":
            try:
                return options_list.index(s)
            except ValueError:
                return ord(s) - ord("A")

    return None


def _normalize_answer_key(raw_key: Dict[Any, Any],
                          options_list: List[str]) -> Dict[int, int]:
    """
    Convert JSON answer key from frontend into {qid: index}.
    raw_key example:
      { "1": "C", "2": 0, "3": "b" }
    """
    norm: Dict[int, int] = {}
    if not raw_key:
        return norm

    for k, v in raw_key.items():
        try:
            qid = int(k)
        except Exception:
            continue
        idx = _normalize_single_answer(v, options_list)
        if idx is not None:
            norm[qid] = idx
    return norm


# =========================================================
# Processor class
# =========================================================

class OMRProcessor:
    def __init__(self,
                 config: Dict[str, Any]):
        self.config = config
        self.canon_w = int(config["sheet"]["paper_size_px"][0])
        self.canon_h = int(config["sheet"]["paper_size_px"][1])
        self.template_positions = _build_template_positions_from_config(config)

        options = config.get("questions", {}).get("options", ["A", "B", "C", "D"])
        self.options_list = [str(o).strip().upper() for o in options]

    def _get_effective_answer_key(self,
                                  override_key: Optional[Dict[Any, Any]]) -> Dict[int, int]:
        """
        Only uses override_key (from frontend).
        Raises if it's missing or empty.
        """
        if not override_key:
            raise ValueError("Answer key not provided.")

        ak = _normalize_answer_key(override_key, self.options_list)
        if not ak:
            raise ValueError("Answer key is empty or invalid after normalization.")

        return ak

    def _warp_image(self, image_path: str) -> np.ndarray:
        gray_r, color_r = _load_and_resize(image_path, self.canon_w, self.canon_h)

        blur = _equalize_and_blur(gray_r)
        expected_r_scaled = max(
            6,
            int(round(self.config["fiducials"]["outer_radius_px"] * (gray_r.shape[1] / self.canon_w)))
        )
        _, _, cleaned = _adaptive_thresh_and_morph(blur, expected_r_scaled)

        hough_cands = _hough_circle_detect(blur, expected_r_scaled)
        contour_cands = _contour_circle_detect(cleaned, expected_r_scaled)
        combined = _dedupe_candidates(hough_cands + contour_cands)

        best4 = _select_fiducial_candidates(combined, gray_r.shape)
        if len(best4) != 4:
            raise RuntimeError("Could not find 4 fiducial marks for homography.")

        ordered = _order_fiducials(best4)
        warped = _compute_homography_and_warp(
            color_r, ordered, self.config, self.canon_w, self.canon_h
        )
        return warped

    def process_image(self,
                      image_path: str,
                      answer_key_override: Optional[Dict[Any, Any]] = None) -> Dict[str, Any]:
        """
        Main method used by main.py.
        Returns:
          {
            "detected": {qid: index | None},
            "status": {qid: "single"/"blank"/"multiple"/"single_infer"},
            "score": int,
            "error": None | str
          }
        """
        result = {
            "detected": {},
            "status": {},
            "score": 0,
            "error": None
        }

        # 1) Answer key
        try:
            effective_key = self._get_effective_answer_key(answer_key_override)
        except Exception as e:
            result["error"] = f"Answer key error: {str(e)}"
            return result

        # 2) Warping
        try:
            warped = self._warp_image(image_path)
        except Exception as e:
            result["error"] = f"Warping error: {str(e)}"
            return result

        # 3) Scoring
        try:
            detected, status, score = _score_warped_sheet(
                warped, self.template_positions, effective_key
            )
            result["detected"] = detected
            result["status"] = status
            result["score"] = int(score)
        except Exception as e:
            result["error"] = f"Scoring error: {str(e)}"

        return result


# =========================================================
# Factory for main.py
# =========================================================

def get_default_processor(model_path: Optional[str] = None,
                          template_path: str = "template.json") -> OMRProcessor:
    """
    Loads template.json, constructs config, and returns an OMRProcessor.
    (model_path is unused but kept for compatibility with your main.py signature)
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("template.json root must be a JSON object")

    processor = OMRProcessor(config=cfg)

    _safe_print(
        f"[INFO] OMRProcessor initialized. "
        f"Sheet size: {cfg['sheet']['paper_size_px']}, "
        f"Questions: {len(processor.template_positions)}"
    )

    return processor
