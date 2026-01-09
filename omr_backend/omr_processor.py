import cv2
import numpy as np
import json
import math
import os
from typing import Dict, Any

# ================================
# Load Template
# ================================
def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_answer_key(key):
    out = {}
    for k, v in (key or {}).items():
        try:
            out[int(k)] = v.strip().upper()
        except:
            pass
    return out

# ================================
# Geometry Helpers
# ================================
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ================================
# Preprocessing
# ================================
def load_and_resize(image_path, max_dim=1800):
    color = cv2.imread(image_path)
    if color is None:
        raise FileNotFoundError("Cannot open image")

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, float(max_dim) / max(h, w))

    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        color = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return gray, color

def equalize_and_blur(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)

# ================================
# Fiducial Detection
# ================================
def adaptive_thresh_and_morph(blur, expected_r):
    block = max(15, int(expected_r / 1.5) | 1)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, 7
    )
    k = max(3, int(round(expected_r * 0.15)))
    kernel = np.ones((k, k), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def hough_circle_detect(gray_blur, expected_r):
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, int(expected_r)),
        param1=100,
        param2=55,
        minRadius=int(expected_r * 0.5),
        maxRadius=int(expected_r * 1.8)
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    return [(x, y, r) for x, y, r in circles]

def contour_circle_detect(binary_img, expected_r):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    min_area = math.pi * (expected_r * 0.5) ** 2

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > min_area * 5:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4 * math.pi * area / (peri * peri)
        if 0.5 < circ <= 1.2:
            (x, y), r = cv2.minEnclosingCircle(c)
            candidates.append((int(x), int(y), int(r)))
    return candidates

def dedupe_candidates(cands, min_center_dist=10):
    keep = []
    for x, y, r in sorted(cands, key=lambda v: v[2], reverse=True):
        if all(dist((x, y), (kx, ky)) >= min_center_dist for kx, ky, _ in keep):
            keep.append((x, y, r))
    return keep

def select_fiducials(candidates, image_shape):
    if len(candidates) < 4:
        return []
    best = None
    best_err = 1e9
    H, W = image_shape[:2]
    img_area = W * H

    import itertools
    for combo in itertools.combinations(candidates, 4):
        pts = np.array([(c[0], c[1]) for c in combo])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        area = (x1 - x0) * (y1 - y0)
        if not (0.1 * img_area < area < 0.95 * img_area):
            continue
        d1 = dist(pts[0], pts[2])
        d2 = dist(pts[1], pts[3])
        err = abs(d1 - d2)
        if err < best_err:
            best_err = err
            best = pts
    return best.tolist() if best is not None else []

def order_fiducials(points):
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def compute_homography_and_warp(img_color, ordered, cfg):
    src = np.array(ordered, dtype=np.float32)
    fid = cfg["fiducials"]["positions_px"]

    dst = np.array([
        fid["top_left"],
        fid["top_right"],
        fid["bottom_right"],
        fid["bottom_left"]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    W, Hh = cfg["sheet"]["paper_size_px"]
    warped = cv2.warpPerspective(img_color, H, (W, Hh))
    return warped, H

# ================================
# Bubble Processing
# ================================
def normalize_sheet(gray):
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(gray)

def mean_intensity_in_circle(img, cx, cy, r):
    H, W = img.shape
    x0 = max(0, int(cx - r))
    x1 = min(W, int(cx + r))
    y0 = max(0, int(cy - r))
    y1 = min(H, int(cy + r))

    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return 255.0

    rr, cc = np.ogrid[:patch.shape[0], :patch.shape[1]]
    cy_local = cy - y0
    cx_local = cx - x0

    inner_r = int(0.65 * r)
    mask = (rr - cy_local) ** 2 + (cc - cx_local) ** 2 <= inner_r * inner_r
    if mask.sum() == 0:
        return float(patch.mean())

    return float(patch[mask].mean())

def extract_bubble_darkness(warped, cfg):
    """
    Improved bubble darkness estimator for pencil + pen outlines.
    Uses:
      - CLAHE contrast normalization
      - inner fill intensity
      - local annulus background
      - edge density (Canny) to detect pen outlines
    """

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = normalize_sheet(gray)

    q = cfg["questions"]
    start_x, start_y = q["start_position_px"]
    R, pad = q["bubble_radius_px"], q["bubble_padding_px"]
    gap = q["column_gap_px"]
    opts = q["options"]
    QPC = q["questions_per_column"]
    COLS = q["columns"]

    H_STEP = (2 * R) + pad
    V_STEP = (2 * R) + pad

    H_img, W_img = gray.shape
    global_bg = float(np.percentile(gray, 90))

    # Edge map for detecting pen outlines
    blur_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur_edges, 50, 140)

    EDGE_SCALE = 10.0     # lowered so outlines do not explode
    EDGE_MIN = 0.05       # minimum edge fraction
    RAW_MIN_FILL = 10.0   # below this bubble is blank

    ANN_IN = int(1.1 * R)
    ANN_OUT = int(2.1 * R)

    raw = {}
    qid = 1
    cx_base = start_x

    for col in range(COLS):
        cy = start_y
        for _ in range(QPC):
            raw[qid] = {}

            for i, opt in enumerate(opts):
                cx = int(cx_base + R + i * H_STEP)
                cy_center = int(cy + R)

                # Mean brightness inside bubble
                mean_in = mean_intensity_in_circle(gray, cx, cy_center, R)

                # Local background estimation
                x0 = max(0, cx - ANN_OUT)
                x1 = min(W_img, cx + ANN_OUT)
                y0 = max(0, cy_center - ANN_OUT)
                y1 = min(H_img, cy_center + ANN_OUT)

                patch = gray[y0:y1, x0:x1]
                if patch.size == 0:
                    local_bg = global_bg
                else:
                    rr, cc = np.ogrid[:patch.shape[0], :patch.shape[1]]
                    cy_l = cy_center - y0
                    cx_l = cx - x0
                    dist2 = (rr - cy_l) ** 2 + (cc - cx_l) ** 2

                    ann_mask = (dist2 <= ANN_OUT**2) & (dist2 >= ANN_IN**2)
                    if ann_mask.sum() > 20:
                        local_bg = float(patch[ann_mask].mean())
                    else:
                        local_bg = global_bg

                # Fill darkness metric
                fill_dark = max(0.0, local_bg - mean_in)

                # Edge-density for pen outlines
                e_patch = edges[y0:y1, x0:x1]
                if e_patch.size > 0:
                    inner_mask = (dist2 <= (0.7 * R) ** 2)
                    inner_edges = e_patch[inner_mask]
                    edge_frac = float(np.count_nonzero(inner_edges)) / float(inner_mask.sum())
                else:
                    edge_frac = 0.0

                edge_score = edge_frac * EDGE_SCALE if edge_frac > EDGE_MIN else 0.0

                # Final combined score
                if fill_dark < RAW_MIN_FILL and edge_frac < EDGE_MIN:
                    raw[qid][opt] = 0.0
                else:
                    raw[qid][opt] = float(fill_dark + edge_score)

            qid += 1
            cy += V_STEP

        cx_base += (len(opts) * H_STEP) + gap

    return raw

def normalize_darkness(raw):
    norm = {}
    for q, vals in raw.items():
        arr = np.array(list(vals.values()))
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-7:
            norm[q] = {k: 0.0 for k in vals}
        else:
            norm[q] = {k: (v - mn) / (mx - mn) for k, v in vals.items()}
    return norm

def detect_selected_options(norm, raw, min_darkness=20.0, dominance_ratio=1.8):
    """
    min_darkness     : absolute ink threshold
    dominance_ratio : how much darker the top bubble must be vs row average
    """

    selected = {}
    comments = {}

    for q in norm:
        optvals_norm = norm[q]
        optvals_raw  = raw[q]

        # ---------- 1. ABSOLUTE BLANK TEST ----------
        max_raw = max(optvals_raw.values())
        if max_raw < min_darkness:
            selected[q] = None
            comments[q] = "Not marked"
            continue

        # ---------- 2. RELATIVE BLANK TEST (VS AVERAGE) ----------
        avg_raw = sum(optvals_raw.values()) / len(optvals_raw)
        if max_raw < dominance_ratio * avg_raw:
            selected[q] = None
            comments[q] = "Not marked"
            continue

        # ---------- 3. NORMALIZED DECISION ----------
        sorted_opts = sorted(optvals_norm.items(), key=lambda x: x[1], reverse=True)
        top_opt, top_val = sorted_opts[0]
        second_val = sorted_opts[1][1]

        # ---------- CLEAR SINGLE ----------
        if top_val >= 0.55 and (top_val - second_val) >= 0.20:
            selected[q] = [top_opt]
            comments[q] = "Single_detected"


        # ---------- MULTIPLE ----------
        elif top_val >= 0.55:
            selected[q] = None
            comments[q] = "Multiple marks"

        # ---------- FAINT ----------
        else:
            selected[q] = None
            comments[q] = "Not marked"

    return selected, comments

def detect_selected_options(norm, raw, min_darkness=20.0, dominance_ratio=1.8):
    """
    Returns:
      - selected[q] = ["A"] for single
      - selected[q] = ["A", "C"] for multiple
      - comments[q] = "Single_detected", "Multiple_detected", "Not marked", "Partial_mark"
    """

    selected = {}
    comments = {}

    for q in norm:
        opt_norm = norm[q]
        opt_raw  = raw[q]

        # ------- BLANK CHECK -------
        max_raw = max(opt_raw.values())
        if max_raw < min_darkness:
            selected[q] = None
            comments[q] = "Not marked"
            continue

        # ------- RELATIVE BLANK CHECK -------
        avg_raw = sum(opt_raw.values()) / len(opt_raw)
        if max_raw < dominance_ratio * avg_raw:
            selected[q] = None
            comments[q] = "Not marked"
            continue

        # ------- SORT OPTIONS -------
        sorted_opts = sorted(opt_norm.items(), key=lambda x: x[1], reverse=True)

        top_opt, top_val = sorted_opts[0]
        second_val = sorted_opts[1][1]

        # ---- DETECT MULTIPLE ----
        multi = []
        for opt, val in opt_norm.items():
            # any bubble close to the top value is considered marked
            if val >= (top_val - 0.15) and opt_raw[opt] >= min_darkness:
                multi.append(opt)

        # ---- SINGLE ANSWER ----
        if len(multi) == 1:
            selected[q] = [multi[0]]
            comments[q] = "Single_detected"
            continue

        # ---- MULTIPLE ANSWERS ----
        if len(multi) > 1:
            selected[q] = multi
            comments[q] = "Multiple_detected"
            continue

        # ---- FALLBACK ----
        selected[q] = None
        comments[q] = "Not marked"

    return selected, comments

def compute_score(selected, key):
    score = 0
    for q, ans in key.items():
        if selected.get(q) == [ans]:
            score += 1
    return score


# ================================
# Roll Number Detection
# ================================
def detect_rollno(warped, cfg, threshold=150):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    roll_cfg = cfg["metadata_fields"]["roll_no"]

    boxes = roll_cfg["boxes"]
    box_w, box_h = roll_cfg["box_size_px"]
    start_x, start_y = roll_cfg["position_px"]
    gap = roll_cfg["gap_px"]

    bubbles = roll_cfg["bubbles_count_below"]
    r = roll_cfg["bubble_radius_below"]
    pad = roll_cfg["bubble_padding_below"]

    roll = []
    for i in range(boxes):
        cx = start_x + i * (box_w + gap) + box_w // 2
        y0 = start_y + box_h + pad

        best_d = None
        best_m = 255
        for d in range(bubbles):
            cy = y0 + d * (2 * r + pad) + r
            meanv = mean_intensity_in_circle(gray, cx, cy, r)
            if meanv < best_m:
                best_m = meanv
                best_d = d

        roll.append(str(best_d) if best_m < threshold else "")

    roll_no = "".join(roll)
    return roll_no if roll_no.strip() else None

# ================================
# Main Processor Class
# ================================
class OMRProcessor:
    def __init__(self, template):
        self.cfg = template

    def process_image(self, image_path, answer_key=None):
        key = normalize_answer_key(answer_key)
        gray, color = load_and_resize(image_path)
        blur = equalize_and_blur(gray)

        expected_r = int(
            self.cfg["fiducials"]["outer_radius_px"]
            * (gray.shape[1] / self.cfg["sheet"]["paper_size_px"][0])
        )

        cleaned = adaptive_thresh_and_morph(blur, expected_r)
        cands = dedupe_candidates(
            hough_circle_detect(blur, expected_r) +
            contour_circle_detect(cleaned, expected_r),
            expected_r
        )

        best4 = select_fiducials(cands, gray.shape)
        if len(best4) != 4:
            return {"error": "fiducials_not_found"}

        ordered = order_fiducials(best4)
        warped, _ = compute_homography_and_warp(color, ordered, self.cfg)

        raw = extract_bubble_darkness(warped, self.cfg)
        norm = normalize_darkness(raw)
        selected, comments = detect_selected_options(norm, raw)
        score = compute_score(selected, key)
        roll_no = detect_rollno(warped, self.cfg)

        return {
            "error": None,
            "score": score,
            "max_score": len(key),
            "detected": selected,
            "comments": comments,
            "roll_number": roll_no
        }

# ================================
# Factory Function (FOR main.py)
# ================================
def get_default_processor(template_path="template-new-v2.json"):
    template = load_template(template_path)
    return OMRProcessor(template)