"""
omr_processor.py

Lightweight, robust OMR processor adapted from YOLOV8-detection-to-scoring.ipynb.
Exports a class OMRProcessor and convenience functions for processing folders/files.

Requirements:
- ultralytics
- opencv-python
- numpy
"""

import os
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omr_processor")


class OMRProcessor:
    def __init__(
        self,
        model_path: str = "best.pt",
        template_path: str = "template.json",
        canonical_w: int = 616,
        canonical_h: int = 795,
    ):
        """
        Initialize the processor. Loads the YOLO model once and the template.
        model_path and template_path are relative to working dir unless absolute.
        """
        self.model_path = model_path
        self.template_path = template_path
        self.CANONICAL_W = canonical_w
        self.CANONICAL_H = canonical_h

        logger.info(f"Loading YOLO model from: {self.model_path}")
        # Fix for PyTorch 2.6+ compatibility
        import torch
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
        self.model = YOLO(self.model_path)

        if not os.path.exists(self.template_path):
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
        with open(self.template_path, "r") as f:
            self.template_cfg = json.load(f)

        # read offsets (default values taken from original notebook)
        self.OFFSET_X_LEFT = self.template_cfg.get("offset_x_left", 214)
        self.OFFSET_X_RIGHT = self.template_cfg.get("offset_x_right", 215)
        self.OFFSET_Y_TOP = self.template_cfg.get("offset_y_top", 280)
        self.OFFSET_Y_BOTTOM = self.template_cfg.get("offset_y_bottom", 290)

        # generate bubble positions once
        self.TEMPLATE_POS = self.generate_bubble_positions(self.template_cfg)
        logger.info("OMRProcessor initialized.")

    def generate_bubble_positions(self, cfg: dict) -> List[dict]:
        """Expand template spacing into explicit bubble coordinates."""
        positions = []
        layout = cfg["layout"]
        q_idx = 0
        cols = cfg["columns"]
        rows_per_col = cfg["rows_per_column"]
        bubbles_per_q = cfg["bubbles_per_question"]

        for col in range(cols):
            x_base = layout["start_x"] + col * layout["col_spacing"]
            for row in range(rows_per_col):
                y_base = layout["start_y"] + row * layout["row_spacing"]
                bubbles = []
                for b in range(bubbles_per_q):
                    cx = x_base + b * layout["bubble_spacing"]
                    cy = y_base
                    bubbles.append((cx, cy))
                positions.append({"q": q_idx + 1, "bubbles": bubbles})
                q_idx += 1
        return positions

    # -------------------------
    # Warping / Fiducial logic
    # -------------------------
    @staticmethod
    def order_points(pts: List[Tuple[float, float]]) -> np.ndarray:
        """
        Order points to TL, TR, BR, BL - safe method based on y then x sorting.
        Input: list of (x, y) tuples
        """
        pts = sorted(pts, key=lambda x: (x[1], x[0]))
        top = sorted(pts[:2], key=lambda x: x[0])
        bottom = sorted(pts[2:], key=lambda x: x[0])
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    def warp_omr(self, image_path: str) -> np.ndarray:
        """Detect fiducials using YOLO and warp to canonical size. Raises on failure."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")

        results = self.model(img, verbose=False)[0]

        rects, squares = [], []
        # Use the same box-parsing logic as the notebook (keeps compatibility)
        for box in results.boxes:
            # box.cls and box.xyxy are usually small tensors; convert to Python numbers
            try:
                cls = int(box.cls[0])
            except Exception:
                # fallback if structure different
                cls = int(np.array(box.cls).ravel()[0])

            try:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = coords
            except Exception:
                # fallback: convert to array
                xy = np.array(box.xyxy).ravel()
                x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if cls == 1:
                rects.append((cx, cy))
            elif cls == 0:
                squares.append((cx, cy))

        # Choose squares if >=4, else rect-based expansion
        if len(squares) >= 4:
            src_pts = self.order_points(squares[:4])
        elif len(rects) >= 4:
            rects_ordered = self.order_points(rects[:4])
            tl, tr, br, bl = rects_ordered
            src_pts = np.array(
                [
                    (tl[0] - self.OFFSET_X_LEFT, tl[1] - self.OFFSET_Y_TOP),
                    (tr[0] + self.OFFSET_X_RIGHT, tr[1] - self.OFFSET_Y_TOP),
                    (br[0] + self.OFFSET_X_RIGHT, br[1] + self.OFFSET_Y_BOTTOM),
                    (bl[0] - self.OFFSET_X_LEFT, bl[1] + self.OFFSET_Y_BOTTOM),
                ],
                dtype="float32",
            )
        else:
            raise ValueError(f"Not enough markers detected for image: {os.path.basename(image_path)}")

        dst_pts = np.array(
            [
                [0, 0],
                [self.CANONICAL_W, 0],
                [self.CANONICAL_W, self.CANONICAL_H],
                [0, self.CANONICAL_H],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (self.CANONICAL_W, self.CANONICAL_H))
        return warped

    # -------------------------
    # Bubble evaluate
    # -------------------------
    @staticmethod
    def safe_roi(gray: np.ndarray, cx: float, cy: float, r: int = 8) -> np.ndarray:
        """Crop ROI safely; return empty array if out of bounds."""
        h, w = gray.shape[:2]
        x1 = max(0, int(np.round(cx - r)))
        x2 = min(w, int(np.round(cx + r)))
        y1 = max(0, int(np.round(cy - r)))
        y2 = min(h, int(np.round(cy + r)))
        if x2 <= x1 or y2 <= y1:
            return np.array([])
        return gray[y1:y2, x1:x2]

    def evaluate_bubbles(self, warped: np.ndarray, answer_key: dict, roi_r: int = 8) -> Tuple[dict, int]:
        """
        Return (detected_dict, score).
        detected_dict: {question_id: filled_index or None}
        """
        results = {}
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        score = 0

        for q in self.TEMPLATE_POS:
            qid = q["q"]
            bubbles = q["bubbles"]
            filled = None
            max_intensity = 1e9  # lower mean intensity => darker bubble

            for i, (cx, cy) in enumerate(bubbles):
                roi = self.safe_roi(gray, cx, cy, r=roi_r)
                if roi.size == 0:
                    continue
                mean_val = float(np.mean(roi))
                if mean_val < max_intensity:
                    max_intensity = mean_val
                    filled = i

            results[qid] = filled
            if qid in answer_key and filled == answer_key[qid]:
                score += 1

        return results, score

    # -------------------------
    # Single image / batch processing
    # -------------------------
    def process_image(self, image_path: str, answer_key: dict) -> dict:
        """Process a single image and return { 'score': int, 'detected': {...} } or error info."""
        start = time.time()
        try:
            warped = self.warp_omr(image_path)
            detected, score = self.evaluate_bubbles(warped, answer_key)
            elapsed = time.time() - start
            return {"score": int(score), "detected": detected, "time_sec": round(elapsed, 3)}
        except Exception as e:
            logger.exception(f"Error processing image {image_path}: {e}")
            return {"error": str(e)}

    def process_folder(self, folder_path: str, answer_key: dict, sort_names: bool = True) -> dict:
        """
        Process all images in a folder and return results dict:
        { 'filename.jpg': {'score': X, 'detected': {...}, 'time_sec': 0.34 }, ... }
        Also returns summary times via logger.
        """
        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if sort_names:
            images.sort()
        results = {}
        batch_start = time.time()
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            res = self.process_image(img_path, answer_key)
            results[img_name] = res
            logger.info(f"Processed {img_name} -> {res.get('score', 'ERR')} (time={res.get('time_sec')})")
        batch_end = time.time()
        total_time = batch_end - batch_start
        avg_time = total_time / len(images) if images else 0.0
        logger.info(f"✅ Processed {len(images)} images in {total_time:.2f}s (avg {avg_time:.3f}s/img)")
        # attach summary
        results["_summary"] = {"count": len(images), "total_time_sec": total_time, "avg_time_sec": avg_time}
        return results


# Convenience module-level object and wrapper for backward compatibility
_default_processor = None


def get_default_processor(model_path="best.pt", template_path="template.json"):
    global _default_processor
    if _default_processor is None:
        _default_processor = OMRProcessor(model_path=model_path, template_path=template_path)
    return _default_processor


def process_folder(folder_path: str, answer_key: dict):
    proc = get_default_processor()
    return proc.process_folder(folder_path, answer_key)


def process_files(file_paths: List[str], answer_key: dict):
    proc = get_default_processor()
    results = {}
    for p in file_paths:
        fname = os.path.basename(p)
        results[fname] = proc.process_image(p, answer_key)
    return results
