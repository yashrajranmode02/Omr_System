


# # # import os
# # # import json
# # # import math
# # # import itertools
# # # from typing import Dict, Any, List, Optional, Tuple

# # # import cv2
# # # import numpy as np


# # # # =========================================================
# # # # Small helpers
# # # # =========================================================

# # # def _safe_print(*args, **kwargs):
# # #     try:
# # #         print(*args, **kwargs)
# # #     except Exception:
# # #         pass


# # # def _dist(a, b) -> float:
# # #     return math.hypot(a[0] - b[0], a[1] - b[1])


# # # # =========================================================
# # # # Image loading & preprocessing
# # # # =========================================================

# # # def _load_and_resize(image_path: str,
# # #                      canon_w: int,
# # #                      canon_h: int,
# # #                      max_dim: int = 1800) -> Tuple[np.ndarray, np.ndarray]:
# # #     """
# # #     Load image from disk, convert to gray & color, resize while preserving aspect.
# # #     """
# # #     if not os.path.exists(image_path):
# # #         raise FileNotFoundError(f"Image not found: {image_path}")

# # #     color = cv2.imread(image_path, cv2.IMREAD_COLOR)
# # #     if color is None:
# # #         raise ValueError(f"Failed to read image with OpenCV: {image_path}")

# # #     gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
# # #     h, w = gray.shape

# # #     scale = min(1.0, float(max_dim) / max(h, w))
# # #     if scale < 1.0:
# # #         new_w, new_h = int(round(w * scale)), int(round(h * scale))
# # #         gray_r = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
# # #         color_r = cv2.resize(color, (new_w, new_h), interpolation=cv2.INTER_AREA)
# # #     else:
# # #         gray_r, color_r = gray, color

# # #     _safe_print(f"[INFO] Loaded {image_path} -> resized to {gray_r.shape[1]}x{gray_r.shape[0]}")
# # #     return gray_r, color_r


# # # def _equalize_and_blur(gray: np.ndarray) -> np.ndarray:
# # #     """Just blur to reduce noise. Histogram equalization not used."""
# # #     blur = cv2.GaussianBlur(gray, (5, 5), 0)
# # #     return blur


# # # # =========================================================
# # # # Fiducial detection & homography
# # # # =========================================================

# # # def _adaptive_thresh_and_morph(blur: np.ndarray, expected_r: int):
# # #     blockSize = max(15, int(expected_r // 1.5) | 1)
# # #     C = 7
# # #     th = cv2.adaptiveThreshold(
# # #         blur, 255,
# # #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# # #         cv2.THRESH_BINARY_INV,
# # #         blockSize, C
# # #     )

# # #     k = max(3, int(round(expected_r * 0.15)))
# # #     kernel = np.ones((k, k), np.uint8)

# # #     opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
# # #     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
# # #     return th, opened, closed


# # # def _hough_circle_detect(gray_blur: np.ndarray,
# # #                          expected_r: int,
# # #                          minRadius: Optional[int] = None,
# # #                          maxRadius: Optional[int] = None):
# # #     dp = 1.2
# # #     minDist = max(10, int(expected_r * 1.0))

# # #     if minRadius is None:
# # #         minRadius = max(2, int(expected_r * 0.5))
# # #     if maxRadius is None:
# # #         maxRadius = max(3, int(expected_r * 1.8))

# # #     circles = cv2.HoughCircles(
# # #         gray_blur,
# # #         cv2.HOUGH_GRADIENT,
# # #         dp=dp,
# # #         minDist=minDist,
# # #         param1=100,
# # #         param2=55,
# # #         minRadius=minRadius,
# # #         maxRadius=maxRadius
# # #     )

# # #     if circles is None:
# # #         return []
# # #     circles = np.round(circles[0]).astype(int)
# # #     return [(int(x), int(y), int(r)) for (x, y, r) in circles]


# # # def _contour_circle_detect(binary_img: np.ndarray,
# # #                            expected_r: int,
# # #                            area_thresh: Optional[float] = None):
# # #     contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # #     candidates = []

# # #     if area_thresh is None:
# # #         area_thresh = max(30, math.pi * (expected_r * 0.4) ** 2)

# # #     for c in contours:
# # #         area = cv2.contourArea(c)
# # #         if area < area_thresh:
# # #             continue

# # #         perimeter = cv2.arcLength(c, True)
# # #         if perimeter <= 0:
# # #             continue

# # #         circularity = 4 * math.pi * area / (perimeter * perimeter)
# # #         if 0.5 < circularity <= 1.2:
# # #             (x, y), r = cv2.minEnclosingCircle(c)
# # #             candidates.append((int(round(x)), int(round(y)), int(round(r))))

# # #     return candidates


# # # def _dedupe_candidates(cands, min_center_dist=10):
# # #     if not cands:
# # #         return []
# # #     cands = sorted(cands, key=lambda v: v[2], reverse=True)
# # #     keep = []
# # #     for c in cands:
# # #         x, y, r = c
# # #         too_close = False
# # #         for k in keep:
# # #             if _dist((x, y), (k[0], k[1])) < min_center_dist:
# # #                 too_close = True
# # #                 break
# # #         if not too_close:
# # #             keep.append(c)
# # #     return keep


# # # def _select_fiducial_candidates(candidates, image_shape):
# # #     h, w = image_shape[:2]
# # #     img_area = w * h
# # #     best_combo = None
# # #     best_score = 1e9

# # #     if len(candidates) < 4:
# # #         _safe_print("[WARN] Not enough candidates for geometry matching.")
# # #         return []

# # #     for combo in itertools.combinations(candidates, 4):
# # #         pts = np.array([(c[0], c[1]) for c in combo], dtype=np.float32)
# # #         x_min, y_min = np.min(pts, axis=0)
# # #         x_max, y_max = np.max(pts, axis=0)
# # #         bbox_w, bbox_h = x_max - x_min, y_max - y_min
# # #         area = bbox_w * bbox_h
# # #         ratio = area / img_area
# # #         if ratio < 0.1 or ratio > 0.95:
# # #             continue

# # #         diag1 = _dist(pts[0], pts[2])
# # #         diag2 = _dist(pts[1], pts[3])
# # #         reg = abs(diag1 - diag2)
# # #         if reg < best_score:
# # #             best_combo = pts
# # #             best_score = reg

# # #     if best_combo is None:
# # #         _safe_print("[WARN] No suitable 4-point combination found.")
# # #         return []
# # #     return best_combo.tolist()


# # # def _order_fiducials(points):
# # #     pts = np.array(points, dtype=np.float32)
# # #     s = pts.sum(axis=1)
# # #     diff = np.diff(pts, axis=1)

# # #     ordered = np.zeros((4, 2), dtype=np.float32)
# # #     ordered[0] = pts[np.argmin(s)]       # TL
# # #     ordered[2] = pts[np.argmax(s)]       # BR
# # #     ordered[1] = pts[np.argmin(diff)]    # TR
# # #     ordered[3] = pts[np.argmax(diff)]    # BL
# # #     return ordered


# # # def _compute_homography_and_warp(img_color: np.ndarray,
# # #                                  src_pts_ordered: np.ndarray,
# # #                                  config: Dict[str, Any],
# # #                                  canon_w: int,
# # #                                  canon_h: int) -> np.ndarray:
# # #     src = np.array(src_pts_ordered, dtype=np.float32).reshape(4, 2)

# # #     fid = config["fiducials"]["positions_px"]
# # #     dst_list = [
# # #         fid["top_left"],
# # #         fid["top_right"],
# # #         fid["bottom_right"],
# # #         fid["bottom_left"],
# # #     ]
# # #     dst = np.array(dst_list, dtype=np.float32).reshape(4, 2)

# # #     H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
# # #     if H is None:
# # #         raise RuntimeError("findHomography failed. Check fiducials/config.")

# # #     warped = cv2.warpPerspective(img_color, H, (canon_w, canon_h), flags=cv2.INTER_LINEAR)
# # #     return warped


# # # # =========================================================
# # # # Question template + scoring
# # # # =========================================================

# # # def _build_template_positions_from_config(cfg: Dict[str, Any]):
# # #     q_cfg = cfg["questions"]
# # #     start_x, start_y = q_cfg["start_position_px"]
# # #     cols = q_cfg["columns"]
# # #     qpc = q_cfg["questions_per_column"]
# # #     R = q_cfg["bubble_radius_px"]
# # #     pad = q_cfg["bubble_padding_px"]
# # #     col_gap = q_cfg["column_gap_px"]
# # #     options = q_cfg["options"]

# # #     H_STEP = 2 * R + pad
# # #     V_STEP = 2 * R + pad

# # #     template_positions = []
# # #     qnum = 1
# # #     x = start_x

# # #     for col in range(cols):
# # #         y = start_y
# # #         for _ in range(qpc):
# # #             bubbles = []
# # #             for i in range(len(options)):
# # #                 cx = x + R + i * H_STEP
# # #                 cy = y + R
# # #                 bubbles.append((cx, cy))
# # #             template_positions.append({"q": qnum, "bubbles": bubbles})
# # #             qnum += 1
# # #             y += V_STEP
# # #         x += len(options) * H_STEP + col_gap

# # #     return template_positions


# # # ROI_R = 14
# # # AREA_THR = 0.25  # tune as needed


# # # def _analyze_roi(roi: np.ndarray):
# # #     if roi.size == 0:
# # #         return 255.0, 0.0
# # #     mean_val = roi.mean()
# # #     _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # #     dark_ratio = float(np.sum(th == 0)) / float(th.size)
# # #     return mean_val, dark_ratio


# # # def _detect_single_question(gray: np.ndarray, bubble_centers: List[Tuple[int, int]]):
# # #     metrics = []
# # #     candidates = []

# # #     for (cx, cy) in bubble_centers:
# # #         x1, y1 = int(cx - ROI_R), int(cy - ROI_R)
# # #         x2, y2 = int(cx + ROI_R), int(cy + ROI_R)

# # #         roi = gray[y1:y2, x1:x2]
# # #         mean_val, dark_ratio = _analyze_roi(roi)
# # #         metrics.append((mean_val, dark_ratio))

# # #         if dark_ratio >= AREA_THR:
# # #             candidates.append(len(metrics) - 1)

# # #     if len(candidates) == 0:
# # #         means = [m[0] for m in metrics]
# # #         if not means:
# # #             return None, "blank"
# # #         idx = int(np.argmin(means))
# # #         if means[idx] < 230:
# # #             return idx, "single_infer"
# # #         return None, "blank"

# # #     if len(candidates) == 1:
# # #         return candidates[0], "single"

# # #     return None, "multiple"


# # # def _score_warped_sheet(warped: np.ndarray,
# # #                         template_pos: List[Dict[str, Any]],
# # #                         answer_key: Dict[int, int]):
# # #     gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# # #     detected: Dict[int, Optional[int]] = {}
# # #     status: Dict[int, str] = {}
# # #     score = 0

# # #     for q in template_pos:
# # #         qid = q["q"]
# # #         choice, st = _detect_single_question(gray, q["bubbles"])
# # #         detected[qid] = choice
# # #         status[qid] = st

# # #         if answer_key.get(qid) == choice and st in ("single", "single_infer"):
# # #             score += 1

# # #     return detected, status, score


# # # # =========================================================
# # # # Answer key normalization
# # # # =========================================================

# # # def _normalize_single_answer(opt,
# # #                              options_list: List[str]) -> Optional[int]:
# # #     """
# # #     Accepts:
# # #       - int index (0,1,2,3)
# # #       - string 'A','B','C','D'
# # #       - digit string '0','1',...
# # #     Returns index or None.
# # #     """
# # #     if opt is None:
# # #         return None

# # #     if isinstance(opt, int):
# # #         return opt

# # #     if isinstance(opt, str):
# # #         s = opt.strip().upper()

# # #         if s.isdigit():
# # #             return int(s)

# # #         if s in options_list:
# # #             return options_list.index(s)

# # #         if len(s) == 1 and "A" <= s <= "Z":
# # #             try:
# # #                 return options_list.index(s)
# # #             except ValueError:
# # #                 return ord(s) - ord("A")

# # #     return None


# # # def _normalize_answer_key(raw_key: Dict[Any, Any],
# # #                           options_list: List[str]) -> Dict[int, int]:
# # #     """
# # #     Convert JSON answer key from frontend into {qid: index}.
# # #     raw_key example:
# # #       { "1": "C", "2": 0, "3": "b" }
# # #     """
# # #     norm: Dict[int, int] = {}
# # #     if not raw_key:
# # #         return norm

# # #     for k, v in raw_key.items():
# # #         try:
# # #             qid = int(k)
# # #         except Exception:
# # #             continue
# # #         idx = _normalize_single_answer(v, options_list)
# # #         if idx is not None:
# # #             norm[qid] = idx
# # #     return norm


# # # # =========================================================
# # # # Processor class
# # # # =========================================================

# # # class OMRProcessor:
# # #     def __init__(self,
# # #                  config: Dict[str, Any]):
# # #         self.config = config
# # #         self.canon_w = int(config["sheet"]["paper_size_px"][0])
# # #         self.canon_h = int(config["sheet"]["paper_size_px"][1])
# # #         self.template_positions = _build_template_positions_from_config(config)

# # #         options = config.get("questions", {}).get("options", ["A", "B", "C", "D"])
# # #         self.options_list = [str(o).strip().upper() for o in options]

# # #     def _get_effective_answer_key(self,
# # #                                   override_key: Optional[Dict[Any, Any]]) -> Dict[int, int]:
# # #         """
# # #         Only uses override_key (from frontend).
# # #         Raises if it's missing or empty.
# # #         """
# # #         if not override_key:
# # #             raise ValueError("Answer key not provided.")

# # #         ak = _normalize_answer_key(override_key, self.options_list)
# # #         if not ak:
# # #             raise ValueError("Answer key is empty or invalid after normalization.")

# # #         return ak

# # #     def _warp_image(self, image_path: str) -> np.ndarray:
# # #         gray_r, color_r = _load_and_resize(image_path, self.canon_w, self.canon_h)

# # #         blur = _equalize_and_blur(gray_r)
# # #         expected_r_scaled = max(
# # #             6,
# # #             int(round(self.config["fiducials"]["outer_radius_px"] * (gray_r.shape[1] / self.canon_w)))
# # #         )
# # #         _, _, cleaned = _adaptive_thresh_and_morph(blur, expected_r_scaled)

# # #         hough_cands = _hough_circle_detect(blur, expected_r_scaled)
# # #         contour_cands = _contour_circle_detect(cleaned, expected_r_scaled)
# # #         combined = _dedupe_candidates(hough_cands + contour_cands)

# # #         best4 = _select_fiducial_candidates(combined, gray_r.shape)
# # #         if len(best4) != 4:
# # #             raise RuntimeError("Could not find 4 fiducial marks for homography.")

# # #         ordered = _order_fiducials(best4)
# # #         warped = _compute_homography_and_warp(
# # #             color_r, ordered, self.config, self.canon_w, self.canon_h
# # #         )
# # #         return warped

# # #     def process_image(self,
# # #                       image_path: str,
# # #                       answer_key_override: Optional[Dict[Any, Any]] = None) -> Dict[str, Any]:
# # #         """
# # #         Main method used by main.py.
# # #         Returns:
# # #           {
# # #             "detected": {qid: index | None},
# # #             "status": {qid: "single"/"blank"/"multiple"/"single_infer"},
# # #             "score": int,
# # #             "error": None | str
# # #           }
# # #         """
# # #         result = {
# # #             "detected": {},
# # #             "status": {},
# # #             "score": 0,
# # #             "error": None
# # #         }

# # #         # 1) Answer key
# # #         try:
# # #             effective_key = self._get_effective_answer_key(answer_key_override)
# # #         except Exception as e:
# # #             result["error"] = f"Answer key error: {str(e)}"
# # #             return result

# # #         # 2) Warping
# # #         try:
# # #             warped = self._warp_image(image_path)
# # #         except Exception as e:
# # #             result["error"] = f"Warping error: {str(e)}"
# # #             return result

# # #         # 3) Scoring
# # #         try:
# # #             detected, status, score = _score_warped_sheet(
# # #                 warped, self.template_positions, effective_key
# # #             )
# # #             result["detected"] = detected
# # #             result["status"] = status
# # #             result["score"] = int(score)
# # #         except Exception as e:
# # #             result["error"] = f"Scoring error: {str(e)}"

# # #         return result


# # # # =========================================================
# # # # Factory for main.py
# # # # =========================================================

# # # def get_default_processor(model_path: Optional[str] = None,
# # #                           template_path: str = "template.json") -> OMRProcessor:
# # #     """
# # #     Loads template.json, constructs config, and returns an OMRProcessor.
# # #     (model_path is unused but kept for compatibility with your main.py signature)
# # #     """
# # #     if not os.path.exists(template_path):
# # #         raise FileNotFoundError(f"Template file not found: {template_path}")

# # #     with open(template_path, "r", encoding="utf-8") as f:
# # #         cfg = json.load(f)

# # #     if not isinstance(cfg, dict):
# # #         raise ValueError("template.json root must be a JSON object")

# # #     processor = OMRProcessor(config=cfg)

# # #     _safe_print(
# # #         f"[INFO] OMRProcessor initialized. "
# # #         f"Sheet size: {cfg['sheet']['paper_size_px']}, "
# # #         f"Questions: {len(processor.template_positions)}"
# # #     )

# # #     return processor


# # # omr_processor.py
# # import cv2
# # import numpy as np
# # import json
# # import math


# # class OMRProcessor:
# #     def __init__(self, template):
# #         self.template = template

# #         # Preload template parameters
# #         self.fid = template["fiducials"]
# #         self.q = template["questions"]

# #         self.paper_w, self.paper_h = template["sheet"]["paper_size_px"]

# #     # ----------------------------------------------------------------------
# #     # 1) Load + convert
# #     # ----------------------------------------------------------------------
# #     def load_image(self, path):
# #         img = cv2.imread(path)
# #         if img is None:
# #             raise Exception("Cannot load image")
# #         return img

# #     # ----------------------------------------------------------------------
# #     # 2) Detect circles (fiducials)
# #     # ----------------------------------------------------------------------
# #     def detect_fiducial(self, gray, expected_pos, radius):
# #         """Find nearest circle to expected fiducial location."""
# #         x0, y0 = expected_pos

# #         circles = cv2.HoughCircles(
# #             gray,
# #             cv2.HOUGH_GRADIENT,
# #             dp=1.2,
# #             minDist=200,
# #             param1=80,
# #             param2=20,
# #             minRadius=int(radius * 0.4),
# #             maxRadius=int(radius * 1.6),
# #         )

# #         if circles is None:
# #             return None

# #         best = None
# #         best_dist = 1e9

# #         for c in circles[0]:
# #             x, y, r = c
# #             d = (x - x0) ** 2 + (y - y0) ** 2
# #             if d < best_dist:
# #                 best = (x, y)
# #                 best_dist = d

# #         return best

# #     # ----------------------------------------------------------------------
# #     # 3) Warp perspective to match template sheet size
# #     # ----------------------------------------------------------------------
# #     def warp_sheet(self, img):
# #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #         blur = cv2.medianBlur(gray, 7)

# #         radius = self.fid["outer_radius_px"]

# #         # Expected fiducial positions (template space)
# #         TL = tuple(self.fid["positions_px"]["top_left"])
# #         TR = tuple(self.fid["positions_px"]["top_right"])
# #         BL = tuple(self.fid["positions_px"]["bottom_left"])
# #         BR = tuple(self.fid["positions_px"]["bottom_right"])

# #         detected = []

# #         for p in [TL, TR, BL, BR]:
# #             found = self.detect_fiducial(blur, p, radius)
# #             if found is None:
# #                 return None
# #             detected.append(found)

# #         # Order
# #         pts_src = np.array(detected, dtype=np.float32)
# #         pts_dst = np.array([TL, TR, BL, BR], dtype=np.float32)

# #         # Perspective transform
# #         M = cv2.getPerspectiveTransform(pts_src, pts_dst)
# #         warped = cv2.warpPerspective(img, M, (self.paper_w, self.paper_h))

# #         return warped

# #     # ----------------------------------------------------------------------
# #     # 4) Detect bubble fill percentage
# #     # ----------------------------------------------------------------------
# #     def detect_bubble(self, img_roi):
# #         """Return fill level from 0.0 to 1.0."""
# #         gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
# #         thr = cv2.adaptiveThreshold(gray, 255,
# #                                     cv2.ADAPTIVE_THRESH_MEAN_C,
# #                                     cv2.THRESH_BINARY_INV,
# #                                     21, 8)
# #         filled_ratio = np.sum(thr == 255) / (thr.size)
# #         return filled_ratio

# #     # ----------------------------------------------------------------------
# #     # 5) Process all question bubbles
# #     # ----------------------------------------------------------------------
# #     def process_questions(self, warped, answer_key):
# #         start_x, start_y = self.q["start_position_px"]
# #         cols = self.q["columns"]
# #         q_per_col = self.q["questions_per_column"]
# #         options = self.q["options"]
# #         pad = self.q["bubble_padding_px"]
# #         radius = self.q["bubble_radius_px"]
# #         col_gap = self.q["column_gap_px"]

# #         results = {}
# #         status = {}
# #         score = 0

# #         question_no = 1

# #         for col in range(cols):
# #             base_x = start_x + col * col_gap

# #             for row in range(q_per_col):
# #                 cy = start_y + row * (radius * 2 + pad)

# #                 bubble_fills = []

# #                 for i, opt in enumerate(options):
# #                     cx = base_x + i * (radius * 2 + pad)

# #                     x1 = int(cx - radius)
# #                     x2 = int(cx + radius)
# #                     y1 = int(cy - radius)
# #                     y2 = int(cy + radius)

# #                     crop = warped[y1:y2, x1:x2]
# #                     if crop.size == 0:
# #                         bubble_fills.append(0)
# #                     else:
# #                         bubble_fills.append(self.detect_bubble(crop))

# #                 # Decide marked
# #                 filled = [i for i, v in enumerate(bubble_fills) if v > 0.18]

# #                 if len(filled) == 1:
# #                     detected = filled[0]
# #                     status[question_no] = "single"
# #                 elif len(filled) > 1:
# #                     detected = None
# #                     status[question_no] = "multiple"
# #                 else:
# #                     detected = None
# #                     status[question_no] = "empty"

# #                 results[question_no] = detected

# #                 # Score
# #                 if question_no in answer_key and detected == answer_key[question_no]:
# #                     score += 1

# #                 question_no += 1

# #         return results, status, score

# #     # ----------------------------------------------------------------------
# #     # Master function
# #     # ----------------------------------------------------------------------
# #     def process_image(self, path, answer_key):
# #         img = self.load_image(path)

# #         warped = self.warp_sheet(img)
# #         if warped is None:
# #             return {"error": "Fiducials not detected", "detected": {}, "score": 0, "status": {}}

# #         # Process questions
# #         detected, status, score = self.process_questions(warped, answer_key)

# #         return {
# #             "detected": detected,
# #             "status": status,
# #             "score": score,
# #             "error": None
# #         }


# # # --------------------------------------------------------------------------
# # # Factory loader
# # # --------------------------------------------------------------------------
# # def get_default_processor(model_path=None, template_path="template.json"):
# #     with open(template_path, "r") as f:
# #         template = json.load(f)

# #     return OMRProcessor(template)


# # # omr_processor.py
# # import cv2
# # import numpy as np
# # import json
# # import math
# # import os
# # from typing import Dict, Any

# # # ----------------------------
# # # Utilities
# # # ----------------------------
# # def dist(a, b):
# #     return math.hypot(a[0] - b[0], a[1] - b[1])


# # # ----------------------------
# # # Template loader & normalizer
# # # ----------------------------
# # def load_template(template_path: str) -> Dict[str, Any]:
# #     if not os.path.exists(template_path):
# #         raise FileNotFoundError(f"Template JSON not found: {template_path}")
# #     with open(template_path, "r", encoding="utf-8") as f:
# #         cfg = json.load(f)
# #     return cfg


# # def normalize_answer_key(key_obj) -> Dict[int, str]:
# #     """
# #     Turn incoming answer_key (possibly string-keyed) into {int_qid: 'A'/'B'/...}
# #     Ignores invalid entries but returns best-effort mapping.
# #     """
# #     norm = {}
# #     if not key_obj:
# #         return norm
# #     # If it's already a dict-like with int keys, handle gracefully
# #     try:
# #         for k, v in key_obj.items():
# #             # coerce key to int (skip if impossible)
# #             try:
# #                 qid = int(k)
# #             except Exception:
# #                 # maybe k already int or not convertible -> skip
# #                 continue
# #             # normalize value: uppercase first char (A/B/C/D) if string
# #             if isinstance(v, str):
# #                 val = v.strip().upper()
# #                 if val == "":
# #                     continue
# #                 # allow forms like "A" or "A\n" or "a"
# #                 norm[qid] = val[0]
# #             elif isinstance(v, (int, float)):
# #                 # numeric options not expected, but store as string
# #                 norm[qid] = str(int(v))
# #             else:
# #                 # unsupported value type - skip
# #                 continue
# #     except Exception:
# #         # defensive fallback: return empty mapping
# #         return {}
# #     return norm


# # # ----------------------------
# # # Fiducial detection helpers
# # # ----------------------------
# # def adaptive_thresh_and_morph(gray, expected_r):
# #     block = max(15, int(expected_r // 1.5) | 1)
# #     th = cv2.adaptiveThreshold(
# #         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #         cv2.THRESH_BINARY_INV, block, 7
# #     )
# #     k = max(3, int(round(expected_r * 0.12)))
# #     kernel = np.ones((k, k), np.uint8)
# #     opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)
# #     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, 1)
# #     return closed


# # def hough_fiducials(gray, expected_r):
# #     circles = cv2.HoughCircles(
# #         gray, cv2.HOUGH_GRADIENT, dp=1.2,
# #         minDist=max(10, int(expected_r * 0.9)),
# #         param1=80, param2=40,
# #         minRadius=max(2, int(expected_r * 0.5)),
# #         maxRadius=max(3, int(expected_r * 1.8))
# #     )
# #     if circles is None:
# #         return []
# #     circles = np.round(circles[0]).astype(int)
# #     return [(int(x), int(y), int(r)) for (x, y, r) in circles]


# # def contour_fiducials(binary, expected_r):
# #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     cand = []
# #     area_thresh = max(30, math.pi * (expected_r * 0.4) ** 2)
# #     for c in contours:
# #         area = cv2.contourArea(c)
# #         if area < area_thresh:
# #             continue
# #         peri = cv2.arcLength(c, True)
# #         if peri <= 0:
# #             continue
# #         circularity = 4 * math.pi * area / (peri * peri)
# #         if 0.45 <= circularity <= 1.25:
# #             (x, y), r = cv2.minEnclosingCircle(c)
# #             cand.append((int(round(x)), int(round(y)), int(round(r))))
# #     return cand


# # def dedupe_candidates(cands, min_center_dist=12):
# #     if not cands:
# #         return []
# #     cands = sorted(cands, key=lambda v: v[2], reverse=True)
# #     keep = []
# #     for c in cands:
# #         x, y, r = c
# #         too_close = False
# #         for kx, ky, kr in keep:
# #             if dist((x, y), (kx, ky)) < min_center_dist:
# #                 too_close = True
# #                 break
# #         if not too_close:
# #             keep.append(c)
# #     return keep


# # def select_fiducial_candidates(candidates, image_shape):
# #     import itertools
# #     h, w = image_shape[:2]
# #     img_area = w * h
# #     best_combo = None
# #     best_score = 1e9

# #     if len(candidates) < 4:
# #         return []

# #     for combo in itertools.combinations(candidates, 4):
# #         pts = np.array([(c[0], c[1]) for c in combo], dtype=np.float32)
# #         x_min, y_min = pts.min(axis=0)
# #         x_max, y_max = pts.max(axis=0)
# #         bbox_w, bbox_h = x_max - x_min, y_max - y_min
# #         area = bbox_w * bbox_h
# #         ratio = area / img_area
# #         if ratio < 0.05 or ratio > 0.99:
# #             continue
# #         diag1 = dist(pts[0], pts[2])
# #         diag2 = dist(pts[1], pts[3])
# #         reg = abs(diag1 - diag2)
# #         if reg < best_score:
# #             best_score = reg
# #             best_combo = pts

# #     return best_combo.tolist() if best_combo is not None else []


# # def order_fiducials(points):
# #     pts = np.array(points, dtype=np.float32)
# #     s = pts.sum(axis=1)
# #     diff = np.diff(pts, axis=1).reshape(-1)
# #     ordered = np.zeros((4, 2), dtype=np.float32)
# #     ordered[0] = pts[np.argmin(s)]  # TL
# #     ordered[2] = pts[np.argmax(s)]  # BR
# #     ordered[1] = pts[np.argmin(diff)]  # TR
# #     ordered[3] = pts[np.argmax(diff)]  # BL
# #     return ordered


# # def compute_homography_and_warp(img_color, src_pts_ordered, cfg):
# #     src = np.array(src_pts_ordered, dtype=np.float32).reshape(4, 2)
# #     fid = cfg['fiducials']['positions_px']
# #     dst_list = [
# #         fid['top_left'],
# #         fid['top_right'],
# #         fid['bottom_right'],
# #         fid['bottom_left']
# #     ]
# #     dst = np.array(dst_list, dtype=np.float32).reshape(4, 2)
# #     H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
# #     if H is None:
# #         raise RuntimeError("Homography failed (H is None)")
# #     W = int(cfg['sheet']['paper_size_px'][0])
# #     Hh = int(cfg['sheet']['paper_size_px'][1])
# #     warped = cv2.warpPerspective(img_color, H, (W, Hh), flags=cv2.INTER_LINEAR)
# #     # compute reprojection error (optional)
# #     src_h = np.concatenate([src.reshape(-1, 2), np.ones((4, 1), dtype=np.float32)], axis=1).T
# #     proj_h = H.dot(src_h)
# #     proj = (proj_h[:2] / proj_h[2]).T
# #     dists = np.linalg.norm(proj - dst, axis=1)
# #     reproj_err = float(dists.mean())
# #     return warped, H, reproj_err


# # # ----------------------------
# # # Bubble probing helpers
# # # ----------------------------
# # def mean_intensity_in_circle(gray, cx, cy, r):
# #     H, W = gray.shape[:2]
# #     x0 = max(0, int(cx - r))
# #     y0 = max(0, int(cy - r))
# #     x1 = min(W, int(cx + r))
# #     y1 = min(H, int(cy + r))
# #     patch = gray[y0:y1, x0:x1]
# #     if patch.size == 0:
# #         return 255.0
# #     rr, cc = np.ogrid[:patch.shape[0], :patch.shape[1]]
# #     cy_local = cy - y0
# #     cx_local = cx - x0
# #     mask = (rr - cy_local)**2 + (cc - cx_local)**2 <= r*r
# #     if mask.sum() == 0:
# #         return float(patch.mean())
# #     return float(patch[mask].mean())


# # # ----------------------------
# # # Extract darkness for every bubble
# # # ----------------------------
# # def extract_bubble_darkness(warped_img, cfg):
# #     if len(warped_img.shape) == 3:
# #         gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
# #     else:
# #         gray = warped_img.copy()

# #     q_cfg = cfg['questions']
# #     start_x, start_y = int(q_cfg['start_position_px'][0]), int(q_cfg['start_position_px'][1])
# #     R = int(q_cfg['bubble_radius_px'])
# #     pad = int(q_cfg['bubble_padding_px'])
# #     col_gap = int(q_cfg['column_gap_px'])
# #     opts = list(q_cfg['options'])
# #     QPC = int(q_cfg['questions_per_column'])
# #     COLS = int(q_cfg['columns'])

# #     H_STEP = 2 * R + pad
# #     V_STEP = 2 * R + pad

# #     raw = {}
# #     qnum = 1
# #     cx_base = start_x

# #     for col in range(COLS):
# #         cy = start_y
# #         for _ in range(QPC):
# #             raw[qnum] = {}
# #             for i, opt in enumerate(opts):
# #                 cx = cx_base + R + i * H_STEP
# #                 cy_c = cy + R
# #                 meanv = mean_intensity_in_circle(gray, cx, cy_c, R)
# #                 darkness = 255.0 - meanv
# #                 raw[qnum][opt] = float(darkness)
# #             qnum += 1
# #             cy += V_STEP
# #         cx_base += len(opts) * H_STEP + col_gap

# #     return raw


# # # ----------------------------
# # # Improved classification using threshold offset per-question
# # # ----------------------------
# # def classify_bubbles_improved(raw, threshold_offset=18.0):
# #     """
# #     raw: { q: { 'A': darkness, 'B': darkness, ... } }
# #     Returns:
# #       selected: {q: 'A' or None or [opts]}
# #       status: {q: 'single'|'multiple'|'blank'}
# #     """
# #     selected = {}
# #     status = {}
# #     for q, optvals in raw.items():
# #         vals = np.array(list(optvals.values()), dtype=float)
# #         mean_d = float(vals.mean())
# #         thresh = mean_d + float(threshold_offset)

# #         chosen = [opt for opt, v in optvals.items() if v >= thresh]

# #         if len(chosen) == 1:
# #             selected[q] = chosen[0]
# #             status[q] = "single"
# #         elif len(chosen) > 1:
# #             selected[q] = chosen  # multiple choices list
# #             status[q] = "multiple"
# #         else:
# #             selected[q] = None
# #             status[q] = "blank"

# #     return selected, status


# # # ----------------------------
# # # Scoring helper
# # # ----------------------------
# # def compute_score(selected_map, answer_key_map):
# #     score = 0
# #     # answer_key_map: {int_qid: 'A'}
# #     for qid, correct in answer_key_map.items():
# #         # only award points when exactly one selected and equals correct
# #         chosen = selected_map.get(qid, None)
# #         if isinstance(chosen, str) and chosen == correct:
# #             score += 1
# #     return score


# # # ----------------------------
# # # OMR Processor class
# # # ----------------------------
# # class OMRProcessor:
# #     def __init__(self, template: Dict[str, Any]):
# #         self.cfg = template
# #         self.CANON_W = int(self.cfg['sheet']['paper_size_px'][0])
# #         self.CANON_H = int(self.cfg['sheet']['paper_size_px'][1])

# #     def process_image(self, image_path: str, incoming_answer_key=None) -> Dict[str, Any]:
# #         """
# #         Main entry:
# #           image_path: path to uploaded sheet image
# #           incoming_answer_key: dict parsed from frontend POST form (may have string keys)
# #         Returns JSON-like dict:
# #           { score, detected, status, max_score, error }
# #         """
# #         # Normalize answer key from frontend
# #         try:
# #             answer_key = normalize_answer_key(incoming_answer_key or {})
# #         except Exception:
# #             answer_key = {}

# #         try:
# #             img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# #             if img is None:
# #                 return {"error": "cannot_read_image", "score": 0, "detected": {}, "status": {}, "max_score": len(answer_key)}

# #             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #             # Estimate expected fiducial radius scaled to image
# #             expected_r = max(6, int(round(self.cfg['fiducials']['outer_radius_px'] * (gray.shape[1] / float(self.CANON_W)))))

# #             # Detect candidates
# #             cleaned = adaptive_thresh_and_morph(gray, expected_r)
# #             hough_cands = hough_fiducials(gray, expected_r)
# #             contour_cands = contour_fiducials(cleaned, expected_r)
# #             combined = dedupe_candidates(hough_cands + contour_cands, min_center_dist=max(8, int(expected_r * 0.6)))

# #             # Select 4
# #             best4 = select_fiducial_candidates(combined, gray.shape)
# #             if len(best4) != 4:
# #                 return {"error": "fiducials_not_found", "score": 0, "detected": {}, "status": {}, "max_score": len(answer_key)}

# #             ordered = order_fiducials(best4)

# #             # Warp
# #             warped, H, reproj_err = compute_homography_and_warp(img, ordered, self.cfg)

# #             # Extract darkness
# #             raw = extract_bubble_darkness(warped, self.cfg)

# #             # Classify
# #             selected_map, status_map = classify_bubbles_improved(raw, threshold_offset=18.0)

# #             # Score
# #             score = compute_score(selected_map, answer_key)
# #             max_score = len(answer_key)

# #             return {
# #                 "score": int(score),
# #                 "detected": selected_map,
# #                 "status": status_map,
# #                 "max_score": int(max_score),
# #                 "error": None,
# #                 "reproj_err": float(reproj_err)
# #             }

# #         except Exception as e:
# #             return {"error": str(e), "score": 0, "detected": {}, "status": {}, "max_score": len(answer_key)}


# # # ----------------------------
# # # Factory function used by main.py
# # # ----------------------------
# # def get_default_processor(model_path=None, template_path="template.json"):
# #     template = load_template(template_path)
# #     return OMRProcessor(template)



# # omr_processor.py
# import cv2
# import numpy as np
# import json
# import math
# import os
# from typing import Dict, Any


# # ------------------------------------------------------
# # Helpers
# # ------------------------------------------------------
# def dist(a, b):
#     return math.hypot(a[0] - b[0], a[1] - b[1])


# def load_template(template_path: str) -> Dict[str, Any]:
#     if not os.path.exists(template_path):
#         raise FileNotFoundError(f"Template JSON not found: {template_path}")

#     with open(template_path, "r") as f:
#         return json.load(f)


# def normalize_answer_key(key_obj) -> Dict[int, str]:
#     norm = {}
#     if not key_obj:
#         return norm

#     for k, v in key_obj.items():
#         try:
#             q = int(k)
#         except:
#             continue

#         if isinstance(v, str) and v.strip():
#             norm[q] = v.strip().upper()[0]

#     return norm


# # ------------------------------------------------------
# # Fiducial Detection
# # ------------------------------------------------------
# def adaptive_thresh_and_morph(gray, expected_r):
#     block = max(15, int(expected_r / 1.5) | 1)

#     th = cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         block, 7
#     )

#     k = max(3, int(expected_r * 0.12))
#     kernel = np.ones((k, k), np.uint8)

#     opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
#     closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

#     return closed


# def hough_fiducials(gray, expected_r):
#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         dp=1.2,
#         minDist=max(10, int(expected_r * 0.8)),
#         param1=80,
#         param2=40,
#         minRadius=int(expected_r * 0.5),
#         maxRadius=int(expected_r * 1.6)
#     )

#     if circles is None:
#         return []

#     circles = np.round(circles[0]).astype(int)
#     return [(x, y, r) for (x, y, r) in circles]


# def contour_fiducials(binary, expected_r):
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     cand = []
#     min_area = math.pi * (expected_r * 0.35) ** 2

#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < min_area:
#             continue

#         peri = cv2.arcLength(c, True)
#         if peri <= 0:
#             continue

#         circ = 4 * math.pi * area / (peri * peri)
#         if 0.4 <= circ <= 1.3:
#             (x, y), r = cv2.minEnclosingCircle(c)
#             cand.append((int(x), int(y), int(r)))

#     return cand


# def dedupe_candidates(cands, min_center_dist=10):
#     if not cands:
#         return []

#     cands = sorted(cands, key=lambda v: v[2], reverse=True)
#     keep = []

#     for x, y, r in cands:
#         if all(dist((x, y), (kx, ky)) >= min_center_dist for (kx, ky, _) in keep):
#             keep.append((x, y, r))

#     return keep


# def select_fiducial_candidates(candidates, img_shape):
#     if len(candidates) < 4:
#         return []

#     import itertools

#     H, W = img_shape[:2]
#     img_area = H * W

#     best = None
#     best_err = 1e9

#     pts = [(c[0], c[1]) for c in candidates]

#     for combo in itertools.combinations(pts, 4):
#         arr = np.array(combo)
#         x_min, y_min = arr.min(axis=0)
#         x_max, y_max = arr.max(axis=0)

#         area = (x_max - x_min) * (y_max - y_min)
#         if not (0.05 * img_area < area < 0.95 * img_area):
#             continue

#         diag1 = dist(combo[0], combo[2])
#         diag2 = dist(combo[1], combo[3])
#         err = abs(diag1 - diag2)

#         if err < best_err:
#             best_err = err
#             best = combo

#     return list(best) if best is not None else []


# def order_fiducials(points):
#     pts = np.array(points, dtype=np.float32)
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1).reshape(-1)

#     ordered = np.zeros((4, 2), dtype=np.float32)
#     ordered[0] = pts[np.argmin(s)]     # TL
#     ordered[2] = pts[np.argmax(s)]     # BR
#     ordered[1] = pts[np.argmin(diff)]  # TR
#     ordered[3] = pts[np.argmax(diff)]  # BL

#     return ordered


# def compute_homography_and_warp(img, src_pts, cfg):
#     src = np.array(src_pts, dtype=np.float32)

#     fid = cfg['fiducials']['positions_px']
#     dst = np.array(
#         [
#             fid['top_left'],
#             fid['top_right'],
#             fid['bottom_right'],
#             fid['bottom_left']
#         ],
#         dtype=np.float32
#     )

#     H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#     if H is None:
#         raise RuntimeError("Homography failed")

#     W = cfg['sheet']['paper_size_px'][0]
#     Hh = cfg['sheet']['paper_size_px'][1]

#     warped = cv2.warpPerspective(img, H, (W, Hh))
#     return warped, H


# # ------------------------------------------------------
# # Bubble Darkness Detection
# # ------------------------------------------------------
# def mean_intensity_in_circle(gray, cx, cy, r):
#     H, W = gray.shape
#     x0, x1 = max(0, cx - r), min(W, cx + r)
#     y0, y1 = max(0, cy - r), min(H, cy + r)

#     patch = gray[y0:y1, x0:x1]
#     if patch.size == 0:
#         return 255.0

#     rr, cc = np.ogrid[:patch.shape[0], :patch.shape[1]]
#     mask = (rr - (cy - y0))**2 + (cc - (cx - x0))**2 <= r*r

#     if mask.sum() == 0:
#         return float(patch.mean())

#     return float(patch[mask].mean())


# def extract_bubble_darkness(warped, cfg):
#     if warped.ndim == 3:
#         gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = warped

#     q = cfg['questions']
#     start_x, start_y = q['start_position_px']
#     R = q['bubble_radius_px']
#     pad = q['bubble_padding_px']
#     col_gap = q['column_gap_px']
#     opts = q['options']
#     QPC = q['questions_per_column']
#     COLS = q['columns']

#     H_STEP = 2 * R + pad
#     V_STEP = 2 * R + pad

#     raw = {}
#     qid = 1
#     cx_base = start_x

#     for col in range(COLS):
#         cy = start_y

#         for _ in range(QPC):
#             raw[qid] = {}
#             for i, opt in enumerate(opts):
#                 cx = cx_base + R + i * H_STEP
#                 cyc = cy + R

#                 meanv = mean_intensity_in_circle(gray, cx, cyc, R)
#                 raw[qid][opt] = 255 - meanv

#             qid += 1
#             cy += V_STEP

#         cx_base += len(opts) * H_STEP + col_gap

#     return raw


# # ------------------------------------------------------
# # Classifier
# # ------------------------------------------------------
# def classify_bubbles(raw, offset=18.0):
#     selected = {}
#     status = {}

#     for q, vals in raw.items():
#         arr = np.array(list(vals.values()))
#         mean_d = arr.mean()
#         thresh = mean_d + offset

#         chosen = [opt for opt, v in vals.items() if v >= thresh]

#         if len(chosen) == 1:
#             selected[q] = chosen[0]
#             status[q] = "single"
#         elif len(chosen) > 1:
#             selected[q] = chosen
#             status[q] = "multiple"
#         else:
#             selected[q] = None
#             status[q] = "blank"

#     return selected, status


# # ------------------------------------------------------
# # Score
# # ------------------------------------------------------
# def compute_score(selected, anskey):
#     score = 0
#     for qid, correct in anskey.items():
#         if selected.get(qid) == correct:
#             score += 1
#     return score


# # ------------------------------------------------------
# # ** PERFECT ROLL NUMBER DETECTION (FROM YOUR COLAB) **
# # ------------------------------------------------------
# def detect_roll_number_bubbles(warped, cfg):
#     roll_cfg = cfg['metadata_fields'].get('roll_no')

#     if not roll_cfg or not roll_cfg.get('bubbles_below', False):
#         return None

#     if warped.ndim == 3:
#         gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = warped

#     boxes = roll_cfg['boxes']
#     box_w, box_h = roll_cfg['box_size_px']
#     start_x, start_y = roll_cfg['position_px']
#     gap = roll_cfg.get('gap_px', 0)

#     bubbles_count = roll_cfg['bubbles_count_below']
#     r = roll_cfg['bubble_radius_below']
#     pad = roll_cfg['bubble_padding_below']

#     gap_under_box = roll_cfg['bubble_padding_below']

#     digits = []

#     for idx in range(boxes):

#         x_box = start_x + idx * (box_w + gap)
#         y_box = start_y

#         cx = x_box + box_w // 2

#         start_y_bub = y_box + box_h + gap_under_box

#         best_digit = None
#         best_mean = 255

#         for d in range(bubbles_count):
#             cy = start_y_bub + d * (2 * r + pad) + r

#             meanv = mean_intensity_in_circle(gray, cx, cy, r)

#             if meanv < best_mean:
#                 best_mean = meanv
#                 best_digit = d

#         if best_mean < 160:
#             digits.append(str(best_digit))
#         else:
#             digits.append("")

#     final_roll = "".join(digits)
#     return final_roll if final_roll.strip() else None


# # ------------------------------------------------------
# # Main Processor Class
# # ------------------------------------------------------
# class OMRProcessor:
#     def __init__(self, template):
#         self.cfg = template

#     def process_image(self, image_path, incoming_answer_key=None):
#         anskey = normalize_answer_key(incoming_answer_key or {})

#         img = cv2.imread(image_path)
#         if img is None:
#             return {"error": "cannot_read_image", "score": 0}

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         expected_r = int(
#             self.cfg['fiducials']['outer_radius_px']
#             * (gray.shape[1] / self.cfg['sheet']['paper_size_px'][0])
#         )

#         cleaned = adaptive_thresh_and_morph(gray, expected_r)
#         hough = hough_fiducials(gray, expected_r)
#         cont = contour_fiducials(cleaned, expected_r)

#         merged = dedupe_candidates(hough + cont, min_center_dist=expected_r)
#         best4 = select_fiducial_candidates(merged, gray.shape)

#         if len(best4) != 4:
#             return {"error": "fiducials_not_found", "score": 0}

#         ordered = order_fiducials(best4)
#         warped, _ = compute_homography_and_warp(img, ordered, self.cfg)

#         raw = extract_bubble_darkness(warped, self.cfg)
#         selected, status = classify_bubbles(raw)

#         score = compute_score(selected, anskey)

#         roll_no = detect_roll_number_bubbles(warped, self.cfg)

#         return {
#             "score": score,
#             "detected": selected,
#             "status": status,
#             "roll_number": roll_no,
#             "max_score": len(anskey),
#             "error": None
#         }


# # ------------------------------------------------------
# # Factory for main.py
# # ------------------------------------------------------
# def get_default_processor(model_path=None, template_path="template.json"):
#     template = load_template(template_path)
#     return OMRProcessor(template)


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