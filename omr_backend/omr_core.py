import numpy as np
import cv2


# ----------------------------
# Default config (updated)
# ----------------------------
DEFAULT_CONFIG = {
  "sheet": {
    "paper_size_px": [2500, 3500]
  },
  "metadata_fields": {
    "grade": {
      "boxes": 2,
      "label": "Grade",
      "position_px": [250, 1350],
      "box_size_px": [120, 120],
      "gap_px": 10
    },
    "division": {
      "boxes": 1,
      "label": "Division",
      "position_px": [650, 1350],
      "box_size_px": [120, 120],
      "gap_px": 0
    },
    "roll_no": {
      "boxes": 3,
      "label": "ROLL NO.",
      "position_px": [250, 1900],
      "box_size_px": [150, 150],
      "gap_px": 20,
      "bubbles_below": "true",
      "bubble_radius_below": 30,
      "bubble_padding_below": 30,
      "bubbles_count_below": 10
    },
    "subject_code": {
      "boxes": 5,
      "label": "Subject Code",
      "position_px": [250, 900],
      "box_size_px": [100, 100],
      "gap_px": 0
    },
    "date": {
      "boxes": 8,
      "label": "Date",
      "position_px": [250, 1600],
      "box_size_px": [100, 100],
      "gap_px": 8
    },
    "name": {
      "boxes": 20,
      "label": "NAME",
      "position_px": [250, 1100],
      "box_size_px": [100, 100],
      "gap_px": 0
    }
  },
  "signatures": {},
  "questions": {
    "start_position_px": [1350, 1400],
    "columns": 2,
    "questions_per_column": 20,
    "bubble_radius_px": 30,
    "bubble_padding_px": 30,
    "options": ["A", "B", "C", "D"],
    "column_gap_px": 200
  },
  "fiducials": {
    "outer_radius_px": 70,
    "inner_radius_px": 0,
    "positions_px": {
      "top_left": [100, 100],
      "top_right": [2400, 100],
      "bottom_left": [100, 3400],
      "bottom_right": [2400, 3400]
    }
  },
  "grid_box": {
    "line_thickness_px": 2
  },
  "image_box": {
    "position_px": [250, 250],
    "size_px": [500, 500],
    "label": ""
  },
  "instructions": {
    "start_position_px": [900, 250],
    "size_px": [1400, 750],
    "border_thickness_px": 2,

    "text_markdown_lines": [
      "## *Instructions*:\n",
      "- Fill each bubble completely using a dark pencil or pen.",
      "- Do not make stray marks or fold the answer sheet.",
      "- Ensure the filled bubble corresponds to your chosen option.",
      "- Write your name and roll number clearly.",
      "- Double-check before submitting."
    ],

    "instructions_font_scale": 1.2,
    "instructions_font_thickness": 2
  }
}

# ----------------------------
# Drawing helpers
# ----------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = 0  # black
import markdown
from bs4 import BeautifulSoup
def markdown_to_plain_text(md_text):
    """Convert markdown to readable plain text for drawing with cv2."""
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def get_markdown_text(instr):
    """
    Convert text_markdown_lines -> single markdown string.
    If user provides text_markdown instead, still support it.
    """
    if "text_markdown_lines" in instr:
        return "\n".join(instr["text_markdown_lines"])
    return instr.get("text_markdown", "")


def _draw_multiline_text_cv2(img, top_left, max_width, text, font_scale=0.6, thickness=1, line_height_px=18):
    """
    Draws multiline text using OpenCV. Text will be wrapped roughly by max_width (in px).
    This is a basic wrapper — markdown formatting is preserved as raw text.
    """
    x0, y0 = top_left
    words = text.split()
    line = ""
    y = y0
    for w in words:
        test = (line + " " + w).strip()
        (tw, th), _ = cv2.getTextSize(test, FONT, font_scale, thickness)
        if tw > max_width and line != "":
            cv2.putText(img, line, (x0, y), FONT, font_scale, TEXT_COLOR, thickness)
            line = w
            y += line_height_px
        else:
            line = test
    if line != "":
        cv2.putText(img, line, (x0, y), FONT, font_scale, TEXT_COLOR, thickness)

def fit_and_paste_image(base_img, img_to_place, top_left, box_size, label=None):
    """
    Paste the image STRETCHED to EXACT box_size (no aspect-ratio preservation).
    If no image is provided, draw placeholder.
    """
    x, y = top_left
    box_w, box_h = box_size

    if img_to_place is None:
        # Draw placeholder
        cv2.rectangle(base_img, (x, y), (x + box_w, y + box_h), 0, 2)
        if label:
            cv2.putText(base_img, label, (x, y - 10), FONT, 0.8, TEXT_COLOR, 2)
        return base_img

    # Convert to grayscale if needed
    if len(img_to_place.shape) == 3:
        gray = cv2.cvtColor(img_to_place, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_to_place

    # STRETCH to exact size
    resized = cv2.resize(gray, (box_w, box_h), interpolation=cv2.INTER_AREA)

    base_img[y: y + box_h, x: x + box_w] = resized

    # Border + label
    cv2.rectangle(base_img, (x, y), (x + box_w, y + box_h), 0, 2)
    if label:
        cv2.putText(base_img, label, (x, y - 10), FONT, 0.8, TEXT_COLOR, 2)

    return base_img

# ----------------------------
# Core drawing functions
# ----------------------------
def draw_fiducials(img, config):
    fid_cfg = config.get('fiducials', {})
    outer_r_px = fid_cfg.get('outer_radius_px', 70)
    inner_r_px = fid_cfg.get('inner_radius_px', 0)
    for px_coords in fid_cfg.get('positions_px', {}).values():
        center_px = tuple(px_coords)
        cv2.circle(img, center_px, outer_r_px, 0, -1)
        if inner_r_px > 0:
            cv2.circle(img, center_px, inner_r_px, 255, -1)

def draw_metadata_boxes(img, config):
    """
    Draw metadata boxes and handle special roll_no behaviour:
    - For metadata_fields.roll_no with bubbles_below = True, draw exactly 10 VERTICAL bubbles
      under each roll-number box, labelled 0..9 (top->bottom).
    - This function intentionally does NOT draw any horizontal bubbles.
    - Signatures are optional: skipped if config['signatures'] is empty or missing.
    """
    line_thickness = config.get('grid_box', {}).get('line_thickness_px', 2)

    # Draw signatures only if provided (optional)
    signatures = config.get('signatures') or {}
    for sig_key, sig_data in signatures.items():
        if not sig_data:
            continue
        sx, sy = sig_data['position_px']
        w, h = sig_data['size_px']
        label = sig_data.get('label', "")
        cv2.putText(img, label, (sx, sy - 30), FONT, 0.9, TEXT_COLOR, 2)
        cv2.rectangle(img, (sx, sy), (sx + w, sy + h), 0, line_thickness)

    # Draw metadata fields
    for field_key, field_data in config.get('metadata_fields', {}).items():
        sx, sy = field_data['position_px']
        box_w, box_h = field_data['box_size_px']
        gap_px = field_data.get('gap_px', 0)
        num_boxes = int(field_data['boxes'])
        cv2.putText(img, field_data['label'], (sx, sy - 30), FONT, 0.9, TEXT_COLOR, 2)

        cx = sx
        for i in range(num_boxes):
            # Draw the main rectangular box for this digit/field
            cv2.rectangle(img, (cx, sy), (cx + box_w, sy + box_h), 0, line_thickness)

            # Special handling for roll_no: draw vertical column of 10 bubbles under each box
            if field_key == 'roll_no' and field_data.get('bubbles_below', False):
                # Force 10 bubbles (0..9)
                bubbles_count = 10

                # Read radius/padding from config with sensible defaults
                r = int(field_data.get('bubble_radius_below', 10))
                pad = int(field_data.get('bubble_padding_below', 6))

                # vertical gap between bottom of box and first bubble
                gap_under_box = int(field_data.get('bubble_padding_below', 12))

                # Start Y just below the box
                start_y_b = sy + box_h + gap_under_box

                # Center the vertical stack under the box horizontally
                center_x = cx + box_w // 2

                # Draw 10 bubbles vertically top-to-bottom, each labelled 0..9
                for b in range(bubbles_count):
                    center_y = start_y_b + b * (2 * r + pad) + r
                    c_x = int(center_x)
                    c_y = int(center_y)
                    # Draw bubble circle (outline)
                    cv2.circle(img, (c_x, c_y), int(r), 0, 2)

                    # Draw the digit label centered inside the bubble
                    digit_label = str(b)  # 0..9
                    BUBBLE_FONT_SCALE = max(0.45, min(1.0, r / 12.0))
                    text_size = cv2.getTextSize(digit_label, FONT, BUBBLE_FONT_SCALE, 1)[0]
                    text_x = c_x - (text_size[0] // 2)
                    text_y = c_y + (text_size[1] // 2)
                    cv2.putText(img, digit_label, (text_x, text_y), FONT, BUBBLE_FONT_SCALE, TEXT_COLOR, 1)

            # advance to next field box horizontally
            cx += box_w + gap_px


def draw_image_box(img, config, pil_image=None):
    ib = config.get('image_box')
    if not ib:
        return img
    pos = tuple(ib['position_px'])
    size = tuple(ib['size_px'])
    label = ib.get('label', None)
    if pil_image is None:
        return fit_and_paste_image(img, None, pos, size, label)
    # convert PIL to OpenCV BGR then pass to fit_and_paste_image
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return fit_and_paste_image(img, cv_img, pos, size, label)

def draw_question_grid(img, config):
    """
    Draws the question grid using explicit start_position_px from config.
    - Bubbles are drawn as outlines, without option letters inside.
    - Option labels (A/B/C/...) are printed once in bold above each option column,
      centered over that option's vertical stack of bubbles.
    """
    q_cfg = config['questions']
    start_x_px, start_y_px = q_cfg['start_position_px']  # user-defined now
    R_px = q_cfg['bubble_radius_px']
    padding_px = q_cfg['bubble_padding_px']
    col_gap_px = q_cfg['column_gap_px']
    line_thickness = config['grid_box']['line_thickness_px']

    questions_per_column = q_cfg['questions_per_column']
    options = q_cfg['options']
    num_options = len(options)

    V_STEP = 2 * R_px + padding_px
    H_STEP = 2 * R_px + padding_px

    # Header style (change these to taste)
    HEADER_FONT_SCALE = 1.0
    HEADER_THICKNESS = 3
    HEADER_OFFSET = 30  # pixels above the first row of bubbles

    current_col_start_x = start_x_px
    q_num = 1

    # Grid tracking variables (to calculate final contour)
    min_x = start_x_px - 100
    max_x = 0
    min_y = start_y_px - 50
    max_y = 0

    for col in range(q_cfg['columns']):
        # Draw option headers once for this main column
        for i, option_char in enumerate(options):
            header_center_x = current_col_start_x + R_px + i * H_STEP
            header_y = start_y_px - HEADER_OFFSET
            # center the option label on the header_center_x
            text_size = cv2.getTextSize(option_char, FONT, HEADER_FONT_SCALE, HEADER_THICKNESS)[0]
            text_x = int(header_center_x - text_size[0] / 2)
            text_y = int(header_y + text_size[1] / 2)
            cv2.putText(img, option_char, (text_x, text_y), FONT, HEADER_FONT_SCALE, TEXT_COLOR, HEADER_THICKNESS)

        current_y_px = start_y_px

        for q in range(questions_per_column):
            # Draw Question Number: Consistent FONT and FONT_SCALE
            q_label = str(q_num).zfill(2)
            q_label_x = current_col_start_x - 100
            q_label_y = current_y_px + R_px + 10
            cv2.putText(img, q_label, (q_label_x, q_label_y), FONT, 1, TEXT_COLOR, 3)

            # Draw the options (bubbles) — no letters inside
            for i in range(num_options):
                center_x_px = current_col_start_x + R_px + i * H_STEP
                center_y_px = current_y_px + R_px

                # Update max X/Y for bounding box calculation (based on the bubble's right edge)
                if center_x_px + R_px > max_x:
                    max_x = center_x_px + R_px
                if center_y_px + R_px > max_y:
                    max_y = center_y_px + R_px

                # Draw the empty bubble (circle outline)
                cv2.circle(img, (int(center_x_px), int(center_y_px)), int(R_px), 0, 3)

            current_y_px += V_STEP
            q_num += 1

        # Calculate the starting X-coordinate for the next main column
        col_content_width_px = num_options * H_STEP
        current_col_start_x += col_content_width_px + col_gap_px

    # --- Draw Contour around the entire Question Grid ---
    pt1_grid = (min_x - 50, min_y - 50)
    pt2_grid = (max_x + 50, max_y + 50)
    cv2.rectangle(img, pt1_grid, pt2_grid, 0, 5)

def draw_instructions_box(img, config):
    instr = config.get('instructions')
    if not instr:
        return img

    sx, sy = instr['start_position_px']
    w, h = instr['size_px']
    border = instr.get('border_thickness_px', 2)

    # --- Load user-defined font settings ---
    font_scale = instr.get("instructions_font_scale", 0.7)
    font_thickness = instr.get("instructions_font_thickness", 1)

    # --- Get markdown as plain text ---
    raw_md = get_markdown_text(instr)
    rendered_text = markdown_to_plain_text(raw_md)

    # Draw border
    cv2.rectangle(img, (sx, sy), (sx + w, sy + h), 0, border)

    # Margins
    margin = 50
    x0 = sx + margin
    y0 = sy + margin
    available_width = w - 2 * margin

    # Draw wrapped text
    _draw_multiline_text_cv2(
        img,
        (x0, y0),
        available_width,
        rendered_text,
        font_scale=font_scale,
        thickness=font_thickness,
        line_height_px=int(30 * font_scale)
    )

    return img

def generate_omr_sheet(config, pil_image=None):
    W, H = config['sheet']['paper_size_px']
    canvas = np.full((H, W), 255, dtype=np.uint8)
    draw_fiducials(canvas, config)
    draw_metadata_boxes(canvas, config)
    draw_image_box(canvas, config, pil_image)
    draw_instructions_box(canvas, config)
    draw_question_grid(canvas, config)
    return canvas
