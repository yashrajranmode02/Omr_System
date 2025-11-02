# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import json
# import os
# import sys

# # ===============================
# # Configuration
# # ===============================
# MODEL_PATH = "mnist_cnn1.h5"     # Your trained digit model
# TEMPLATE_PATH = "template.json"  # JSON file with roll number box coordinates

# def enhance_and_prepare(digit_img, size=(28, 28)):
#     """Prepare digit image for MNIST model"""
#     gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     resized = cv2.resize(thresh, size)
#     normalized = resized.astype("float32") / 255.0
#     normalized = np.expand_dims(normalized, axis=-1)
#     return np.expand_dims(normalized, axis=0)

# def predict_roll_number(image_path):
#     """Predict the roll number from OMR sheet"""
#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"❌ Could not read image from {image_path}")

#     # Load model
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
#     model = load_model(MODEL_PATH)

#     # Load roll number box coordinates
#     if not os.path.exists(TEMPLATE_PATH):
#         raise FileNotFoundError(f"❌ Template not found: {TEMPLATE_PATH}")
#     with open(TEMPLATE_PATH, "r") as f:
#         template = json.load(f)

#     roll_box = template["roll_number_box"]
#     x, y, w, h = roll_box["x"], roll_box["y"], roll_box["width"], roll_box["height"]
#     num_digits = roll_box["num_digits"]

#     # Crop and split into digits
#     roi = image[y:y+h, x:x+w]
#     digit_width = w // num_digits
#     digits = [roi[:, i*digit_width:(i+1)*digit_width] for i in range(num_digits)]

#     # Predict each digit
#     predictions = []
#     for i, digit_img in enumerate(digits):
#         input_data = enhance_and_prepare(digit_img)
#         pred = model.predict(input_data, verbose=0)
#         digit = np.argmax(pred)
#         predictions.append(str(digit))
#         print(f"Digit {i+1}: {digit} (confidence {np.max(pred):.2%})")

#     roll_number = "".join(predictions)
#     print(f"\n✅ Predicted Roll Number: {roll_number}")
#     return roll_number


# # ===============================
# # Entry point
# # ===============================
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python roll_predictor.py <omr_image_path>")
#         sys.exit(1)

#     input_image = sys.argv[1]
#     predict_roll_number(input_image)


import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os
import sys
import random

# ===============================
# Configuration
# ===============================
MODEL_PATH = "mnist_cnn1.h5"     # Trained digit model
TEMPLATE_PATH = "template.json"  # JSON file with roll number box coordinates


# ===============================
# Image Processing
# ===============================
def enhance_and_prepare(digit_img, size=(28, 28)):
    """Prepare digit image for MNIST model"""
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, size)
    normalized = resized.astype("float32") / 255.0
    normalized = np.expand_dims(normalized, axis=-1)
    return np.expand_dims(normalized, axis=0)


# ===============================
# Core Prediction Function
# ===============================
def predict_roll_number(image_path, template_path=TEMPLATE_PATH):
    """Predict the roll number from OMR sheet"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Could not read image from {image_path}")

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # Load template
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"❌ Template not found: {template_path}")
    with open(template_path, "r") as f:
        template = json.load(f)

    roll_box = template["roll_number_box"]
    x, y, w, h = roll_box["x"], roll_box["y"], roll_box["width"], roll_box["height"]
    num_digits = roll_box["num_digits"]

    # Crop and split into digits
    roi = image[y:y+h, x:x+w]
    digit_width = w // num_digits
    digits = [roi[:, i*digit_width:(i+1)*digit_width] for i in range(num_digits)]

    # Predict each digit
    predictions = []
    for i, digit_img in enumerate(digits):
        input_data = enhance_and_prepare(digit_img)
        pred = model.predict(input_data, verbose=0)
        digit = np.argmax(pred)
        predictions.append(str(digit))
        print(f"Digit {i+1}: {digit} (confidence {np.max(pred):.2%})")

    roll_number = "".join(predictions)
    print(f"\n✅ Predicted Roll Number: {roll_number}")
    return roll_number


# ===============================
# Helper Functions for Testing
# ===============================
def save_canonical_sheet(image_path):
    """Placeholder - directly returns same image"""
    # In future, you can align or preprocess here
    return image_path


def generate_captcha(length=4):
    """Generate a simple numeric CAPTCHA"""
    return ''.join(str(random.randint(0, 9)) for _ in range(length))


def save_result_json(roll_number, captcha):
    """Save roll number and captcha result as JSON"""
    result = {"roll_number": roll_number, "captcha": captcha}
    with open("roll_result.json", "w") as f:
        json.dump(result, f, indent=4)
    return "roll_result.json"


# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python roll_predictor.py <omr_image_path>")
        sys.exit(1)

    input_image = sys.argv[1]
    predict_roll_number(input_image)
