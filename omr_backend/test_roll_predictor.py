# """
# Standalone test script for roll_predictor.py
# Run this to test roll number prediction independently before integration.

# Usage:
#     python test_roll_predictor.py [image_path]
    
# Example:
#     python test_roll_predictor.py test_omr_sheets/OMR\ Sheet\ -\ from\ Nupoor\ updated.jpg
# """

# import sys
# import os
# from roll_predictor import (
#     save_canonical_sheet,
#     predict_roll_number,
#     generate_captcha,
#     save_result_json
# )

# def main():
#     # Default test image if none provided
#     if len(sys.argv) > 1:
#         image_path = sys.argv[1]
#     else:
#         # Use first available test image
#         test_images = [
#             "test_omr_sheets/OMR Sheet - from Nupoor updated.jpg",
#             "test_omr_sheets/40 ques omr_1.jpg",
#             "test_omr_sheets/40 ques omr_1(1).jpg"
#         ]
#         image_path = None
#         for img in test_images:
#             if os.path.exists(img):
#                 image_path = img
#                 break
        
#         if not image_path:
#             print("❌ No test image found. Please provide an image path:")
#             print("   python test_roll_predictor.py <image_path>")
#             sys.exit(1)
    
#     # Check if image exists
#     if not os.path.exists(image_path):
#         print(f"❌ Image not found: {image_path}")
#         sys.exit(1)
    
#     print(f"\n{'='*60}")
#     print(f"🧪 Testing Roll Predictor")
#     print(f"{'='*60}")
#     print(f"📸 Input image: {image_path}\n")
    
#     # Check if required files exist
#     required_files = {
#         "best.pt": "YOLO model for fiducial detection",
#         "mnist_cnn1.h5": "MNIST model for digit recognition",
#         "template.json": "Template configuration"
#     }
    
#     missing_files = []
#     for file, desc in required_files.items():
#         if not os.path.exists(file):
#             missing_files.append(f"  - {file} ({desc})")
    
#     if missing_files:
#         print("❌ Missing required files:")
#         for f in missing_files:
#             print(f)
#         sys.exit(1)
    
#     try:
#         # STEP 1: Create canonical sheet
#         print("STEP 1: Creating canonical sheet from uploaded image...")
#         canonical_path = save_canonical_sheet(image_path)
        
#         if not os.path.exists(canonical_path):
#             print(f"❌ Failed to create canonical sheet at {canonical_path}")
#             sys.exit(1)
        
#         print(f"✅ Canonical sheet created: {canonical_path}\n")
        
#         # STEP 2: Predict roll number
#         print("STEP 2: Predicting roll number from canonical sheet...")
#         try:
#             roll_number = predict_roll_number(canonical_path, "template.json")
#             print(f"✅ Predicted Roll Number: {roll_number}\n")
#         except KeyError as e:
#             if "roll_number_box" in str(e):
#                 print("❌ Error: template.json is missing 'roll_number_box' configuration!")
#                 print("\nPlease add this to template.json:")
#                 print("""
#     "roll_number_box": {
#         "x": <x_coordinate>,
#         "y": <y_coordinate>,
#         "width": <width>,
#         "height": <height>,
#         "num_digits": <number_of_digits>
#     }
#                 """)
#                 print("Tip: Check the canonical_sheet.jpeg file to find the exact coordinates.")
#                 sys.exit(1)
#             else:
#                 raise
#         except Exception as e:
#             print(f"❌ Error predicting roll number: {e}")
#             print("\nPossible issues:")
#             print("  - Roll number box coordinates in template.json may be incorrect")
#             print("  - Check canonical_sheet.jpeg to verify the roll number area")
#             sys.exit(1)
        
#         # STEP 3: Generate CAPTCHA
#         print("STEP 3: Generating CAPTCHA...")
#         captcha = generate_captcha()
#         print(f"✅ CAPTCHA: {captcha}\n")
        
#         # STEP 4: Save results
#         print("STEP 4: Saving results...")
#         result = save_result_json(roll_number, captcha)
        
#         # Final summary
#         print(f"\n{'='*60}")
#         print("✅ TEST COMPLETED SUCCESSFULLY!")
#         print(f"{'='*60}")
#         print(f"📋 Results:")
#         print(f"   Roll Number: {roll_number}")
#         print(f"   CAPTCHA: {captcha}")
#         print(f"   JSON saved: roll_result.json")
#         print(f"   Canonical sheet: {canonical_path}")
#         print(f"{'='*60}\n")
        
#     except Exception as e:
#         print(f"\n❌ TEST FAILED: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

# if __name__ == "__main__":
#     main()



"""
Standalone test script for roll_predictor.py
Run this to test roll number prediction independently.

Usage:
    python test_roll_predictor.py [image_path]

Example:
    python test_roll_predictor.py "test_omr_sheets/40 ques omr_1.jpg"
"""

import sys
import os
from roll_predictor import (
    save_canonical_sheet,
    predict_roll_number,
    generate_captcha,
    save_result_json
)

def main():
    # Default test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        test_images = [
            "test_omr_sheets/40 ques omr_1.jpg",
            "test_omr_sheets/40 ques omr_1(1).jpg"
        ]
        image_path = next((img for img in test_images if os.path.exists(img)), None)

    if not image_path:
        print("❌ No test image found. Please provide an image path.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"🧪 Testing Roll Predictor")
    print(f"{'='*60}")
    print(f"📸 Input image: {image_path}\n")

    required_files = {
        "mnist_cnn1.h5": "MNIST model for digit recognition",
        "template.json": "Template configuration"
    }

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"  - {f} ({required_files[f]})")
        sys.exit(1)

    try:
        print("STEP 1: Creating canonical sheet...")
        canonical_path = save_canonical_sheet(image_path)
        print(f"✅ Canonical sheet ready: {canonical_path}\n")

        print("STEP 2: Predicting roll number...")
        roll_number = predict_roll_number(canonical_path)
        print(f"✅ Predicted Roll Number: {roll_number}\n")

        print("STEP 3: Generating CAPTCHA...")
        captcha = generate_captcha()
        print(f"✅ CAPTCHA: {captcha}\n")

        print("STEP 4: Saving results...")
        save_result_json(roll_number, captcha)

        print(f"\n{'='*60}")
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"📋 Results:")
        print(f"   Roll Number: {roll_number}")
        print(f"   CAPTCHA: {captcha}")
        print(f"   JSON saved: roll_result.json")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
