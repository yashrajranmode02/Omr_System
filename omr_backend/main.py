# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
import os
import json
import time
from roll_predictor import predict_roll_number  # keep your existing roll predictor
from omr_processor import get_default_processor

app = FastAPI(title="OMR + Roll Number Processor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = "uploaded_sheets"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize processor once (YOLO not required)
# template.json must be present in working directory
OMR = get_default_processor(model_path=None, template_path="template.json")

@app.post("/process-omr")
async def process_omr(files: List[UploadFile] = File(...), answer_key: Optional[str] = Form(None)):
    results = {}
    total_time = 0.0
    key_dict = {}
    if answer_key:
        try:
            parsed = json.loads(answer_key)
            key_dict = parsed
        except Exception as e:
            return {"error": f"Invalid answer key JSON: {str(e)}"}

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger_msg = f"📄 Received: {file.filename}"
        print(logger_msg)
        start_time = time.time()

        # OMR processing
        try:
            omr_result = OMR.process_image(file_path, key_dict)
            detected = omr_result.get("detected", {})
            score = omr_result.get("score", 0)
            status = omr_result.get("status", {})
            error = omr_result.get("error", None)
        except Exception as e:
            detected = {}
            score = 0
            status = {}
            error = str(e)

        # roll number (optional)
        try:
            roll_number = predict_roll_number(file_path)
        except Exception as e:
            roll_number = None

        processing_time = round(time.time() - start_time, 3)
        total_time += processing_time

        results[file.filename] = {
            "roll_number": roll_number,
            "score": score,
            "detected": detected,
            "status": status,
            "error": error,
            "processing_time_sec": processing_time
        }

    results["_summary"] = {
        "files_processed": len(files),
        "total_time_sec": round(total_time, 3),
        "avg_time_sec": round(total_time / len(files), 3) if files else 0
    }

    return results

@app.get("/")
def home():
    return {"message": "✅ OMR Processor API is running! Use /process-omr to upload sheets."}
