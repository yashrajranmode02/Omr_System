from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
import os
import json
import time

from omr_processor import get_default_processor   # roll number included

app = FastAPI(title="OMR + Roll Number Processor")

# ✔ FIXED CORS middleware spelling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_sheets"
os.makedirs(UPLOAD_DIR, exist_ok=True)

OMR = get_default_processor(template_path="template.json")


@app.post("/process-omr")
async def process_omr(
    files: List[UploadFile] = File(...),
    answer_key: Optional[str] = Form(None)
):
    results = {}
    total_time = 0.0
    key_dict = {}

    if answer_key:
        try:
            key_dict = json.loads(answer_key)
        except Exception as e:
            return {"error": f"Invalid answer key JSON: {str(e)}"}

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📄 Received: {file.filename}")
        start_time = time.time()

        try:
            omr_result = OMR.process_image(file_path, key_dict)

            detected = omr_result.get("detected", {})
            score = omr_result.get("score", 0)
            status = omr_result.get("status", {})
            roll_number = omr_result.get("roll_number")
            error = omr_result.get("error")
        except Exception as e:
            detected = {}
            score = 0
            status = {}
            roll_number = None
            error = str(e)

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
