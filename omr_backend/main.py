"""
main.py - FastAPI server exposing /process-omr endpoint
Saves uploaded files to a timestamped folder and invokes omr_processor.process_folder
"""

import os
import time
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from omr_processor import get_default_processor

app = FastAPI(title="OMR Processing API")

# allow CORS from your frontend origin in production adjust accordingly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Example answer key (you should replace with actual per-test answer keys)
# This is kept here for demo. In practice, client will send a JSON answer key or testId.
DEFAULT_ANSWER_KEY = {i + 1: 2 for i in range(60)}

# Initialize processor once (adjust model/template paths if needed)
processor = get_default_processor(model_path="best.pt", template_path="template.json")


@app.post("/process-omr")
async def process_omr(files: List[UploadFile] = File(...), use_answer_key: str = None):
    """
    Accepts one or multiple files.
    Optional query param use_answer_key isn't used here — just placeholder. In practice, you should
    pass the answer key in body (or reference a testId that maps to an answer key in DB).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    timestamp = str(int(time.time()))
    batch_folder = os.path.join(UPLOAD_FOLDER, timestamp)
    os.makedirs(batch_folder, exist_ok=True)

    saved_paths = []
    try:
        # Save files
        for f in files:
            # sanitize filename if needed
            filename = os.path.basename(f.filename)
            target_path = os.path.join(batch_folder, filename)
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_paths.append(target_path)

        # Decide which answer key to use
        # TODO: retrieve per-test answer_key from DB if you pass testId
        answer_key = DEFAULT_ANSWER_KEY

        # Call ML processing (synchronous) - it returns JSON-like dict
        results = processor.process_folder(batch_folder, answer_key)

        return {"status": "success", "batch_id": timestamp, "results": results}

    except Exception as e:
        # In case of any server error, ensure we cleanup if required (optional)
        # shutil.rmtree(batch_folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
