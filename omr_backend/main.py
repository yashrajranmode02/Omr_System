# # # main.py
# # from fastapi import FastAPI, UploadFile, File, Form
# # from fastapi.middleware.cors import CORSMiddleware
# # from typing import List, Optional
# # import shutil
# # import os
# # import json
# # import time
# # from roll_predictor import predict_roll_number
# # from omr_processor import get_default_processor

# # app = FastAPI(title="OMR + Roll Number Processor")

# # # ✅ FIXED CORS
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # UPLOAD_DIR = "uploaded_sheets"
# # os.makedirs(UPLOAD_DIR, exist_ok=True)

# # OMR = get_default_processor(model_path=None, template_path="template.json")


# # @app.post("/process-omr")
# # async def process_omr(files: List[UploadFile] = File(...), answer_key: Optional[str] = Form(None)):
# #     results = {}
# #     total_time = 0.0
# #     key_dict = {}

# #     if answer_key:
# #         try:
# #             key_dict = json.loads(answer_key)
# #         except Exception as e:
# #             return {"error": f"Invalid answer key JSON: {str(e)}"}

# #     for file in files:
# #         file_path = os.path.join(UPLOAD_DIR, file.filename)

# #         with open(file_path, "wb") as buffer:
# #             shutil.copyfileobj(file.file, buffer)

# #         print(f"📄 Received: {file.filename}")
# #         start_time = time.time()

# #         try:
# #             omr_result = OMR.process_image(file_path, key_dict)
# #             detected = omr_result.get("detected", {})
# #             score = omr_result.get("score", 0)
# #             status = omr_result.get("status", {})
# #             error = omr_result.get("error")
# #         except Exception as e:
# #             detected = {}
# #             score = 0
# #             status = {}
# #             error = str(e)

# #         try:
# #             roll_number = predict_roll_number(file_path)
# #         except:
# #             roll_number = None

# #         processing_time = round(time.time() - start_time, 3)
# #         total_time += processing_time

# #         results[file.filename] = {
# #             "roll_number": roll_number,
# #             "score": score,
# #             "detected": detected,
# #             "status": status,
# #             "error": error,
# #             "processing_time_sec": processing_time
# #         }

# #     results["_summary"] = {
# #         "files_processed": len(files),
# #         "total_time_sec": round(total_time, 3),
# #         "avg_time_sec": round(total_time / len(files), 3) if files else 0
# #     }

# #     return results


# # @app.get("/")
# # def home():
# #     return {"message": "✅ OMR Processor API is running! Use /process-omr to upload sheets."}
# # main.py

# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List, Optional
# import shutil
# import os
# import json
# import time

# from omr_processor import get_default_processor

# app = FastAPI(title="OMR + Roll Number Processor")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_DIR = "uploaded_sheets"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# OMR = get_default_processor(model_path=None, template_path="template.json")


# @app.post("/process-omr")
# async def process_omr(files: List[UploadFile] = File(...), answer_key: Optional[str] = Form(None)):
#     results = {}
#     total_time = 0.0
#     key_dict = {}

#     if answer_key:
#         try:
#             key_dict = json.loads(answer_key)
#         except Exception as e:
#             return {"error": f"Invalid answer key JSON: {str(e)}"}

#     for file in files:
#         file_path = os.path.join(UPLOAD_DIR, file.filename)

#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         print(f"📄 Received: {file.filename}")
#         start_time = time.time()

#         try:
#             omr_result = OMR.process_image(file_path, key_dict)
#             detected = omr_result.get("detected", {})
#             score = omr_result.get("score", 0)
#             status = omr_result.get("status", {})
#             roll_number = omr_result.get("roll_number")   # ✅ USE BUBBLE-BASED ROLL NUMBER
#             error = omr_result.get("error")
#         except Exception as e:
#             detected = {}
#             score = 0
#             status = {}
#             roll_number = None
#             error = str(e)

#         processing_time = round(time.time() - start_time, 3)
#         total_time += processing_time

#         results[file.filename] = {
#             "roll_number": roll_number,
#             "score": score,
#             "detected": detected,
#             "status": status,
#             "error": error,
#             "processing_time_sec": processing_time
#         }

#     results["_summary"] = {
#         "files_processed": len(files),
#         "total_time_sec": round(total_time, 3),
#         "avg_time_sec": round(total_time / len(files), 3) if files else 0
#     }

#     return results


# @app.get("/")
# def home():
#     return {"message": "✅ OMR Processor API is running! Use /process-omr to upload sheets."}

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
import os
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

from omr_processor import get_default_processor

load_dotenv()

# -------------------------------------------------------------
# APP SETUP
# -------------------------------------------------------------
app = FastAPI(title="OMR + Roll Number Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_sheets"
DEBUG_DIR = "debug"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ✅ Load OMR engine ONCE
OMR = get_default_processor(template_path="template.json")

# -------------------------------------------------------------
# ✅ MongoDB Setup
# -------------------------------------------------------------
MONGO_STRING = os.getenv("MONGO_STRING")
db_collection = None

if MONGO_STRING:
    try:
        client = MongoClient(MONGO_STRING)
        db = client["EDAI"]
        db_collection = db["results"]
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
else:
    print("⚠ MONGO_STRING not found in environment variables")

executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------------------
# ✅ HELPERS
# -------------------------------------------------------------
def run_omr_sync(image_path, key_dict):
    return OMR.process_image(image_path, key_dict)

def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_keys(x) for x in obj]
    else:
        return obj

# ✅ FULL FLATTEN (FIXES YOUR ERROR)
def flatten_files(files):
    flat = []
    for f in files:
        if isinstance(f, list):
            flat.extend(f)
        else:
            flat.append(f)
    return flat

# -------------------------------------------------------------
# ✅ MAIN ENDPOINT (100% SAFE NOW)
# -------------------------------------------------------------
@app.post("/process-omr")
async def process_omr(
    files: List[UploadFile] = File(...),
    answer_key: Optional[str] = Form(None)
):
    print("🔔 /process-omr called")

    # ✅ FIX NESTED LIST ISSUE
    files = flatten_files(files)
    print("✅ FILES RECEIVED AFTER FLATTEN:", len(files))

    results = {}
    total_time = 0.0
    key_dict = {}

    if answer_key:
        try:
            key_dict = json.loads(answer_key)
            print("✅ Parsed answer key")
        except Exception as e:
            return {"error": f"Invalid answer key JSON: {str(e)}"}

    loop = asyncio.get_running_loop()

    for idx, file in enumerate(files):
        if not hasattr(file, "filename"):
            print("⚠ Skipping invalid file object:", type(file))
            continue

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        finally:
            await file.close()

        print(f"📄 [{idx+1}/{len(files)}] Saved: {file.filename}")
        start_time = time.time()

        try:
            omr_result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    run_omr_sync,
                    file_path,
                    key_dict
                ),
                timeout=30.0
            )

            detected = omr_result.get("detected", {})
            score = omr_result.get("score", 0)
            comments = omr_result.get("comments", {})
            roll_number = omr_result.get("roll_number")
            error = omr_result.get("error")

            print(f"✅ OMR done for {file.filename}")

        except Exception as e:
            detected = {}
            score = 0
            comments = {}
            roll_number = None
            error = str(e)
            print(f"❌ OMR failed for {file.filename}: {e}")

        processing_time = round(time.time() - start_time, 3)
        total_time += processing_time

        results[file.filename] = {
            "roll_number": roll_number,
            "score": score,
            "detected": detected,
            "comments": comments,
            "error": error,
            "processing_time_sec": processing_time
        }

        # ✅ SAVE TO MONGODB
        if db_collection is not None:
            try:
                doc = {
                    "fileName": file.filename,
                    "rollNumber": roll_number,
                    "score": score,
                    "detected": stringify_keys(detected),
                    "comments": stringify_keys(comments),
                    "error": error,
                    "processingTime": processing_time,
                    "timestamp": datetime.utcnow()
                }
                db_collection.insert_one(doc)
                print(f"💾 Saved to MongoDB: {file.filename}")
            except Exception as e:
                print(f"❌ Mongo save failed: {e}")

    results["_summary"] = {
        "files_processed": len(results),
        "total_time_sec": round(total_time, 3),
        "avg_time_sec": round(total_time / max(len(results), 1), 3)
    }

    print("🏁 Done")
    return results

# -------------------------------------------------------------
# ✅ HEALTH CHECK
# -------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "✅ OMR Processor API is running! Use /process-omr to upload sheets."}
