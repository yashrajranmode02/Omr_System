# # """
# # main.py - FastAPI server exposing /process-omr endpoint
# # Saves uploaded files to a timestamped folder and invokes omr_processor.process_folder
# # """

# # import os
# # import time
# # import shutil
# # from typing import List
# # from fastapi import FastAPI, UploadFile, File, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # import uvicorn

# # from omr_processor import get_default_processor

# # app = FastAPI(title="OMR Processing API")

# # # allow CORS from your frontend origin in production adjust accordingly
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # change to your frontend origin e.g. ["http://localhost:3000"]
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # UPLOAD_FOLDER = "uploads"
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # # Example answer key (you should replace with actual per-test answer keys)
# # # This is kept here for demo. In practice, client will send a JSON answer key or testId.
# # DEFAULT_ANSWER_KEY = {i + 1: 2 for i in range(60)}

# # # Initialize processor once (adjust model/template paths if needed)
# # processor = get_default_processor(model_path="best.pt", template_path="template.json")


# # @app.post("/process-omr")
# # async def process_omr(files: List[UploadFile] = File(...), use_answer_key: str = None):
# #     """
# #     Accepts one or multiple files.
# #     Optional query param use_answer_key isn't used here — just placeholder. In practice, you should
# #     pass the answer key in body (or reference a testId that maps to an answer key in DB).
# #     """
# #     if not files:
# #         raise HTTPException(status_code=400, detail="No files uploaded.")

# #     timestamp = str(int(time.time()))
# #     batch_folder = os.path.join(UPLOAD_FOLDER, timestamp)
# #     os.makedirs(batch_folder, exist_ok=True)

# #     saved_paths = []
# #     try:
# #         # Save files
# #         for f in files:
# #             # sanitize filename if needed
# #             filename = os.path.basename(f.filename)
# #             target_path = os.path.join(batch_folder, filename)
# #             with open(target_path, "wb") as buffer:
# #                 shutil.copyfileobj(f.file, buffer)
# #             saved_paths.append(target_path)

# #         # Decide which answer key to use
# #         # TODO: retrieve per-test answer_key from DB if you pass testId
# #         answer_key = DEFAULT_ANSWER_KEY

# #         # Call ML processing (synchronous) - it returns JSON-like dict
# #         results = processor.process_folder(batch_folder, answer_key)

# #         return {"status": "success", "batch_id": timestamp, "results": results}

# #     except Exception as e:
# #         # In case of any server error, ensure we cleanup if required (optional)
# #         # shutil.rmtree(batch_folder, ignore_errors=True)
# #         raise HTTPException(status_code=500, detail=f"Server error: {e}")
# # main.py - FastAPI server exposing /process-omr endpoint
# # Saves uploaded files to a timestamped folder and invokes omr_processor.process_folder
# # Also provides /process-omr-memory for in-memory processing w/o saving to disk.

# import os
# import time
# import shutil
# import json
# from typing import List, Optional
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from starlette.concurrency import run_in_threadpool
# import uvicorn
# import numpy as np
# import cv2

# from omr_processor import get_default_processor

# app = FastAPI(title="OMR Processing API")

# # allow CORS from your frontend origin in production - adjust accordingly
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # change to your frontend origin e.g. ["http://localhost:3000"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Example answer key (you should replace with actual per-test answer keys)
# DEFAULT_ANSWER_KEY = {        "1": 3,
#       "2": 1,
#       "3": 1,
#       "4": 2,
#       "5": 1,
#       "6": 0,
#       "7": 2,
#       "8": 3,
#       "9": 2,
#       "10": 2,
#       "11": 1,
#       "12": 0,
#       "13": 0,
#       "14": 2,
#       "15": 0,
#       "16": 2,
#       "17": 2,
#       "18": 2,
#       "19": 3,
#       "20": 0,
#       "21": 1,
#       "22": 1,
#       "23": 2,
#       "24": 2,
#       "25": 3,
#       "26": 1,
#       "27": 2,
#       "28": 3,
#       "29": 3,
#       "30": 2,
#       "31": 3,
#       "32": 2,
#       "33": 1,
#       "34": 1,
#       "35": 0,
#       "36": 2,
#       "37": 0,
#       "38": 0,
#       "39": 2,
#       "40": 0,
#       "41": 3,
#       "42": 3,
#       "43": 3,
#       "44": 1,
#       "45": 1,
#       "46": 3,
#       "47": 3,
#       "48": 3,
#       "49": 3,
#       "50": 3,
#       "51": 3,
#       "52": 3,
#       "53": 3,
#       "54": 3,
#       "55": 3,
#       "56": 3,
#       "57": 3,
#       "58": 3,
#       "59": 3,
#       "60": 3
# }

# # Initialize processor once (adjust model/template paths if needed)
# processor = get_default_processor(model_path="best.pt", template_path="template.json")

# # Safety / limits
# MAX_FILES_PER_REQUEST = 50


# @app.post("/process-omr")
# async def process_omr(
#     files: List[UploadFile] = File(...),
#     answer_key: Optional[str] = Form(None),  # client can send JSON stringified answer_key
# ):
#     """
#     Save uploaded files to a timestamped folder and process the whole folder with processor.process_folder.
#     Accepts optional form field `answer_key` as a JSON string.
#     """
#     if not files:
#         raise HTTPException(status_code=400, detail="No files uploaded.")
#     if len(files) > MAX_FILES_PER_REQUEST:
#         raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_FILES_PER_REQUEST}).")

#     # parse optional answer_key if provided
#     if answer_key:
#         try:
#             answer_key_obj = json.loads(answer_key)
#         except Exception:
#             raise HTTPException(status_code=400, detail="answer_key must be a valid JSON string.")
#     else:
#         answer_key_obj = DEFAULT_ANSWER_KEY

#     timestamp = str(int(time.time()))
#     batch_folder = os.path.join(UPLOAD_FOLDER, timestamp)
#     os.makedirs(batch_folder, exist_ok=True)

#     saved_paths = []
#     try:
#         # Save files to disk
#         for f in files:
#             # sanitize filename if needed - here we just take basename
#             filename = os.path.basename(f.filename)
#             target_path = os.path.join(batch_folder, filename)
#             # Save binary contents to disk
#             with open(target_path, "wb") as buffer:
#                 shutil.copyfileobj(f.file, buffer)
#             saved_paths.append(target_path)

#         # Run the blocking process_folder in a threadpool so the event loop isn't blocked
#         results = await run_in_threadpool(processor.process_folder, batch_folder, answer_key_obj)

#         return {"status": "success", "batch_id": timestamp, "results": results}

#     except Exception as e:
#         # Optionally cleanup on failure (commented out so you can inspect uploaded images)
#         # shutil.rmtree(batch_folder, ignore_errors=True)
#         raise HTTPException(status_code=500, detail=f"Server error: {e}")


# @app.post("/process-omr-memory")
# async def process_omr_memory(
#     files: List[UploadFile] = File(...),
#     answer_key: Optional[str] = Form(None),
# ):
#     """
#     Process uploaded files in-memory (no disk writes). Each file is decoded to an OpenCV BGR numpy array
#     and passed to processor.process_image_array.
#     This is good when you don't want to persist incoming uploads.
#     """
#     if not files:
#         raise HTTPException(status_code=400, detail="No files uploaded.")
#     if len(files) > MAX_FILES_PER_REQUEST:
#         raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_FILES_PER_REQUEST}).")

#     if answer_key:
#         try:
#             answer_key_obj = json.loads(answer_key)
#         except Exception:
#             raise HTTPException(status_code=400, detail="answer_key must be a valid JSON string.")
#     else:
#         answer_key_obj = DEFAULT_ANSWER_KEY

#     results = {}
#     for upload_file in files:
#         try:
#             contents = await upload_file.read()
#             np_arr = np.frombuffer(contents, np.uint8)
#             img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             if img is None:
#                 results[upload_file.filename] = {"error": "Unable to decode image"}
#                 continue

#             # call the (blocking) process_image_array inside threadpool
#             res = await run_in_threadpool(processor.process_image_array, img, answer_key_obj)
#             results[upload_file.filename] = res

#         except Exception as e:
#             results[upload_file.filename] = {"error": str(e)}

#     return {"status": "success", "results": results}


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import os
import time
import shutil
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import uvicorn
import numpy as np
import cv2

from omr_processor import get_default_processor

app = FastAPI(title="OMR Processing API")

# allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize processor once
processor = get_default_processor(model_path="best.pt", template_path="template.json")

# Safety / limits
MAX_FILES_PER_REQUEST = 50

@app.post("/process-omr")
async def process_omr(
    files: List[UploadFile] = File(...),
    answer_key: str = Form(...),  # required: user must provide 60 answers
):
    """
    Save uploaded files to a folder and process them.
    Requires answer_key as JSON string with exactly 60 answers.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_FILES_PER_REQUEST}).")

    # parse and validate answer_key
    try:
        answer_key_obj = json.loads(answer_key)
    except Exception:
        raise HTTPException(status_code=400, detail="answer_key must be a valid JSON string.")

    if len(answer_key_obj) != 60:
        raise HTTPException(status_code=400, detail="Answer key must contain exactly 60 answers.")

    timestamp = str(int(time.time()))
    batch_folder = os.path.join(UPLOAD_FOLDER, timestamp)
    os.makedirs(batch_folder, exist_ok=True)

    saved_paths = []
    try:
        for f in files:
            filename = os.path.basename(f.filename)
            target_path = os.path.join(batch_folder, filename)
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_paths.append(target_path)

        results = await run_in_threadpool(processor.process_folder, batch_folder, answer_key_obj)
        return {"status": "success", "batch_id": timestamp, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


@app.post("/process-omr-memory")
async def process_omr_memory(
    files: List[UploadFile] = File(...),
    answer_key: str = Form(...),  # required
):
    """
    Process uploaded files in-memory. Requires 60-answer key.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_FILES_PER_REQUEST}).")

    try:
        answer_key_obj = json.loads(answer_key)
    except Exception:
        raise HTTPException(status_code=400, detail="answer_key must be a valid JSON string.")

    if len(answer_key_obj) != 60:
        raise HTTPException(status_code=400, detail="Answer key must contain exactly 60 answers.")

    results = {}
    for upload_file in files:
        try:
            contents = await upload_file.read()
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                results[upload_file.filename] = {"error": "Unable to decode image"}
                continue

            res = await run_in_threadpool(processor.process_image_array, img, answer_key_obj)
            results[upload_file.filename] = res

        except Exception as e:
            results[upload_file.filename] = {"error": str(e)}

    return {"status": "success", "results": results}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)