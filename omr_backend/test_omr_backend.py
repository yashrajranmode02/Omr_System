import requests
import os

# URL of your FastAPI server
url = "http://127.0.0.1:8000/process-omr"

# Folder containing OMR sheets for testing
test_folder = "test_omr_sheets"
file_paths = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

# Prepare files for multipart/form-data
files = [("files", open(fp, "rb")) for fp in file_paths]

response = requests.post(url, files=files)
print(response.json())

# Close files
for _, f in files:
    f.close()
