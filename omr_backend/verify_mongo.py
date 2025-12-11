from pymongo import MongoClient
import os
from dotenv import load_dotenv
import time

load_dotenv()

MONGO_STRING = os.getenv("MONGO_STRING")
if not MONGO_STRING:
    print("❌ MONGO_STRING not found in environment!")
    exit(1)

try:
    client = MongoClient(MONGO_STRING)
    db = client["EDAI"]
    collection = db["results"]
    
    count = collection.count_documents({})
    print(f"📊 Current document count: {count}")
    
    print("⏳ Waiting for new documents (10s)...")
    time.sleep(10)
    
    new_count = collection.count_documents({})
    print(f"📊 New document count: {new_count}")
    
    if new_count > count:
        print("✅ New documents added!")
        latest = collection.find_one(sort=[('_id', -1)])
        print("📄 Latest document keys:", latest.keys())
        print(f"📄 Latest Roll Number: {latest.get('rollNumber')}")
        print(f"📄 Latest File Name: {latest.get('fileName')}")
    else:
        print("⚠ No new documents detected.")

except Exception as e:
    print(f"❌ Error verifying MongoDB: {e}")
