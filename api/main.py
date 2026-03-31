import sys
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import time, psutil, platform
import json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_PATH = os.path.join(BASE_DIR, "data", "job_descriptions.json")


from src.parser import extract_text_from_bytes
from src.engine import ResumeEngine

app = FastAPI()
engine = ResumeEngine()



def load_jobs():
    with open(JOBS_PATH, "r") as f:
        return json.load(f)

@app.get("/jobs")
def get_jobs():
    return load_jobs()

@app.post("/match")
async def match_resume(file: UploadFile = File(...), job_id: str = Form(...)):
    start_time = time.perf_counter()
    
    # 1. Load data
    jobs = load_jobs()
    job = next((j for j in jobs if j["role_id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job Role not found.")

    # 2. Extract Text
    content = await file.read()
    ext = file.filename.split(".")[-1].lower()
    resume_text = extract_text_from_bytes(content, ext)

    # 3. Analyze with Logic
    analysis = engine.analyze(resume_text, job)
    
    end_time = time.perf_counter()
    return {
        "analysis": analysis,
        "performance": {
            "latency": f"{round(end_time - start_time, 3)}s",
            "ram_usage": f"{round(psutil.Process().memory_info().rss / 1024**2, 2)} MB"
        }
    }