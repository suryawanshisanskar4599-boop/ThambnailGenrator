import os
import shutil
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve
from werkzeug.utils import secure_filename

# backend.py currently relies on relative output directories.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from backend import get_output, main, progress_data

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDERS = ["Categorized_Thumbnails", "Top_Recommended"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# backend.py uses shared progress and shared output folders, so processing must
# remain single-job to avoid cross-job corruption.
jobs = {}
jobs_lock = threading.Lock()
processing_lock = threading.Lock()


def reset_progress():
    progress_data["percent"] = 0
    progress_data["status"] = "Idle"


def has_active_job():
    return any(not job["done"] for job in jobs.values())


def run_ai_task(job_id, filepath):
    """Process a single upload in a background thread."""
    with processing_lock:
        try:
            reset_progress()
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id].update({
                        "progress": 1,
                        "status": "Starting analysis..."
                    })

            main(filepath)
            categories, top = get_output()

            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id].update({
                        "progress": 100,
                        "status": "Analysis Complete",
                        "done": True,
                        "result": {
                            "categories": categories,
                            "top": top,
                        },
                    })
        except Exception as exc:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id].update({
                        "progress": 0,
                        "status": f"Error: {exc}",
                        "done": True,
                        "result": None,
                    })
        finally:
            reset_progress()
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass


@app.route("/")
def home():
    return "Premium AI Engine Online"


@app.route("/init_upload", methods=["POST"])
def init_upload():
    with jobs_lock:
        if has_active_job():
            return jsonify({
                "error": "Another video is already being processed. Please wait for it to finish."
            }), 409
            
        # Optimization: Free up RAM by deleting old completed jobs before starting a new one
        completed_jobs = [jid for jid, j in jobs.items() if j["done"]]
        for jid in completed_jobs:
            del jobs[jid]

    job_id = str(uuid.uuid4())
    return jsonify({"job_id": job_id}), 200


@app.route("/upload_chunk", methods=["POST"])
def upload_chunk():
    chunk_index = int(request.form.get("chunk_index", 0))
    total_chunks = int(request.form.get("total_chunks", 1))
    job_id = request.form.get("job_id")
    filename = request.form.get("filename", "upload.bin")

    if "file" not in request.files:
        return jsonify({"error": "No chunk uploaded"}), 400

    chunk_file = request.files["file"]
    
    # Store chunks in a temporary directory unique to this job
    temp_dir = os.path.join(UPLOAD_FOLDER, f"temp_{job_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index}")
    try:
        chunk_file.save(chunk_path)
    except Exception as exc:
        return jsonify({"error": f"Failed to save chunk: {exc}"}), 500

    # Check if all chunks are physically present (fixes out-of-order chunk arrivals)
    if len(os.listdir(temp_dir)) == total_chunks:
        # Assemble the final file
        safe_name = secure_filename(filename)
        ext = os.path.splitext(filename)[1]
        if not safe_name or "." not in safe_name:
            safe_name = f"upload{ext}"
        
        final_filename = f"{int(time.time())}_{safe_name}"
        final_filepath = os.path.join(UPLOAD_FOLDER, final_filename)
        
        try:
            if total_chunks == 1:
                # Fast path: Skip the heavy copy process if it's a single chunk upload
                os.rename(os.path.join(temp_dir, "chunk_0"), final_filepath)
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                with open(final_filepath, "wb") as outfile:
                    for i in range(total_chunks):
                        c_path = os.path.join(temp_dir, f"chunk_{i}")
                        with open(c_path, "rb") as infile:
                            shutil.copyfileobj(infile, outfile)
                # Cleanup temp chunks
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as exc:
            return jsonify({"error": f"Failed assembling chunks: {exc}"}), 500

        # Start the analysis pipeline seamlessly
        reset_progress()
        for folder in OUTPUT_FOLDERS:
            folder_path = os.path.join(BASE_DIR, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)

        with jobs_lock:
            jobs[job_id] = {
                "progress": 0,
                "status": "Initializing Intelligence...",
                "done": False,
                "result": None,
            }

        thread = threading.Thread(target=run_ai_task, args=(job_id, final_filepath), daemon=True)
        thread.start()

        return jsonify({
            "message": "Processing started",
            "job_id": job_id,
            "complete": True
        }), 202

    return jsonify({"message": f"Chunk {chunk_index} received", "complete": False}), 200



@app.route("/progress/<job_id>", methods=["GET"])
def handle_progress(job_id):
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404
            
        if not jobs[job_id]["done"]:
            jobs[job_id]["progress"] = progress_data.get("percent", jobs[job_id]["progress"])
            jobs[job_id]["status"] = progress_data.get("status", jobs[job_id]["status"])
            
        job = jobs[job_id]

        return jsonify({
            "percent": job["progress"],
            "status": job["status"],
            "done": job["done"],
            "categories": job["result"]["categories"] if job["result"] else {},
            "top": job["result"]["top"] if job["result"] else [],
        })


if __name__ == "__main__":
    print("Starting Heavy-Duty Premium Server...")
    serve(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        threads=12,
        max_request_body_size=10737418240,
        channel_timeout=1800,
        send_bytes=18000,
    )
