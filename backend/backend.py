import cv2
import numpy as np
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import base64
import re


progress_data = {
    "percent": 0,
    "status": "Idle"
}

# --- CONFIGURATION ---

BASE_OUTPUT_DIR = "Categorized_Thumbnails"
TOP_RECOMMENDED_DIR = "Top_Recommended"

FRAMES_PER_CATEGORY = 5
TOP_N = 10
RESIZE_WIDTH = 256

# --- AI ENHANCEMENT CONFIG ---
ENABLE_FACE_ENHANCEMENT = True  
CODEFORMER_FIDELITY = 0.5       # Lowered to 0.5 to force the AI to reconstruct heavily damaged faces more aggressively
UPSCALE_FACTOR = 2              
MAX_PARALLEL_API_CALLS = 5      

CATEGORIES = [
    "Close-Up Face Shot",
    "Landscape - Wide Shot",
    "Action - Motion-Based",
    "Object-Focused",
    "Sharp - Non-Blurry"
]

TOP_DISTRIBUTION = {
    "Close-Up Face Shot": 6,
    "Action - Motion-Based": 3,
    "Landscape - Wide Shot": 2
}

# --- AUTO-DOWNLOAD DEEP LEARNING FACE MODELS ---
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"⬇️ Downloading required AI model: {filename}...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

def setup_face_detector():
    download_file(PROTOTXT_URL, "deploy.prototxt")
    download_file(MODEL_URL, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

POWERS_OF_TWO = 1 << np.arange(64, dtype=object)

def get_dhash(gray_img):
    resized = cv2.resize(gray_img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = (resized[:, 1:] > resized[:, :-1]).flatten()
    return sum(POWERS_OF_TWO[diff])

def hamming_dist(h1, h2):
    try:
        return (h1 ^ h2).bit_count()
    except AttributeError:
        return bin(h1 ^ h2).count('1')

def fast_seek(cap, target_idx, current_idx):
    diff = int(target_idx - current_idx)
    if diff == 0:
        return
    if 0 < diff < 100: 
        for _ in range(diff):
            cap.grab()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

def enhance_face_image(image_path):
    # pyrefly: ignore [missing-import]
    import replicate
    filename = os.path.basename(image_path)
    print(f"   🪄 Restoring {filename} via AI...")
    
    try:
        output = replicate.run(
            "sczhou/codeformer:7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56",
            input={
                "image": open(image_path, "rb"),
                "upscale": UPSCALE_FACTOR,
                "face_upsample": True,
                "background_enhance": True,
                "codeformer_fidelity": CODEFORMER_FIDELITY
            }
        )
        
        response = requests.get(output)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✅ Pristine Face Saved: {filename}")
        else:
            print(f"   ❌ Download Failed for {filename}: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ API Error on {filename}: {e}")

def main(video_path):
    global progress_data
    os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
    start_time = time.time()
    
    print("🚀 STARTING THUMBNAIL ENGINE...\n")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TOP_RECOMMENDED_DIR, exist_ok=True)
    for cat in CATEGORIES:
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, cat), exist_ok=True)

    # Initialize Deep Learning Face Detector
    print("🧠 Loading OpenCV Deep Learning Face Detector...")
    face_net = setup_face_detector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_f = int(total_frames * 0.05)
    end_f = int(total_frames * 0.95)
    sample_indices = np.linspace(start_f, end_f, 150, dtype=int) 
    
    global_pool = []

    print("\n📊 Phase 1: Scanning video...\n")
    phase1_start = time.time()
    
    cap_ahead = cv2.VideoCapture(video_path)
    pos_cap = 0
    pos_ahead = 0
    center_mask = None

    for i, frame_idx in enumerate(sample_indices):
        fast_seek(cap, frame_idx, pos_cap)
        ret, frame = cap.read()
        pos_cap = frame_idx + 1
        if not ret: continue
            
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        brightness = gray_full.mean()
        if brightness < 20 or brightness > 240: continue
        
        sharpness = cv2.Laplacian(gray_full, cv2.CV_64F).var()
        if sharpness < 85: continue 
            
        target_ahead = frame_idx + int(fps / 2)
        fast_seek(cap_ahead, target_ahead, pos_ahead)
        ret2, frame_ahead = cap_ahead.read()
        pos_ahead = target_ahead + 1

        h, w = frame.shape[:2]
        new_h = int((RESIZE_WIDTH / w) * h)
        small = cv2.resize(frame, (RESIZE_WIDTH, new_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        motion_score = 0
        if ret2:
            small_ahead = cv2.resize(frame_ahead, (RESIZE_WIDTH, new_h))
            gray_ahead = cv2.cvtColor(small_ahead, cv2.COLOR_BGR2GRAY)
            motion_score = np.mean(cv2.absdiff(gray, gray_ahead))

        # --- NEW DEEP LEARNING FACE DETECTION ---
        blob = cv2.dnn.blobFromImage(small, 1.0, (300, 300), [104, 117, 123], False, False)
        face_net.setInput(blob)
        detections = face_net.forward()
        
        closeup_score = 0
        has_face = False
        
        # Loop over detections
        for j in range(detections.shape[2]):
            confidence = detections[0, 0, j, 2]
            # Filter out weak detections (must be > 60% confident it's a real human face)
            if confidence > 0.8:
                has_face = True
                box = detections[0, 0, j, 3:7] * np.array([RESIZE_WIDTH, new_h, RESIZE_WIDTH, new_h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Calculate how much of the screen the face takes up
                face_width = endX - startX
                face_height = endY - startY
                face_area_ratio = (face_width * face_height) / (RESIZE_WIDTH * new_h)
                
                if face_area_ratio * 1000 > closeup_score:
                    closeup_score = face_area_ratio * 1000 

        if has_face:
            object_score = 0
            landscape_score = 0
        else:
            edges = cv2.Canny(gray, 50, 150)
            if center_mask is None or center_mask.shape != gray.shape:
                center_mask = np.zeros_like(gray)
                cv2.rectangle(center_mask,
                              (int(RESIZE_WIDTH*0.25), int(new_h*0.25)),
                              (int(RESIZE_WIDTH*0.75), int(new_h*0.75)), 255, -1)

            center_edges = cv2.bitwise_and(edges, edges, mask=center_mask).mean()
            border_edges = edges.mean() - center_edges
            
            object_score = center_edges - border_edges
            landscape_score = sharpness + gray.std()

        scores = {
            "Close-Up Face Shot": closeup_score,
            "Landscape - Wide Shot": landscape_score,
            "Action - Motion-Based": motion_score * 10,
            "Object-Focused": object_score * 5,
            "Sharp - Non-Blurry": sharpness
        }

        global_pool.append({
            'frame': frame,
            'scores': scores,
            'hash': get_dhash(gray),
            'used': False
        })

        progress = ((i + 1) / len(sample_indices)) * 100
        progress_data["percent"] = int(progress)
        progress_data["status"] = "Scanning video frames..."
        print(f"\rProgress: {progress:.1f}%", end="")

    cap.release()
    cap_ahead.release()
    print(f"\n✅ Phase 1 Done. Found {len(global_pool)} candidate frames.")
    phase1_time = time.time() - phase1_start

    print("\n📂 Phase 2: Category selection...")
    phase2_start = time.time()
    results = {cat: [] for cat in CATEGORIES}

    for _ in range(FRAMES_PER_CATEGORY):
        for cat in CATEGORIES:
            best_sample = None
            best_score = -1
            for sample in global_pool:
                if sample['used']: continue
                if sample['scores'][cat] > best_score:
                    best_score = sample['scores'][cat]
                    best_sample = sample

            if best_sample and best_score > 5:
                best_sample['used'] = True
                results[cat].append(best_sample)

    phase2_time = time.time() - phase2_start

    print("\n🏆 Phase 3: Top 10 selection...")
    phase3_start = time.time()
    top_results = []

    def pick(category, limit):
        candidates = sorted(global_pool, key=lambda x: x['scores'][category], reverse=True)
        for sample in candidates:
            if len(top_results) >= limit: break
            if not any(hamming_dist(sample['hash'], ex['hash']) < 8 for ex in top_results):
                top_results.append(sample)

    for cat, count in TOP_DISTRIBUTION.items():
        pick(cat, len(top_results) + count)

    if len(top_results) < TOP_N:
        remaining = sorted(global_pool, key=lambda x: max(x['scores'].values()), reverse=True)
        for sample in remaining:
            if len(top_results) >= TOP_N: break
            if not any(hamming_dist(sample['hash'], ex['hash']) < 8 for ex in top_results):
                top_results.append(sample)

    phase3_time = time.time() - phase3_start

    print("\n💾 Phase 4: Saving raw images...")
    saved_face_paths = []

    for cat in CATEGORIES:
        for i, item in enumerate(results[cat]):
            file_path = os.path.join(BASE_OUTPUT_DIR, cat, f"{cat}_{i+1}.jpg")
            cv2.imwrite(file_path, item['frame'])
            if cat == "Close-Up Face Shot":
                saved_face_paths.append(file_path)

    for i, item in enumerate(top_results):
        file_path = os.path.join(TOP_RECOMMENDED_DIR, f"Top_{i+1}.jpg")
        cv2.imwrite(file_path, item['frame'])
        if item['scores']["Close-Up Face Shot"] > 50:
             saved_face_paths.append(file_path)

    phase5_time = 0
    if ENABLE_FACE_ENHANCEMENT:
        print("\n🤖 Phase 5: Parallel AI Face Restoration...")
        phase5_start = time.time()
        
        saved_face_paths = list(set(saved_face_paths)) 
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_API_CALLS) as executor:
            executor.map(enhance_face_image, saved_face_paths)
                
        phase5_time = time.time() - phase5_start

    print("\n✂️ Phase 6: Removing Black Borders from Final Outputs...")
    progress_data["status"] = "Cropping Letterboxes..."
    
    def crop_dir(directory):
        if not os.path.exists(directory): return
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, f)
                    img = cv2.imread(img_path)
                    if img is None: continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                    blur = cv2.medianBlur(thresh, 15)
                    coords = cv2.findNonZero(blur)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        img_h, img_w = img.shape[:2]
                        if w > 50 and h > 50 and (w < img_w - 10 or h < img_h - 10):
                            # Pad a few pixels to ensure we don't accidentally clip valid content 
                            # from median blur edge rounding
                            cx1 = max(0, x - 5)
                            cy1 = max(0, y - 5)
                            cx2 = min(img_w, x + w + 5)
                            cy2 = min(img_h, y + h + 5)
                            cv2.imwrite(img_path, img[cy1:cy2, cx1:cx2])

    crop_dir(BASE_OUTPUT_DIR)
    crop_dir(TOP_RECOMMENDED_DIR)

    total_time = time.time() - start_time
    print("\n==============================")
    print("⏱️ EXECUTION TIME REPORT")
    print("==============================")
    print(f"Phase 1 (Scanning): {phase1_time:.2f} sec")
    print("\n📂 Phase 2: Category selection...")
    progress_data["status"] = "Categorizing best shots..." # 👈 ADD THIS
    phase2_start = time.time()
    print(f"Phase 3 (Top Selection): {phase3_time:.2f} sec")
    if ENABLE_FACE_ENHANCEMENT:
        print("\n🤖 Phase 5: Parallel AI Face Restoration...")
        progress_data["status"] = "AI Enhancing Faces..." # 👈 ADD THIS
        phase5_start = time.time()
    print("------------------------------")
    print(f"TOTAL TIME: {total_time:.2f} sec")
    print("==============================")
    progress_data["percent"] = 100
    progress_data["status"] = "Complete"
    print("\n✅ DONE! Smart thumbnails generated and enhanced successfully 🚀")
    
    

if __name__ == "__main__":
    main()
    

# Helper function to sort files like "Top_2.jpg", "Top_10.jpg" correctly
def sort_files_numerically(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

def get_output():
    categories = {}
    top = []

    # Sort categories to match the frontend selection reliably
    for cat in CATEGORIES: 
        cat_path = os.path.join(BASE_OUTPUT_DIR, cat)
        if not os.path.exists(cat_path):
            continue

        images = []
        # 🔥 FIX: Sort the files numerically before converting to base64
        sorted_files = sorted(os.listdir(cat_path), key=sort_files_numerically)

        for file in sorted_files:
            path = os.path.join(cat_path, file)
            with open(path, "rb") as f:
                images.append(base64.b64encode(f.read()).decode())

        categories[cat] = images

    if os.path.exists(TOP_RECOMMENDED_DIR):
        # 🔥 FIX: Sort the top files numerically
        sorted_top = sorted(os.listdir(TOP_RECOMMENDED_DIR), key=sort_files_numerically)
        
        for file in sorted_top:
            path = os.path.join(TOP_RECOMMENDED_DIR, file)
            with open(path, "rb") as f:
                top.append(base64.b64encode(f.read()).decode())

    return categories, top