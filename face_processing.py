import cv2
import numpy as np
import face_recognition
import os, glob, time, math, threading, queue
from facenet_pytorch import MTCNN
import torch
import dlib

# --- Global variable for temporal smoothing (history of face detections) ---
face_histories = []

# --- Helper: Euclidean Distance ---
def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# --- (Optional) Video Recorder Thread ---
class VideoRecorder(threading.Thread):
    def __init__(self, filename, codec, fps, resolution, max_queue_size=100):
        super(VideoRecorder, self).__init__()
        self.filename = filename
        self.fps = fps
        self.resolution = resolution
        self.writer = cv2.VideoWriter(filename, codec, fps, resolution)
        self.queue = queue.Queue(max_queue_size)
        self.stopped = False

    def run(self):
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1)
                self.writer.write(frame)
            except queue.Empty:
                continue
        self.writer.release()

    def write(self, frame):
        try:
            self.queue.put(frame, timeout=1)
        except queue.Full:
            pass

    def stop(self):
        self.stopped = True

# --- Load Known Face Encodings ---
def load_known_encodings():
    known_encodings = []
    folder = "known_faces"
    if os.path.exists(folder):
        image_paths = glob.glob(os.path.join(folder, "*.jpg"))
        for path in image_paths:
            print("Loading known image:", path)
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
            else:
                print("No face found in", path)
    else:
        if os.path.exists("known_face.jpg"):
            print("Loading known image: known_face.jpg")
            img = face_recognition.load_image_file("known_face.jpg")
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
            else:
                print("No face found in known_face.jpg")
        else:
            print("No reference images available. Exiting.")
            exit(1)
    print(f"Loaded {len(known_encodings)} known face encoding(s).")
    return known_encodings

# Load known encodings once when the module is imported.
KNOWN_ENCODINGS = load_known_encodings()

# --- Recognition Parameters ---
KNOWN_THRESHOLD_LOW = 0.5    # Below this average distance, classify as "Known"
KNOWN_THRESHOLD_HIGH = 0.6  # Above this, classify as "Blurred"
HISTORY_DURATION = 2.0       # Keep history for 2 seconds
MATCH_DIST_THRESHOLD = 50    # If detection centers are within 50 pixels, consider them the same face
HISTORY_LENGTH = 20          # Maximum number of frames in history

# --- (Optional) Face Alignment ---
align_enabled = False  # Set True if you have shape_predictor_68_face_landmarks.dat
if align_enabled:
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if os.path.exists(predictor_path):
        predictor = dlib.shape_predictor(predictor_path)
    else:
        print("shape_predictor_68_face_landmarks.dat not found; alignment disabled.")
        align_enabled = False

# --- Initialize MTCNN for Detection ---
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)

# --- Function to Process a Single Frame with Temporal Smoothing ---
def process_frame(frame, known_encodings=KNOWN_ENCODINGS, scale_factor=0.5):
    global face_histories
    current_time = time.time()
    frame_height, frame_width = frame.shape[:2]
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    boxes, probs = mtcnn.detect(rgb_small)
    current_detections = []
    if boxes is not None:
        print("Detected faces:", len(boxes))
        for i, box in enumerate(boxes):
            if probs[i] is None or probs[i] < 0.90:
                continue
            sx1, sy1, sx2, sy2 = map(int, box)
            x1 = int(sx1 / scale_factor)
            y1 = int(sy1 / scale_factor)
            x2 = int(sx2 / scale_factor)
            y2 = int(sy2 / scale_factor)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width, x2)
            y2 = min(frame_height, y2)
            margin = 60  # Increase margin if needed.
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame_width, x2 + margin)
            y2 = min(frame_height, y2 + margin)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            face_region = frame[y1:y2, x1:x2]
            print("Face region:", x1, y1, x2, y2, "Size:", face_region.shape)
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Compute face locations explicitly.
            face_locations = face_recognition.face_locations(face_rgb)
            print("Face locations:", face_locations)
            if face_locations:
                encodings = face_recognition.face_encodings(face_rgb, known_face_locations=face_locations)
                if encodings:
                    distances = face_recognition.face_distance(known_encodings, encodings[0])
                    if distances.size > 0:
                        min_distance = np.min(distances)
                    else:
                        min_distance = 1.0
                    print("Computed distance:", min_distance)
                else:
                    min_distance = 1.0
                    print("No encoding computed for this face.")
            else:
                encodings = []
                min_distance = 1.0
                print("No face locations found in cropped region.")
            current_detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "distance": min_distance
            })
    else:
        print("No faces detected by MTCNN")
    
    # Update histories for smoothing.
    for det in current_detections:
        center = det["center"]
        matched = False
        for hist in face_histories:
            if euclidean_distance(center, hist["center"]) < MATCH_DIST_THRESHOLD:
                hist["distances"].append(det["distance"])
                if len(hist["distances"]) > HISTORY_LENGTH:
                    hist["distances"].pop(0)
                hist["center"] = ((hist["center"][0] + center[0]) // 2, (hist["center"][1] + center[1]) // 2)
                hist["bbox"] = det["bbox"]
                hist["last_seen"] = current_time
                matched = True
                break
        if not matched:
            face_histories.append({
                "center": center,
                "distances": [det["distance"]],
                "bbox": det["bbox"],
                "last_seen": current_time,
                "stable_label": None
            })
    face_histories = [h for h in face_histories if current_time - h["last_seen"] < HISTORY_DURATION]
    
    # Update stable label using hysteresis.
    for hist in face_histories:
        avg_distance = np.mean(hist["distances"])
        midpoint = (KNOWN_THRESHOLD_LOW + KNOWN_THRESHOLD_HIGH) / 2
        if hist["stable_label"] is None:
            hist["stable_label"] = "Known" if avg_distance < midpoint else "Blurred"
        else:
            if hist["stable_label"] == "Known" and avg_distance > KNOWN_THRESHOLD_HIGH:
                hist["stable_label"] = "Blurred"
            elif hist["stable_label"] == "Blurred" and avg_distance < KNOWN_THRESHOLD_LOW:
                hist["stable_label"] = "Known"
        hist["avg_distance"] = avg_distance

    # Draw results on frame.
    for hist in face_histories:
        label = hist["stable_label"]
        distance_display = f"{hist['avg_distance']:.2f}"
        x1, y1, x2, y2 = hist["bbox"]
        color = (0, 255, 0) if label == "Known" else (0, 0, 255)
        if label == "Blurred":
            face_region = frame[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_region, (75, 75), 30)
            frame[y1:y2, x1:x2] = blurred_face
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({distance_display})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame
import cv2
import face_recognition

def detect_friends_in_recording(recording_path, friend_encodings, frame_skip=30, tolerance=0.6):
    """
    Scans a video file, sampling every `frame_skip` frames,
    runs face recognition, and returns a list of friend IDs
    whose face encoding was detected in at least one frame.
    
    - recording_path: full path to the video file (e.g., MP4)
    - friend_encodings: a dictionary mapping user IDs to their face encoding(s).
      This may be a single encoding or a list of encodings.
    - tolerance: maximum face_distance to consider a match.
    """
    cap = cv2.VideoCapture(recording_path)
    if not cap.isOpened():
        print(f"Error opening video file: {recording_path}")
        return []
    
    detected = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect all faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
            for enc in encodings:
                for user_id, f_enc in friend_encodings.items():
                    # Check if the stored encoding is a list (multiple images)
                    if isinstance(f_enc, list):
                        distances = face_recognition.face_distance(f_enc, enc)
                        if any(d <= tolerance for d in distances):
                            detected.add(user_id)
                    else:
                        # f_enc is a single encoding
                        distance = face_recognition.face_distance([f_enc], enc)[0]
                        if distance <= tolerance:
                            detected.add(user_id)
        frame_idx += 1

    cap.release()
    return list(detected)

# --- Optional: Standalone Testing ---
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        cv2.imshow("Processed", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
