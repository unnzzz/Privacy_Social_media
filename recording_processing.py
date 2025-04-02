import cv2
import numpy as np
import face_recognition
from models import Consent  # Ensure your models.py exports Consent

def get_friend_id_for_face(face_encoding, recording_owner):
    """
    Given a face encoding and the recording owner's friends, returns the friend's id if a match is found,
    otherwise returns None.
    """
    threshold = 0.6  # Adjust threshold as needed
    for friend in recording_owner.friends:
        if friend.face_encoding:
            distances = face_recognition.face_distance(friend.face_encoding, face_encoding)
            if distances.size > 0 and min(distances) < threshold:
                return friend.id
    return None

def process_recording_frame(frame, recording):
    """
    Process a single video frame from a recording. For each detected face, if it matches a friend's encoding
    and that friend did not consent (Consent.consent is False), blur that face.
    """
    # Detect faces in the frame.
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return frame

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # Build a dictionary of consent: friend_id -> consent value.
    consent_map = {consent.friend_id: consent.consent for consent in recording.consents}
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        friend_id = get_friend_id_for_face(face_encoding, recording.user)
        if friend_id is not None:
            # If the friend exists in our consent map and has not consented, blur the face.
            if consent_map.get(friend_id) is False:
                face_region = frame[top:bottom, left:right]
                blurred_face = cv2.GaussianBlur(face_region, (75, 75), 30)
                frame[top:bottom, left:right] = blurred_face
    return frame

def gen_recording_frames(recording):
    """
    Generator that reads a video file (recording.filename), processes each frame according to consent rules,
    and yields the frame as a JPEG stream.
    """
    # Adjust this if needed; ensure the filename is a full path.
    cap = cv2.VideoCapture(recording.filename)
    if not cap.isOpened():
        print("Error: Unable to open recording file.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_recording_frame(frame, recording)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
