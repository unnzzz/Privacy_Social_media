from flask import Flask, render_template, redirect, url_for, request, flash, Response
from config import Config
from models import db, User, FriendRequest, Recording, Consent
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import time
import subprocess



def convert_video(input_path, output_path):
    """
    Converts a video file at input_path into an HTML5-friendly MP4 using H.264 (Baseline profile) and
    moves the moov atom to the beginning for progressive download.
    """
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-movflags', 'faststart',
        output_path
    ]
    subprocess.run(command, check=True)

# Import face processing function, known encodings, and VideoRecorder from your module.
from face_processing import process_frame, KNOWN_ENCODINGS, VideoRecorder

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

from flask_migrate import Migrate
migrate = Migrate(app, db)

login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized_callback():
    return redirect(url_for('login'))

first_request = True
@app.before_request
def initialize_once():
    global first_request
    if first_request:
        db.create_all()
        first_request = False
from datetime import datetime
import pytz

@app.template_filter('local_time')
def local_time_filter(utc_dt):
    # Convert a UTC datetime to local time. Adjust the timezone as needed.
    local_tz = pytz.timezone("America/New_York")  # Change to your local timezone.
    return utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S')

# --- Signup Route ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        account_type = request.form.get('account_type')  # "public" or "private"
        
        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please log in or use a different email.")
            return redirect(url_for('signup'))
        
        # Get images from both upload and capture options.
        upload_files = request.files.getlist('upload_images')
        captured_files = request.files.getlist('captured_images')
        files = upload_files + captured_files
        
        if not files or len(files) == 0:
            flash("Please provide at least one face image (upload or capture).")
            return redirect(url_for('signup'))
        
        # Create a user-specific folder.
        upload_folder = os.path.join(app.root_path, 'known_faces', email)
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        all_encodings = []
        saved_filenames = []
        import face_recognition
        from werkzeug.utils import secure_filename
        
        for file in files:
            if file.filename == "":
                continue
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                all_encodings.append(encodings[0])
                saved_filenames.append(filename)
            else:
                flash(f"No face detected in {filename}. Skipping that image.")
        
        if not all_encodings:
            flash("No faces detected in any provided image. Please try again.")
            return redirect(url_for('signup'))
        
        is_public = True if account_type == "public" else False
        
        new_user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            reference_media=os.path.join('known_faces', email),
            face_encoding=all_encodings,
            is_public_account=is_public
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Account created. Please log in.")
        return redirect(url_for('login'))
    return render_template('signup.html')





# --- Login Route ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash("Invalid credentials.")
    return render_template('login.html')

# --- Logout Route ---
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Dashboard Route ---
@app.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

# --- Recordings Page ---
@app.route('/recordings')
@login_required
def recordings():
    # Query recordings for the current user.
    user_recordings = Recording.query.filter_by(user_id=current_user.id).all()
    return render_template('recordings.html', recordings=user_recordings)

# --- Global Variables for Recording ---
recording = False
recorder = None

# --- Start Recording Route ---
@app.route('/start_recording')
@login_required
def start_recording():
    global recording, recorder
    if not recording:
        recordings_folder = os.path.join(app.root_path, 'static', 'recordings')
        if not os.path.exists(recordings_folder):
            os.makedirs(recordings_folder)
        # Use .mp4 extension for an MP4 file.
        filename = f"recording_{current_user.id}_{int(time.time())}.mp4"
        filepath = os.path.join(recordings_folder, filename)
        cap_temp = cv2.VideoCapture(0)
        frame_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
        frame_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
        fps = cap_temp.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        cap_temp.release()
        # Use the 'mp4v' codec, which typically produces MP4 files playable in browsers.
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        from face_processing import VideoRecorder
        recorder = VideoRecorder(filepath, codec, fps, (frame_width, frame_height))
        recorder.start()
        recording = True
        print(f"Started recording, file will be saved to: {filepath}")
        flash("Recording started.")
    return redirect(url_for('live'))



# --- Stop Recording Route ---
@app.route('/stop_recording')
@login_required
def stop_recording():
    global recording, recorder
    if recording and recorder:
        recorder.stop()
        recorder.join()
        recordings_folder = os.path.join(app.root_path, 'static', 'recordings')
        filename = os.path.basename(recorder.filename)
        new_recording = Recording(user_id=current_user.id, filename=filename, is_shared=False)
        db.session.add(new_recording)
        db.session.commit()

        # Use the helper to detect which friends appear in the recording.
        friend_ids_in_recording = get_friends_in_recording(recorder.filename, current_user)
        print("Detected friend IDs in recording:", friend_ids_in_recording)
        for friend_id in friend_ids_in_recording:
            consent_entry = Consent(recording_id=new_recording.id, friend_id=friend_id, consent=None)
            db.session.add(consent_entry)
        db.session.commit()

        recorder = None
        recording = False
        flash("Recording stopped. Awaiting friend consents for unblurring their faces.")
    return redirect(url_for('live'))


# --- Generator for Video Feed ---
def gen_frames(user_face_encodings):
    cap_live = cv2.VideoCapture(0)
    if not cap_live.isOpened():
        print("Error: Unable to open camera")
        return
    frame_width = int(cap_live.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_live.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = cap_live.read()
        if not ret:
            break
        processed_frame = process_frame(frame, user_face_encodings, scale_factor=0.5)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        global recording, recorder
        if recording and recorder:
            rec_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
            cv2.putText(rec_frame, "REC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            recorder.write(rec_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap_live.release()

@app.route('/video_feed')
@login_required
def video_feed():
    # Build the known encodings list from current user and their friends.
    known_encodings = current_user.get_all_known_encodings()
    if not known_encodings:
        known_encodings = KNOWN_ENCODINGS  # fallback
    return Response(gen_frames(known_encodings), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
@login_required
def live():
    return render_template('live.html')



# --- Send Friend Request ---
@app.route('/send_friend_request', methods=['GET', 'POST'])
@login_required
def send_friend_request():
    if request.method == 'POST':
        # For simplicity, we'll ask for the friend's email.
        friend_email = request.form.get('friend_email')
        friend = User.query.filter_by(email=friend_email).first()
        if not friend:
            flash("No user found with that email.")
            return redirect(url_for('send_friend_request'))
        # Check if already friends.
        if friend in current_user.friends:
            flash("You are already friends with that user.")
            return redirect(url_for('dashboard'))
        # Check if a pending request already exists.
        existing_request = FriendRequest.query.filter_by(sender_id=current_user.id, receiver_id=friend.id, status='pending').first()
        if existing_request:
            flash("Friend request already sent.")
            return redirect(url_for('dashboard'))
        # Create a new friend request.
        new_request = FriendRequest(sender_id=current_user.id, receiver_id=friend.id)
        db.session.add(new_request)
        db.session.commit()
        flash("Friend request sent.")
        return redirect(url_for('dashboard'))
    return render_template('send_friend_request.html')


# --- View Incoming Friend Requests ---
@app.route('/friend_requests')
@login_required
def friend_requests():
    requests = FriendRequest.query.filter_by(receiver_id=current_user.id, status='pending').all()
    return render_template('friend_requests.html', requests=requests)



# --- Accept Friend Request ---
@app.route('/accept_friend_request/<int:request_id>')
@login_required
def accept_friend_request(request_id):
    friend_request = FriendRequest.query.get(request_id)
    if friend_request and friend_request.receiver_id == current_user.id:
        friend_request.status = 'accepted'
        sender = User.query.get(friend_request.sender_id)
        # Add each other as friends mutually if not already added.
        if sender not in current_user.friends:
            current_user.friends.append(sender)
        if current_user not in sender.friends:
            sender.friends.append(current_user)
        db.session.commit()
        flash("Friend request accepted.")
    else:
        flash("Friend request not found or unauthorized.")
    return redirect(url_for('friend_requests'))



# --- Reject Friend Request ---
@app.route('/reject_friend_request/<int:request_id>')
@login_required
def reject_friend_request(request_id):
    friend_request = FriendRequest.query.get(request_id)
    if friend_request and friend_request.receiver_id == current_user.id:
        friend_request.status = 'rejected'
        db.session.commit()
        flash("Friend request rejected.")
    else:
        flash("Friend request not found or invalid.")
    return redirect(url_for('friend_requests'))

@app.route('/friends')
@login_required
def friends():
    # current_user.friends comes from the many-to-many relationship in your User model.
    return render_template('friends.html', friends=current_user.friends)


from flask import send_from_directory

@app.route('/user_faces/<path:filename>')
@login_required
def user_faces(filename):
    # Build the full path to the user's folder.
    user_folder = os.path.join(app.root_path, 'known_faces', current_user.email)
    return send_from_directory(user_folder, filename)

@app.route('/my_faces')
@login_required
def my_faces():
    user_folder = os.path.join(app.root_path, 'known_faces', current_user.email)
    if os.path.exists(user_folder):
        photos = [f for f in os.listdir(user_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        photos = []
    return render_template('my_faces.html', photos=photos)

@app.route('/share_recording/<int:recording_id>')
@login_required
def share_recording(recording_id):
    rec = Recording.query.get(recording_id)
    if rec and rec.user_id == current_user.id:
        rec.is_shared = True
        db.session.commit()
        print(f"Recording {recording_id} marked as shared.")
        flash("Recording shared to feed.")
    else:
        flash("Recording not found or unauthorized.")
    return redirect(url_for('recordings'))




from models import Recording
from recording_processing import gen_recording_frames  # Import the generator

@app.route('/play_recording/<int:recording_id>')
@login_required
def play_recording(recording_id):
    recording = Recording.query.get(recording_id)
    if not recording:
        flash("Recording not found.")
        return redirect(url_for('feed'))
    # Optionally add access checks (e.g., if current_user is allowed to view this recording)
    return Response(gen_recording_frames(recording),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/feed')
@login_required
def feed():
    # Get all recordings from public accounts that are shared.
    public_recordings = Recording.query.join(User).filter(
        User.is_public_account == True,
        Recording.is_shared == True
    ).all()
    
    # For private accounts, include shared recordings if the owner is a friend or the current user.
    friend_ids = [friend.id for friend in current_user.friends]
    private_recordings = Recording.query.join(User).filter(
        User.is_public_account == False,
        Recording.is_shared == True,
        ((Recording.user_id.in_(friend_ids)) | (Recording.user_id == current_user.id))
    ).all()
    
    # Combine the recordings and sort them by timestamp (newest first).
    all_recordings = public_recordings + private_recordings
    all_recordings.sort(key=lambda r: r.timestamp, reverse=True)
    
    print("Public recordings:", public_recordings)
    print("Private recordings:", private_recordings)
    print("All recordings:", all_recordings)
    
    return render_template('feed.html', recordings=all_recordings)

import face_recognition

def get_friends_in_recording(file_path, current_user, frame_interval=10, threshold=0.6):
    """
    Processes the video at file_path and returns a list of friend IDs (from current_user.friends)
    whose face appears in the video. The function processes one out of every 'frame_interval' frames.
    
    :param file_path: Full path to the video file.
    :param current_user: The currently logged-in user. Assumes current_user.friends is populated and 
                         each friend has a 'face_encoding' attribute (a list of encodings).
    :param frame_interval: Process one frame every 'frame_interval' frames.
    :param threshold: Distance threshold for face matching.
    :return: List of friend IDs detected in the video.
    """
    detected_friend_ids = set()
    
    # Build a dictionary mapping friend_id to their face encodings.
    friend_encodings = {}
    for friend in current_user.friends:
        if friend.face_encoding:
            friend_encodings[friend.id] = friend.face_encoding
    if not friend_encodings:
        return list(detected_friend_ids)
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Unable to open video file:", file_path)
        return list(detected_friend_ids)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process one frame every 'frame_interval' frames.
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect face locations.
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            if face_locations:
                # Get encodings for all detected faces.
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                for encoding in face_encodings:
                    # Compare this face with each friend's stored encodings.
                    for friend_id, encodings in friend_encodings.items():
                        distances = face_recognition.face_distance(encodings, encoding)
                        if distances.size > 0 and min(distances) < threshold:
                            detected_friend_ids.add(friend_id)
                            # Once we detect this friend in one frame, we can break out.
                            break
        frame_count += 1
    cap.release()
    return list(detected_friend_ids)
@app.route('/consent/<int:recording_id>/<int:friend_id>', methods=['GET', 'POST'])
@login_required
def consent(recording_id, friend_id):
    # Only allow the friend (the intended recipient) to access this page.
    if current_user.id != friend_id:
        flash("Unauthorized access.")
        return redirect(url_for('dashboard'))
    
    # Retrieve the consent entry from the database.
    consent_entry = Consent.query.filter_by(recording_id=recording_id, friend_id=friend_id).first()
    if not consent_entry:
        flash("No consent request found.")
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        decision = request.form.get('decision')
        if decision == 'accept':
            consent_entry.consent = True
        elif decision == 'reject':
            consent_entry.consent = False
        db.session.commit()
        flash("Your consent decision has been recorded.")
        return redirect(url_for('dashboard'))
    
    return render_template('consent.html', recording_id=recording_id, friend_id=friend_id)

@app.route('/pending_consents')
@login_required
def pending_consents():
    # Query for Consent entries where current_user is the friend and consent is None (pending)
    pending = Consent.query.filter_by(friend_id=current_user.id, consent=None).all()
    return render_template('pending_consents.html', consents=pending)




if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
