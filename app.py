from flask import Flask, render_template, redirect, url_for, request, flash, Response, abort, session, g
from config import Config
from pathlib import Path
from models import db, User, FriendRequest, Recording , ConsentRequest
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import time
import subprocess

# --- active recordings -------------------------------------------------
# key = user_id   •   value = { "recorder": VideoRecorder , "raw_fp": path }
ACTIVE_RECS = {}
recording = False
recorder = None

def convert_video(input_path, output_path):
    cmd = [
        "ffmpeg", "-y",                # ‑y = overwrite
        "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)



# Import face processing function, known encodings, and VideoRecorder from your module.
from face_processing import process_frame, KNOWN_ENCODINGS, VideoRecorder, _graceful_recorder_close, FPS_FOR_PIPE, PIPE_WIDTH, PIPE_HEIGHT

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
# ----------------------------------------------------------------------
# 1)  put this helper near the top of the file (after the imports)
# ----------------------------------------------------------------------
def get_live_dimensions(source):
    """Return (width, height, fps) depending on which camera is active."""
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        ok, w, h, fps = (
            cap.isOpened(),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS) or 30,
        )
        cap.release()
        if not ok:
            raise RuntimeError("Web‑cam not available.")
        return w, h, fps
    else:                                # hololens → 640 × 360 @ 30 FPS
        return 640, 360, 30

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
    pending_count = ConsentRequest.query.filter_by(
        recipient_id=current_user.id, status='pending'
    ).count()
    return render_template('dashboard.html',
                           user=current_user,
                           pending_consents=pending_count)


# --- Recordings Page ---
@app.route('/recordings')
@login_required
def recordings():
    # Query Recording model, not list of strings
    recs = Recording.query.filter_by(user_id=current_user.id)\
                         .order_by(Recording.timestamp.desc())\
                         .all()
    return render_template('recordings.html', recordings=recs)



# --- Global Variables for Recording ---


# --- Start Recording Route ---
# ----------------------------------------------------------------------
# 2)  START‑RECORDING ROUTE – replace the body with this version
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
#  Start a new recording for the logged‑in user
# ----------------------------------------------------------------------
# --- Start Recording Route ---
# --- Start Recording Route ---
@app.route('/start_recording')
@login_required
def start_recording():
    global recording, recorder
    if not recording:
        # ensure output folder exists
        recordings_folder = os.path.join(app.root_path, 'static', 'recordings')
        os.makedirs(recordings_folder, exist_ok=True)

        # use AVI + MJPG so ffmpeg can read it later
        filename = f"recording_{current_user.id}_{int(time.time())}.avi"
        filepath = os.path.join(recordings_folder, filename)

        # probe webcam for dims/fps
        cap_temp = cv2.VideoCapture(0)
        W = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))  // 2
        H = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
        FPS = cap_temp.get(cv2.CAP_PROP_FPS) or 30
        cap_temp.release()

        # MJPG in AVI container
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        recorder = VideoRecorder(filepath, codec, FPS, (W, H))
        recorder.start()

        recording = True
        flash("Recording started.")
    else:
        flash("Already recording!")
    return redirect(url_for('live'))


# --- Stop Recording Route ---
@app.route('/stop_recording')
@login_required
def stop_recording():
    global recording, recorder
    if not recording or recorder is None:
        flash("No recording in progress.")
        return redirect(url_for('live'))

    # stop the background thread
    recorder.stop()
    recorder.join()

    # raw AVI on disk
    raw_fp = recorder.filename

    # convert to H.264‑MP4
    base, _    = os.path.splitext(raw_fp)
    out_fp     = f"{base}_converted.mp4"
    try:
        convert_video(raw_fp, out_fp)
        os.remove(raw_fp)

        # save to DB
        new_rec = Recording(
            user_id=current_user.id,
            filename=os.path.basename(out_fp),
            is_shared=False
        )
        db.session.add(new_rec)
        db.session.commit()
        flash("Recording saved!")
    except Exception as e:
        print("Conversion failed:", e)
        flash("Recording stopped, conversion failed.")

    # reset globals
    recorder  = None
    recording = False

    return redirect(url_for('recordings'))







# --- Generator for Video Feed ---
# def gen_frames(user_face_encodings):
#     HL_USER = "ayush"
#     HL_PWD = "password123"
#     HL_IP = "192.168.1.144"
#     hololens_stream=(f"http://{HL_USER}:{HL_PWD}@{HL_IP}/api/holographic/stream/live.mp4?olo=true&pv=true&mic=true&loopback=true")
#     cap_live = cv2.VideoCapture(hololens_stream,cv2.CAP_FFMPEG)
#     if not cap_live.isOpened():
#         print("Error: Unable to open camera")
#         return
#     frame_width = int(cap_live.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap_live.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     while True:
#         ret, frame = cap_live.read()
#         if not ret:
#             break
#         processed_frame = process_frame(frame, user_face_encodings, scale_factor=0.5)
#         ret, buffer = cv2.imencode('.jpg', processed_frame)
#         if not ret:
#             continue
#         global recording, recorder
#         if recording and recorder:
#             rec_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
#             cv2.putText(rec_frame, "REC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             recorder.write(rec_frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#     cap_live.release()
import subprocess, io, struct, numpy as np

def open_holo_stream():
    HL_USER = "unnati"
    HL_PWD  = "unnati5"
    HL_IP   = "10.154.27.75"
    url = (f"http://{HL_USER}:{HL_PWD}@{HL_IP}"
           "/api/holographic/stream/live.mp4?pv=true&loopback=true")
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",  # or "fatal"
        "-i", url,
        "-vf", f"fps={FPS_FOR_PIPE},scale={PIPE_WIDTH}:{PIPE_HEIGHT}",   #  ← changed
        "-c:v", "mjpeg", "-qscale:v", "4",
        "-f", "image2pipe", "-"
]

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
# ─── top of file, *below* your HL_USER / HL_PWD / HL_IP constants ──────────────
VIDEO_SOURCE_DEFAULT = "webcam"               # fallback

def get_video_source():
    # Source is stored in session so one user’s choice
    # doesn’t affect another’s.
    return session.get("video_source", VIDEO_SOURCE_DEFAULT)

def set_video_source(src):
    session["video_source"] = src
# ───────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------
#   generator: gen_frames
#   ─ streams MJPEG from webcam ❶ or HoloLens ❷
# ---------------------------------------------------------------
def gen_frames(user_face_encodings, source):
    """
    Yield Motion‑JPEG frames for streaming, and write raw frames
    into recorder when recording==True.

    Parameters
    ----------
    user_face_encodings : list
        All encodings (current user + friends) to leave unblurred.
    source : str
        "webcam"  → use local camera index 0  
        "hololens"→ pull frames from the HoloLens Device‑Portal via open_holo_stream()
    """
    global recording, recorder

    # ────────────────────────────── ❶ webcam ──────────────────────────────
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌  Webcam not available")
            return

        def grab():
            ok, frm = cap.read()
            return frm if ok else None

    # ────────────────────────────── ❷ hololens ────────────────────────────
    else:
        ff = open_holo_stream()   # you must have defined open_holo_stream() above

        def grab():
            # read one JPEG from ffmpeg stdout
            buf = bytearray()
            while True:
                b = ff.stdout.read(1)
                if not b:
                    return None            # stream ended
                buf.append(b[0])
                if len(buf) > 1 and buf[-2:] == b"\xff\xd9":
                    break                  # end‑of‑JPEG
            return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

    # ─────────────────────────── main streaming loop ─────────────────────
    while True:
        frame = grab()
        if frame is None:
            break                          # EOF or error → exit

        # 1) face processing & blurring
        processed = process_frame(frame, user_face_encodings)

        # 2) record if requested (use the raw frame, not the blurred one)
        if recording and recorder:
            # downscale to recorder.resolution
            rec_frame = cv2.resize(frame, recorder.resolution)
            recorder.write(rec_frame)

        # 3) encode to JPEG for MJPEG stream
        ok, jpeg = cv2.imencode(".jpg", processed)
        if not ok:
            continue                      # skip bad frame

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )

    # ───────────────────────────── clean‑up ───────────────────────────────
    if source == "webcam":
        cap.release()
    else:
        ff.terminate()




@app.route('/video_feed')
@login_required
def video_feed():
    # decide once, *inside* the request‑context
    source = session.get("video_source", VIDEO_SOURCE_DEFAULT)  # "webcam" or "hololens"

    # build encodings, etc. (unchanged)
    known_encs = current_user.get_all_known_encodings() or KNOWN_ENCODINGS

    # pass the chosen source into the generator
    return Response(
        gen_frames(known_encs, source),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/set_source/<src>')
@login_required
def set_source(src):
    if src not in ("webcam", "hololens"):
        abort(400)
    set_video_source(src)
    flash(f"Video source switched to {src.title()}.")
    return redirect(url_for('live'))

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

from face_processing import detect_friends_in_recording

@app.route('/share_recording/<int:rec_id>')
@login_required
def share_recording(rec_id):
    rec = Recording.query.get_or_404(rec_id)
    if rec.user_id != current_user.id:
        abort(403)

    # 1) Who are your friends?
    friends = current_user.get_friends()
    print("🔍 Friends list:", [u.email for u in friends])

    # 2) Build the dict of encodings
    friend_encs = {u.id: u.face_encoding for u in friends if u.face_encoding is not None}
    print("🔍 Friend encodings keys:", list(friend_encs.keys()))

    # 3) Path to the video
    path = os.path.join(app.root_path, 'static', 'recordings', rec.filename)
    print("🔍 Recording path:", path)

    # 4) Run detection
    present_ids = detect_friends_in_recording(path, friend_encs)
    print("🔍 Detected friend IDs in video:", present_ids)

    # 5) Create ConsentRequest rows
    for fid in present_ids:
        exists = ConsentRequest.query.filter_by(recording_id=rec.id, recipient_id=fid).first()
        print(f"   - Friend {fid} already has request? {bool(exists)}")
        if not exists:
            cr = ConsentRequest(
                recording_id=rec.id,
                requester_id=current_user.id,
                recipient_id=fid
            )
            db.session.add(cr)

    # 6) If no friends found, auto‑share
    if not present_ids:
        print("⚠️  No friends detected, auto‑sharing")
        rec.consent_status = 'approved'
        rec.is_shared     = True

    db.session.commit()
    flash(f"Consent requested from {len(present_ids)} friend(s).")
    return redirect(url_for('consent_requests'))







@app.route('/consent/<int:cr_id>', methods=['GET', 'POST'])
@login_required
def consent(cr_id):
    cr = ConsentRequest.query.get_or_404(cr_id)
    # only the intended recipient may respond
    if cr.recipient_id != current_user.id:
        abort(403)

    if request.method == 'POST':
        choice = request.form['consent']  # 'approved' or 'denied'
        cr.status = choice
        db.session.commit()
        flash("Your choice has been recorded.")

        # Now check if ALL requests for this recording are approved
        rec = Recording.query.get(cr.recording_id)
        pending = ConsentRequest.query.filter_by(
            recording_id=rec.id, status='pending'
        ).count()
        denied  = ConsentRequest.query.filter_by(
            recording_id=rec.id, status='denied'
        ).count()

        if pending == 0 and denied == 0:
            # everyone approved!
            rec.is_shared = True
            rec.consent_status = 'approved'
            db.session.commit()
            flash("All consents received—recording is now shared!")

        return redirect(url_for('consent_requests'))


    # GET: show the form
    return render_template('consent.html', consent_request=cr)

@app.route('/consents')
@login_required
def consent_requests():
    pending = ConsentRequest.query.filter_by(
        recipient_id=current_user.id, status='pending'
    ).all()
    return render_template('consents_list.html', requests=pending)


@app.route('/respond_consent/<int:cr_id>/<decision>')
@login_required
def respond_consent(cr_id, decision):
    cr = ConsentRequest.query.get_or_404(cr_id)
    if cr.recipient_id != current_user.id:
        abort(403)
    if decision not in ('approve','deny'):
        abort(400)
    cr.status = 'approved' if decision=='approve' else 'denied'
    db.session.commit()

    # Now update the parent recording’s overall status:
    rec = cr.recording
    all_reqs = rec.consent_requests
    if any(r.status=='denied' for r in all_reqs):
        rec.consent_status = 'denied'
        rec.is_shared     = False
    elif all(r.status=='approved' for r in all_reqs):
        rec.consent_status = 'approved'
        rec.is_shared     = True
    # else still pending
    db.session.commit()

    flash("Your response has been recorded.")
    return redirect(url_for('consent_requests'))


@app.route('/feed')
@login_required
def feed():
    # 1) Public recordings from public users
    public_recs = (
        Recording.query
        .join(User, Recording.user_id == User.id)
        .filter(Recording.is_shared == True, User.is_public_account == True)
        .order_by(Recording.timestamp.desc())
        .all()
    )

    # 2) Private recordings from your friends
    friend_ids = [u.id for u in current_user.friends]
    private_recs = (
        Recording.query
        .join(User, Recording.user_id == User.id)
        .filter(
            Recording.is_shared == True,
            Recording.user_id.in_(friend_ids),
            User.is_public_account == False
        )
        .order_by(Recording.timestamp.desc())
        .all()
    )

    # Debug prints
    print("🔍 Public in feed:", [r.id for r in public_recs])
    print("🔍 Private in feed:", [r.id for r in private_recs])

    return render_template(
        'feed.html',
        public_recordings=public_recs,
        private_recordings=private_recs
    )







if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
