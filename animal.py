import signal
import cv2
import torch
import os
import time
import pygame
import threading
from flask import Flask, render_template, Response, request, redirect, url_for
import sendSMS
from flask import jsonify


app = Flask(__name__)

# Initialize pygame mixer for sound alerts
pygame.mixer.init()

# Load YOLOv5 model
model_path = 'best.pt'  # Replace with your trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

elephant_detected_timeout = 2  # Time in seconds to stop sound after last detection
last_detection_time = 0
sound_playing = False  # Track sound state

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def play_sound(sound_file):
    """Play an alert sound if an elephant is detected."""
    global sound_playing
    if not sound_playing:
        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play(-1)  # Loop sound
            sound_playing = True
        except Exception as e:
            print(f"Error playing sound: {e}")


def stop_sound():
    """Stop the currently playing sound."""
    global sound_playing
    if sound_playing:
        pygame.mixer.music.stop()
        sound_playing = False


def detect_objects(frame):
    """Process a frame through YOLOv5 and return detections."""
    results = model(frame)
    return results


def generate_frames(video_path):
    global last_detection_time, sound_playing

    cap = cv2.VideoCapture(video_path)
    sms_sent = False
    confidence_threshold = 0.60
    sound_file = "sound.mp3"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects(frame)
        elephant_detected = False
        detected_labels = []  # Store detected object labels

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf < confidence_threshold:
                continue

            label_name = model.names[int(cls)]
            if label_name == "None":  # Check if class name is 'None'
                label_name = "Tiger"  # Replace it with 'Tiger'

            label = f'{label_name} {conf:.2f}'
            detected_labels.append(label_name)

            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            data =model.names[int(cls)]
            print(data)
            if model.names[int(cls)] == "elephant" or model.names[int(cls)] == "Bison" or model.names[int(cls)] == "Wild Boar" or model.names[int(cls)] == "Deer" or model.names[int(cls)] == "None":
                elephant_detected = True
                last_detection_time = time.time()

                if not sms_sent:
                    try:
                        sendSMS.send("Animal Detected")
                        print("SMS sent: Animal Detected")
                        sms_sent = True
                    except Exception as e:
                        print(f"Error sending SMS: {e}")

                if not sound_playing:
                    play_sound(sound_file)

        if not elephant_detected and sound_playing:
            if time.time() - last_detection_time > elephant_detected_timeout:
                stop_sound()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    stop_sound()

def generate_webcam_frames():
    """Real-time object detection using webcam."""
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects(frame)
        annotated_frame = results.render()[0]

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the upload page and handle video uploads."""
    if request.method == 'POST':
        video = request.files['video']
        if video:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(video_path)
            return redirect(url_for('video_feed', path=video.filename))
    return render_template('index.html')


@app.route('/video_feed/<path:path>')
def video_feed(path):
    """Stream the processed video with detections."""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam_feed')
def webcam_feed():
    """Stream real-time webcam feed with YOLOv5 detections."""
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def handle_sigint(signal_received, frame):
    """Handle SIGINT (Ctrl+C) to clean up resources."""
    print("\nSIGINT received. Cleaning up...")
    pygame.mixer.quit()
    exit(0)

@app.route('/get_labels', methods=['GET'])
def get_labels():
    """Return detected object labels as JSON."""
    return jsonify({'labels': detected_labels})


# Register the SIGINT handler
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == '__main__':
    app.run(debug=True)
