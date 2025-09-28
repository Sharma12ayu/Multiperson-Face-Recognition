from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Load known encodings
with open("encodings/known_faces.pkl", "rb") as f:
    data = pickle.load(f)

# Ensure logs and uploads folders exist
os.makedirs("logs", exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

log_path = "logs/recognized_log.csv"
if not os.path.exists(log_path):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(log_path, index=False)

def log_recognition(name):
    now = datetime.now()
    entry = pd.DataFrame([[name, now.date(), now.time().strftime("%H:%M:%S")]],
                         columns=["Name", "Date", "Time"])
    entry.to_csv(log_path, mode='a', index=False, header=False)

# Webcam video stream
def generate_frames():
    cap = cv2.VideoCapture(0)
    seen = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), encoding in zip(boxes, encodings):
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            if face_distances.any():
                best = face_distances.argmin()
                if matches[best]:
                    name = data["names"][best]

            if name not in seen:
                log_recognition(name)
                seen.add(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    df = pd.read_csv(log_path)
    return render_template("log.html", data=df.to_dict(orient="records"))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            image = cv2.imread(filepath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for (top, right, bottom, left), encoding in zip(boxes, encodings):
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(data["encodings"], encoding)
                if face_distances.any():
                    best = face_distances.argmin()
                    if matches[best]:
                        name = data["names"][best]

                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
            cv2.imwrite(result_path, image)
            return redirect(url_for("uploaded_image"))

    return render_template("upload.html")

@app.route('/uploaded')
def uploaded_image():
    return send_from_directory(app.config["UPLOAD_FOLDER"], "result.jpg")

if __name__ == "__main__":
    app.run(debug=True)
