# 🎯 Multi-Person Face Recognition Web App

A real-time face recognition system built with **Flask**, **OpenCV**, and **face_recognition**.  
Detect and identify multiple people through live webcam or uploaded images.  
Includes automatic CSV logging, browser UI, and personalizable dataset setup.

---

## 🚀 Features

- 🎥 Real-time face recognition using webcam
- 🖼️ Upload an image to detect faces
- 🧠 Trained on your custom face dataset
- 📊 Auto-logs name, date, and time to CSV
- 🌐 Web interface using Flask (no command-line needed)
- 🛡️ Privacy-first — no private data included in this repo

---

## Sample detection 
<img src ="uploads\result.jpg" alt="prediction-example" width="500"/>
## 📁 Project Structure

```
MPFR/
├── app.py                # Main Flask app
├── create_encodings.py   # Generate known face encodings
├── dataset/              # Place folders of person's images here
├── encodings/            # Encoded .pkl file saved here
├── uploads/              # Uploaded images stored here
├── logs/                 # Recognized faces logged to CSV
├── templates/            # HTML pages (index, logs, upload)
├── static/               # Optional: custom CSS or images
```

---

## 🛠️ Setup Instructions

1. **Install dependencies** (in a virtualenv)
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your face data**
   - Create folders inside `dataset/` (e.g., `dataset/Ayush/`)
   - Put clear JPG/PNG images of faces inside each folder

3. **Generate encodings**
   ```bash
   python create_encodings.py
   ```

4. **Run the Flask app**
   ```bash
   python app.py
   ```

5. **Open in your browser**
   ```
   http://localhost:5000/
   ```

---

## 📤 Upload and Test

- Upload any image at `/upload`
- View recognition logs at `/logs`

---

## ⚠️ Privacy Notice

This public repo does **not include**:
- Personal images
- Pre-trained encodings
- Recognition logs

You must generate your own dataset and `.pkl` file before running.

---

## 🧠 Powered By

- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [Pandas](https://pandas.pydata.org/)

---

## 🧑‍💻 Author

Made with ❤️ by **Ayush**  
Feel free to fork and improve!
