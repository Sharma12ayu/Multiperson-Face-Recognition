import face_recognition
import os
import pickle
from imutils import paths

dataset_path = "../dataset/"
output_file = "../encodings/known_faces.pkl"

known_encodings = []
known_names = []

image_paths = list(paths.list_images(dataset_path))

for img_path in image_paths:
    name = img_path.split(os.path.sep)[-2]
    print(f"[INFO] Processing: {name} from {img_path}")

    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)

data = {"encodings": known_encodings, "names": known_names}
with open(output_file, "wb") as f:
    f.write(pickle.dumps(data))

print(f"[INFO] Encodings saved to {output_file}")
