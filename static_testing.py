import cv2
import face_recognition
import pickle

# Load encodings
with open("encodings/known_faces.pkl", "rb") as f:
    data = pickle.load(f)

# Load the test image
test_img = cv2.imread("1.jpg")
rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Detect and encode faces
boxes = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, boxes)

for (top, right, bottom, left), encoding in zip(boxes, encodings):
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    face_distances = face_recognition.face_distance(data["encodings"], encoding)
    if len(face_distances) > 0:
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = data["names"][best_match_index]

    cv2.rectangle(test_img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(test_img, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Show result
cv2.imshow("Detected Faces", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
