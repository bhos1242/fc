from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import dlib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Load pre-trained models
pose_predictor = dlib.shape_predictor("C:/Users/Nikhil Darji/Downloads/shape_predictor_68_face_landmarks_GTX.dat/shape_predictor_68_face_landmarks_GTX.dat")
face_encoder = dlib.face_recognition_model_v1("C:/Users/Nikhil Darji/Downloads/dlib_face_recognition_resnet_model_v1/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()

# Define facial regions and colors
FEATURE_REGIONS = {
    "eyes": list(range(36, 48)),
    "nose": list(range(27, 36)),
    "mouth": list(range(48, 68)),
    "jawline": list(range(0, 17))
}
FEATURE_COLORS = {
    "eyes": (255, 0, 0),
    "nose": (0, 255, 0),
    "mouth": (0, 0, 255),
    "jawline": (255, 255, 0)
}

# Detect face and landmarks
def get_face_landmarks(image):
    faces = face_detector(image, 1)
    if len(faces) == 0:
        raise ValueError("No face detected.")
    face_location = faces[0]
    landmarks = pose_predictor(image, face_location)
    return landmarks

# Encode face
def encode_face(image, landmarks):
    aligned_face = dlib.get_face_chip(image, landmarks)
    encodings = np.array(face_encoder.compute_face_descriptor(aligned_face))
    return encodings

# Compute feature-wise similarity
def compute_feature_similarity(landmarks1, landmarks2):
    similarities = {}
    for feature, indices in FEATURE_REGIONS.items():
        points1 = np.array([[landmarks1.part(i).x, landmarks1.part(i).y] for i in indices])
        points2 = np.array([[landmarks2.part(i).x, landmarks2.part(i).y] for i in indices])
        norm_points1 = points1 - points1.mean(axis=0)
        norm_points2 = points2 - points2.mean(axis=0)
        similarity = np.linalg.norm(norm_points1 - norm_points2)
        similarities[feature] = similarity
    return similarities

# Visualize features
def visualize_features(image, landmarks, output_path):
    for feature, indices in FEATURE_REGIONS.items():
        for i in indices:
            point = (landmarks.part(i).x, landmarks.part(i).y)
            cv2.circle(image, point, 2, FEATURE_COLORS[feature], -1)
    cv2.imwrite(output_path, image)

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        # Get uploaded files
        file1 = request.files["image1"]
        file2 = request.files["image2"]
        file1_path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
        file2_path = os.path.join(app.config["UPLOAD_FOLDER"], file2.filename)
        file1.save(file1_path)
        file2.save(file2_path)

        # Read and process images
        img1 = cv2.imread(file1_path)
        img2 = cv2.imread(file2_path)
        img1 = cv2.resize(img1, (600, 600))
        img2 = cv2.resize(img2, (600, 600))

        # Detect landmarks
        landmarks1 = get_face_landmarks(img1)
        landmarks2 = get_face_landmarks(img2)

        # Compute encodings and similarities
        encodings1 = encode_face(img1, landmarks1)
        encodings2 = encode_face(img2, landmarks2)
        overall_similarity = np.linalg.norm(encodings1 - encodings2)
        feature_similarities = compute_feature_similarity(landmarks1, landmarks2)

        # Annotate images
        output1 = os.path.join(app.config["OUTPUT_FOLDER"], "annotated1.jpg")
        output2 = os.path.join(app.config["OUTPUT_FOLDER"], "annotated2.jpg")
        visualize_features(img1.copy(), landmarks1, output1)
        visualize_features(img2.copy(), landmarks2, output2)

        # Prepare response
        return jsonify({
            "overall_similarity": overall_similarity,
            "feature_similarities": feature_similarities,
            "annotated1": "annotated1.jpg",
            "annotated2": "annotated2.jpg"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
