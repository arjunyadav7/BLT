import cv2
import os
import json
from deepface import DeepFace

# 1. Load Known Faces from Database
DB_PATH = "Face_Images_DB"

def load_faces():
    faces = {}
    for file in os.listdir(DB_PATH):
        if file.lower().endswith(("jpg", "png", "jpeg")):
            name = os.path.splitext(file)[0]  # Use filename (without extension) as the person's name.
            faces[name] = os.path.join(DB_PATH, file)
    return faces

faces_db = load_faces()
if not faces_db:
    print("No images found in Face_Images_DB. Please add images first.")
    exit()

print(f"Loaded {len(faces_db)} known faces from the database.")

# 2. Face Recognition Function (Multiple Faces)
def recognize_faces(frame, faces_db, detector_backend="opencv", tolerance=0.68):
    matched_faces = []  # List of dicts: {"name": ..., "similarity": ...}
    unknown_faces = 0
    
    try:
        # Extract faces from the frame
        detected_faces = DeepFace.extract_faces(frame, enforce_detection=False, detector_backend=detector_backend)
    except Exception as e:
        print(f"Face extraction error: {e}")
        return matched_faces, unknown_faces
    
    for face_dict in detected_faces:
        face_img = face_dict["face"]  # Cropped face image (numpy array)
        best_match = None
        best_similarity = None
        
        for name, img_path in faces_db.items():
            try:
                result = DeepFace.verify(face_img, img_path, enforce_detection=False, detector_backend=detector_backend)
                # Convert cosine distance to similarity: similarity = 1 - distance
                similarity = 1 - result["distance"]
                if result["verified"]:
                    if best_similarity is None or similarity > best_similarity:
                        best_match = name
                        best_similarity = similarity
            except Exception as e:
                print(f"Error processing {name}: {e}")
        
        if best_match:
            matched_faces.append({"name": best_match, "similarity": round(best_similarity, 4)})
        else:
            unknown_faces += 1
    
    return matched_faces, unknown_faces

# 3. Process Video File and Aggregate Results
def process_video(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    # Use a dictionary to keep track of unique matched faces and best similarity so far
    unique_matches = {}  # key: name, value: similarity
    total_unknown_faces = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        frame_count += 1
        if frame_count % frame_interval == 0:  # Process every Nth frame
            matched_faces, unknown_faces = recognize_faces(frame, faces_db)
            # Update unknown count
            total_unknown_faces += unknown_faces
            # For each matched face, update unique_matches with best similarity
            for match in matched_faces:
                name = match["name"]
                similarity = match["similarity"]
                if name in unique_matches:
                    if similarity > unique_matches[name]:
                        unique_matches[name] = similarity
                else:
                    unique_matches[name] = similarity
    
    cap.release()
    
    # Create JSON output with unique matches and unknown face count
    output = {
        "matched_faces": [{"name": name, "similarity": unique_matches[name]} for name in unique_matches],
        "unknown_faces": total_unknown_faces
    }
    print(json.dumps(output, indent=2))

# Example usage: Change video_path to the path of your video file.
video_path = "C:/Users/arjun/Pictures/Camera Roll/WIN_20250304_16_44_20_Pro.mp4"
process_video(video_path, frame_interval=10)
