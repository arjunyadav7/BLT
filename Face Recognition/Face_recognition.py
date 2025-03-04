import cv2
import os
import time
import json
from datetime import datetime
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
    matched_faces = []
    unknown_faces = 0
    
    # Extract faces from the frame using DeepFace's extract_faces
    try:
        detected_faces = DeepFace.extract_faces(frame, enforce_detection=False, detector_backend=detector_backend)
    except Exception as e:
        print(f"Face extraction error: {e}")
        return matched_faces, unknown_faces
    
    # Loop over each detected face
    for face_dict in detected_faces:
        face_img = face_dict["face"]  # This is a cropped face image (as a numpy array)
        match_found = False
        best_match = None
        best_similarity = None
        
        # Compare this face against every face in the database
        for name, img_path in faces_db.items():
            try:
                result = DeepFace.verify(face_img, img_path, enforce_detection=False, detector_backend=detector_backend)
                similarity = 1 - result["distance"]  # Convert distance to similarity (1 - distance)
                
                if result["verified"]:
                    if best_similarity is None or similarity > best_similarity:
                        best_match = name
                        best_similarity = similarity
                    match_found = True
            except Exception as e:
                print(f"Error processing {name}: {e}")
        
        if match_found:
            matched_faces.append({"name": best_match, "similarity": round(best_similarity, 4)})
        else:
            unknown_faces += 1
    
    return matched_faces, unknown_faces

# 3. Initialize Video Capture and Main Loop
cap = cv2.VideoCapture(0)
print("Face recognition started... Press 'q' to exit.")

last_checked = time.time()  # To track time interval

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Display the live video feed
    cv2.imshow("Face Recognition", frame)

    # Check faces every 10 seconds
    if time.time() - last_checked >= 10:
        last_checked = time.time()  # Update time
        matched_faces, unknown_faces = recognize_faces(frame, faces_db)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output = {
            "timestamp": timestamp,
            "matched_faces": matched_faces,
            "unknown_faces": unknown_faces
        }
        # Print output in JSON format
        print(json.dumps(output, indent=2))
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
