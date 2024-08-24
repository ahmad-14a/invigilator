import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import utils

# Define constants for eye tracking
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Initialize MediaPipe face mesh and face detection
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Set up face detection model
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize face mesh model
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def landmark_detect(img, res, draw=False):
    height = img.shape[0]
    width = img.shape[1]
    mesh_coor = [(int(point.x * width), int(point.y * height)) for point in res.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coor]
    return mesh_coor

def eyes_extractor(img, right_eye_corr, left_eye_corr):
    cv2.polylines(img, [np.array(right_eye_corr, dtype=np.int32)], True, (0, 255, 0), 1)
    cv2.polylines(img, [np.array(left_eye_corr, dtype=np.int32)], True, (0, 255, 0), 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)

    cv2.fillPoly(mask, [np.array(right_eye_corr, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_corr, dtype=np.int32)], 255)

    eyes = cv2.bitwise_and(gray, gray, mask=mask)
    # cv2.imshow("Eye Draw", eyes)
    eyes[mask == 0] = 155

    r_min_x = min(right_eye_corr, key=lambda item: item[0])[0]
    r_max_x = max(right_eye_corr, key=lambda item: item[0])[0]
    r_min_y = min(right_eye_corr, key=lambda item: item[1])[1]
    r_max_y = max(right_eye_corr, key=lambda item: item[1])[1]

    l_min_x = min(left_eye_corr, key=lambda item: item[0])[0]
    l_max_x = max(left_eye_corr, key=lambda item: item[0])[0]
    l_min_y = min(left_eye_corr, key=lambda item: item[1])[1]
    l_max_y = max(left_eye_corr, key=lambda item: item[1])[1]

    cropped_right = eyes[r_min_y:r_max_y, r_min_x:r_max_x]
    cropped_left = eyes[l_min_y:l_max_y, l_min_x:l_max_x]

    return cropped_right, cropped_left

def pos_estimation(cropped_eye):
    h, w = cropped_eye.shape

    # gaussian_blur = cv2.GaussianBlur(cropped_eye, (9, 9), 0)
    if cropped_eye is not None and cropped_eye.size > 0:
        gaussian_blur = cv2.GaussianBlur(cropped_eye, (7, 7), 0)
    else:
        print("Empty eye region detected.")
        return "UNKNOWN", [0, 0, 0]
    
    median_blur = cv2.medianBlur(gaussian_blur, 3)

    thres_eye = cv2.adaptiveThreshold(median_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    piece = int(w / 3)

    right_piece = thres_eye[0:h, 0:piece]
    center_piece = thres_eye[0:h, piece:piece + piece]
    left_piece = thres_eye[0:h, piece + piece:w]

    eye_pos, color = pixel_counter(right_piece, center_piece, left_piece)

    if np.sum(thres_eye == 0) > (0.7* h * w):
        eye_pos = "CLOSED"
        color = [utils.RED, utils.BLACK]

    return eye_pos, color

def pixel_counter(first_piece, second_piece, third_piece):
    # print("-----------------------------------------")
    # print(f"First piece: {np.sum(first_piece == 0)}")
    # print(f"Second piece: {np.sum(second_piece == 0)}")
    # print(f"Third piece: {np.sum(third_piece == 0)}")
    
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)

    eye_parts = [right_part, center_part, left_part]

    max_ind = eye_parts.index(max(eye_parts))
    threshold = max(5, 0.1 * (right_part + left_part))
    
    # if right_part > left_part and (right_part / (left_part + 1)) > 1.2:
    #     pos_eye = "RIGHT"
    #     color = [utils.BLACK, utils.RED]
    # elif left_part > right_part and (left_part / (right_part + 1)) > 1.2:
    #     pos_eye = "LEFT"
    #     color = [utils.BLACK, utils.RED]
    # else:
    #     pos_eye = "CENTER"
    #     color = [utils.BLACK, utils.GREEN]

    # if max_ind == 0 and (eye_parts[0] > 1.2 * eye_parts[1]):
    #     pos_eye = "RIGHT"
    #     color = [utils.BLACK, utils.RED]
    # elif max_ind == 2 and (eye_parts[2] > 1.2 * eye_parts[1]):
    #     pos_eye = "LEFT"
    #     color = [utils.BLACK, utils.RED]
    
    right_ratio = right_part / (left_part + center_part)
    left_ratio = left_part / (right_part + center_part)
    
    # if right_ratio > 1.2:
    #     pos_eye = "RIGHT"
    #     color = [utils.BLACK, utils.RED]
    # elif left_ratio > 1.2:
    #     pos_eye = "LEFT"
    #     color = [utils.BLACK, utils.RED]
    # else:
    #     pos_eye = "CENTER"
    
    # if max_ind == 0 and (eye_parts[0] > eye_parts[2] + 15) and (eye_parts[0] > eye_parts[1] + 15):
    #     pos_eye = "RIGHT"
    #     color = [utils.BLACK, utils.RED]
    # elif max_ind == 2 and (eye_parts[2] > eye_parts[0] + 15) and (eye_parts[2] > eye_parts[1] + 15):
    #     pos_eye = "LEFT"
    #     color = [utils.BLACK, utils.RED]

    if max_ind == 0 and (eye_parts[0] > eye_parts[1] - 5):
        pos_eye = "Looking Away"
        color = [utils.RED, utils.BLACK]
    elif max_ind == 2 and (eye_parts[2] > eye_parts[1] - threshold):
        pos_eye = "Looking Away"
        color = [utils.RED, utils.BLACK]
    else:
        pos_eye = "CENTER"
        color = [utils.BLACK, utils.GREEN]
        
    # with open("eye_tracking.txt", "a") as file:
    #     file.write(f"Right: {right_part}, Center: {center_part}, Left: {left_part} ::: detected: {pos_eye}\n")

    return pos_eye, color


# Video processor class
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame
        processed_frame, num_faces = self.process_frame(img)

        # Return the processed frame
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

    def process_frame(self, frame):
        # Flip the image horizontally and convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Process the frame for face detection
        results_multiface = face_detection.process(image)

        # Draw face detections and count the number of faces
        num_faces = 0
        if results_multiface.detections:
            num_faces = len(results_multiface.detections)

        # Get the result
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                # Convert to NumPy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera matrix and distortion parameters
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix and angles
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # Determine head pose
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                # Display the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Extract eye regions
                face_landmarks = landmark_detect(image, results)
                right_eye_corr = [face_landmarks[i] for i in RIGHT_EYE]
                left_eye_corr = [face_landmarks[i] for i in LEFT_EYE]
                cropped_right, cropped_left = eyes_extractor(image, right_eye_corr, left_eye_corr)

                # Estimate eye position
                eye_pos_right, color_right = pos_estimation(cropped_right)
                eye_pos_left, color_left = pos_estimation(cropped_left)

                # Display eye position
                cv2.putText(image, f"Right Eye: {eye_pos_right}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color_right[0], 2)
                cv2.putText(image, f"Left Eye: {eye_pos_left}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color_left[0], 2)
        
                if num_faces > 1:
                    cv2.putText(image, f'Faces detected: {num_faces}', (20, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(image, f'Faces detected: {num_faces}', (20, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image, num_faces

# Streamlit app
st.title("Real-Time Eye Tracking with Webcam")

# Start the webcam stream
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)