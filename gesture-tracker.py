import cv2
import mediapipe as mp
import numpy as np
import os

# -----------------------------
# IMAGE CONFIG
# -----------------------------
IMAGE_PATHS = {
    "THUMBS_UP": "thumbs_up.jpg",
    "POINTING": "pointing.jpg",
    "NEUTRAL": "neutral.jpg",
    "THINKING": "thinking.jpg"
}

# -----------------------------
# MEDIAPIPE INIT
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# LOAD IMAGES ONCE (IMPORTANT)
# -----------------------------
gesture_images = {}
for key, path in IMAGE_PATHS.items():
    img = cv2.imread(path)
    if img is None:
        print(f"[WARNING] Missing image: {path}")
    gesture_images[key] = img

# -----------------------------
# GESTURE CLASSIFICATION
# -----------------------------
def classify_gesture(hand_landmarks):
    y_thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    y_index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    y_middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    y_ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    y_pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    y_middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

    # 👍 Thumbs Up
    if (
        y_thumb_tip < y_middle_pip and
        y_index_tip > y_middle_pip and
        y_middle_tip > y_middle_pip and
        y_ring_tip > y_middle_pip and
        y_pinky_tip > y_middle_pip
    ):
        return "THUMBS_UP"

    # 👉 Pointing
    if (
        y_index_tip < y_middle_pip and
        y_middle_tip > y_middle_pip and
        y_ring_tip > y_middle_pip and
        y_pinky_tip > y_middle_pip and
        y_thumb_tip > y_middle_pip
    ):
        return "POINTING"

    return "NEUTRAL"


def check_thinking_gesture(hand_landmarks, face_landmarks, frame_width, frame_height):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

    nose_tip = face_landmarks.landmark[4]
    nose_x, nose_y = int(nose_tip.x * frame_width), int(nose_tip.y * frame_height)

    distance = np.sqrt((index_x - nose_x) ** 2 + (index_y - nose_y) ** 2)

    MAX_DISTANCE = int(0.06 * frame_width)

    y_middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    y_middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

    if distance < MAX_DISTANCE and y_middle_tip > y_middle_pip:
        return True

    return False

# -----------------------------
# MAIN LOOP
# -----------------------------
cap = cv2.VideoCapture(0)
print("Gesture Tracker running. Press 'q' or ESC to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    current_gesture = "NEUTRAL"

    hand_landmarks = (
        hand_results.multi_hand_landmarks[0]
        if hand_results.multi_hand_landmarks else None
    )

    face_landmarks = (
        face_results.multi_face_landmarks[0]
        if face_results.multi_face_landmarks else None
    )

    if hand_landmarks:
        if face_landmarks and check_thinking_gesture(
            hand_landmarks, face_landmarks, frame_width, frame_height
        ):
            current_gesture = "THINKING"
        else:
            current_gesture = classify_gesture(hand_landmarks)

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2),
            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2)
        )

    gesture_image = gesture_images.get(current_gesture)

    if gesture_image is not None:
        ratio = frame_height / gesture_image.shape[0]
        resized = cv2.resize(
            gesture_image,
            (int(gesture_image.shape[1] * ratio), frame_height)
        )
        output_frame = np.concatenate((frame, resized), axis=1)
        text_x = frame_width + 10
    else:
        output_frame = frame
        text_x = 10

    cv2.putText(
        output_frame,
        f"Gesture: {current_gesture.replace('_', ' ')}",
        (text_x, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture & Image Pairing", output_frame)

    key = cv2.waitKey(5)
    if key == ord('q') or key == 27:
        break

# -----------------------------
# CLEANUP
# -----------------------------
hands.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()

