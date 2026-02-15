import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
from collections import deque

# -----------------------------
# IMAGE CONFIG
# -----------------------------
IMAGE_PATHS = {
    "THUMBS_UP": "thumbs_up.jpg",
    "POINTING": "pointing.jpg",
    "THINKING": "thinking.jpg",
    "STOP": "Stop.png",
    "OK": "ok.jpg",
    "FIST": "fist.jpg",
    "NAMASTE": "namaste.jpg",
    "HEART": "heart.jpg",
    "MIDDLE_FINGER": "middle_finger.jpg",
    "SHY": "shy.jpg",
    "NEUTRAL": "neutral.jpg",
}

# -----------------------------
# MEDIAPIPE INIT
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# LOAD IMAGES
# -----------------------------
gesture_images = {}

for k, path in IMAGE_PATHS.items():
    img = cv2.imread(path)
    if img is None:
        print(f"[WARNING] Image not found or failed to load: {path}")
    gesture_images[k] = img

# -----------------------------
# CONSTANTS
# -----------------------------
EPS = 0.02
gesture_buffer = deque(maxlen=7)

thinking_frame_count = 0
THINKING_REQUIRED_FRAMES = 8

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def folded(tip, pip):
    return tip.y > pip.y - EPS


def thumb_folded(thumb_tip, thumb_ip):
    return thumb_tip.y > thumb_ip.y - EPS


def palm_facing_camera(lm):
    return lm[0].z < lm[9].z  # wrist closer than middle MCP


def all_extended_relaxed(lm):
    return (
        lm[8].y < lm[6].y - EPS and
        lm[12].y < lm[10].y - EPS and
        lm[16].y < lm[14].y - EPS
    )


def classify_single_hand(hand_landmarks):
    lm = hand_landmarks.landmark

    thumb_tip, thumb_ip = lm[4], lm[3]
    index_tip, index_pip = lm[8], lm[6]
    middle_tip, middle_pip = lm[12], lm[10]
    ring_tip, ring_pip = lm[16], lm[14]
    pinky_tip, pinky_pip = lm[20], lm[18]

    # ✋ STOP (robust – matches real camera pose)
    index_extended  = lm[8].y < lm[6].y - EPS
    middle_extended = lm[12].y < lm[10].y - EPS
    ring_extended   = lm[16].y < lm[14].y - EPS
    pinky_extended  = lm[20].y < lm[18].y - EPS

    # hand spread (index ↔ pinky distance)
    hand_width = abs(lm[8].x - lm[20].x)

    if (
        index_extended and
        middle_extended and
        ring_extended and
        pinky_extended and
        hand_width > 0.22
    ):
        return "STOP"



    # 🖕 MIDDLE FINGER
    if (
        middle_tip.y < middle_pip.y - EPS and
        folded(index_tip, index_pip) and
        folded(ring_tip, ring_pip) and
        folded(pinky_tip, pinky_pip)
    ):
        return "MIDDLE_FINGER"

    # ✊ FIST
    if (
        folded(index_tip, index_pip) and
        folded(middle_tip, middle_pip) and
        folded(ring_tip, ring_pip) and
        folded(pinky_tip, pinky_pip) and
        thumb_folded(thumb_tip, thumb_ip)
    ):
        return "FIST"

    # 👍 THUMBS UP
    if (
        thumb_tip.y < thumb_ip.y - EPS and
        folded(index_tip, index_pip) and
        folded(middle_tip, middle_pip)
    ):
        return "THUMBS_UP"

    # 👉 POINTING
    if (
        index_tip.y < index_pip.y - EPS and
        folded(middle_tip, middle_pip) and
        folded(ring_tip, ring_pip)
    ):
        return "POINTING"

    # 👌 OK
    if np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) < 0.04:
        return "OK"

    return "NEUTRAL"


def check_shy(hand_landmarks, face_landmarks, w, h):
    lm = hand_landmarks.landmark
    index_tip = lm[8]
    middle_tip = lm[12]
    mouth = face_landmarks.landmark[13]

    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
    mx, my = int(mouth.x * w), int(mouth.y * h)

    shy_fingers = (
        index_tip.y > lm[6].y - EPS and
        middle_tip.y > lm[10].y - EPS
    )

    return np.hypot(ix - mx, iy - my) < 0.10 * w and shy_fingers


def check_indian_thinking(hand_landmarks, face_landmarks, w, h):
    index_tip = hand_landmarks.landmark[8]
    ix, iy = int(index_tip.x * w), int(index_tip.y * h)

    for idx in [4, 13, 152]:
        fx = int(face_landmarks.landmark[idx].x * w)
        fy = int(face_landmarks.landmark[idx].y * h)
        if np.hypot(ix - fx, iy - fy) < 0.12 * w:
            return True
    return False


def check_namaste(hand1, hand2):
    lm1, lm2 = hand1.landmark, hand2.landmark

    palm_dist = np.hypot(lm1[0].x - lm2[0].x, lm1[0].y - lm2[0].y)

    fingers_up_1 = lm1[8].y < lm1[6].y
    fingers_up_2 = lm2[8].y < lm2[6].y

    return palm_dist < 0.07 and fingers_up_1 and fingers_up_2



def check_heart(hand1, hand2):
    i1, t1 = hand1.landmark[8], hand1.landmark[4]
    i2, t2 = hand2.landmark[8], hand2.landmark[4]
    return (
        np.hypot(i1.x - i2.x, i1.y - i2.y) < 0.06 and
        np.hypot(t1.x - t2.x, t1.y - t2.y) < 0.06
    )

# -----------------------------
# MAIN LOOP
# -----------------------------
cap = cv2.VideoCapture(0)
print("Indian Gesture System Running... Press Q or ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_res = hands.process(rgb)
    face_res = face_mesh.process(rgb)

    detected = "NEUTRAL"
    hands_list = hand_res.multi_hand_landmarks
    face_lm = face_res.multi_face_landmarks[0] if face_res.multi_face_landmarks else None

    if hands_list:
        # ---- 2 HAND GESTURES ----
        if len(hands_list) == 2:
            if check_namaste(hands_list[0], hands_list[1]):
                detected = "NAMASTE"
            elif check_heart(hands_list[0], hands_list[1]):
                detected = "HEART"

        # ---- SINGLE HAND ----
        else:
            hand = hands_list[0]

            if face_lm and check_shy(hand, face_lm, w, h):
                detected = "SHY"
                thinking_frame_count = 0

            elif face_lm and check_indian_thinking(hand, face_lm, w, h):
                thinking_frame_count += 1
                if thinking_frame_count >= THINKING_REQUIRED_FRAMES:
                    detected = "THINKING"

            else:
                thinking_frame_count = 0
                detected = classify_single_hand(hand)

        for hnd in hands_list:
            mp_drawing.draw_landmarks(frame, hnd, mp_hands.HAND_CONNECTIONS)
    else:
        thinking_frame_count = 0

    # TEMPORAL SMOOTHING
    gesture_buffer.append(detected)
    stable = max(set(gesture_buffer), key=gesture_buffer.count)

    # DISPLAY
    img = gesture_images.get(stable)
    if img is not None:
        r = h / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * r), h))
        frame = np.concatenate((frame, img), axis=1)
        text_x = w + 10
    else:
        text_x = 10

    cv2.putText(
        frame,
        f"Gesture: {stable}",
        (text_x, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(5) & 0xFF in [27, ord('q')]:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
hands.close()
face_mesh.close()
cv2.destroyAllWindows()
