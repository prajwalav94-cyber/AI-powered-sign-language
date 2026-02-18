import cv2
import mediapipe as mp
import joblib
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# =========================
# FONTS
# =========================
font_unicode = ImageFont.truetype(
    "NotoSansDevanagari-VariableFont_wdth,wght.ttf",
    40
)
font_en = ImageFont.load_default()
font_kn = ImageFont.truetype(
    "NotoSansKannada-VariableFont_wdth,wght.ttf",
    40
)

# =========================
# LANGUAGE SETTINGS
# =========================
LANGUAGE = "en"

TRANSLATIONS = {
    "en": {
        "Hello": "Hello",
        "Yes": "Yes",
        "No": "No",
        "Thank You": "Thank You",
        "Sorry": "Sorry",
        "Good Morning": "Good Morning",
        "OK": "OK"
    },
    "hi": {
        "Hello": "नमस्ते",
        "Yes": "हाँ",
        "No": "नहीं",
        "Thank You": "धन्यवाद",
        "Sorry": "माफ़ कीजिए",
        "Good Morning": "सुप्रभात",
        "OK": "ठीक है"
    },
    "kn": {
        "Hello": "ನಮಸ್ಕಾರ",
        "Yes": "ಹೌದು",
        "No": "ಇಲ್ಲ",
        "Thank You": "ಧನ್ಯವಾದಗಳು",
        "Sorry": "ಕ್ಷಮಿಸಿ",
        "Good Morning": "ಶುಭೋದಯ",
        "OK": "ಸರಿ"
    }
}

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")

GESTURES = [
    "Hello",
    "Yes",
    "No",
    "Thank You",
    "Sorry",
    "Good Morning",
    "OK"
]

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, None

    hand = result.multi_hand_landmarks[0]
    points = []

    for lm in hand.landmark:
        points.extend([lm.x, lm.y, lm.z])

    return points, hand

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press E = English | H = Hindi | K = Kannada | Q = Quit")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    landmarks, hand_lms = extract_landmarks(frame)

    if hand_lms:
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    if landmarks:
        data = np.array(landmarks).reshape(1, -1)
        pred = model.predict(data)[0]
        gesture = GESTURES[pred]
        display_text = TRANSLATIONS[LANGUAGE][gesture]

        cv2.rectangle(frame, (10, 10), (520, 90), (0, 0, 0), -1)

        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        if LANGUAGE == "en":
            draw.text((20, 30), display_text, font=font_en, fill=(0, 255, 0))
        elif LANGUAGE == "hi":
            draw.text((20, 30), display_text, font=font_unicode, fill=(0, 255, 0))
        elif LANGUAGE == "kn":
            draw.text((20, 30), display_text, font=font_kn, fill=(0, 255, 0))

        frame = np.array(img_pil)

    else:
        cv2.putText(
            frame,
            "Show a hand...",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            3
        )

    cv2.putText(
        frame,
        f"Language: {LANGUAGE.upper()} (E/H/K)",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.imshow("Sign Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        LANGUAGE = "en"
    elif key == ord('h'):
        LANGUAGE = "hi"
    elif key == ord('k'):
        LANGUAGE = "kn"

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

