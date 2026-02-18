print("STARTING DATA COLLECTION SCRIPT...")
print("DEBUG: Importing modules...")

import cv2
print("DEBUG: cv2 imported OK")

import mediapipe as mp
print("DEBUG: mediapipe imported OK")

import pandas as pd
print("DEBUG: pandas imported OK")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Gesture list
GESTURES = {
    "1": ("Hello", 0),
    "2": ("Yes", 1),
    "3": ("No", 2),
    "4": ("Thank You", 3),
    "5": ("Sorry", 4),
    "6": ("Good Morning", 5),
    "7": ("OK", 6),
}

# Extract hand landmarks
def extract_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0]
    output = []
    for p in lm.landmark:
        output.extend([p.x, p.y, p.z])
    return output


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: CAMERA FAILED TO OPEN!")
    exit()

print("Camera ready.")
print("\n---------------------------------")
print("PRESS A NUMBER TO START CAPTURING")
print("---------------------------------")
print("1 = Hello")
print("2 = Yes")
print("3 = No")
print("4 = Thank You")
print("5 = Sorry")
print("6 = Good Morning")
print("7 = OK")
print("---------------------------------")
print("Press q to quit\n")

df = []

while True:
    key = input("Enter gesture number: ")

    if key == "q":
        break

    if key not in GESTURES:
        print("Invalid key! Try again.")
        continue

    gesture_name, label = GESTURES[key]

    print(f"\n--- Get ready for gesture: {gesture_name} ---")
    input("Press ENTER to start capturing...")

    count = 0
    while count < 200:
        ret, frame = cap.read()
        if not ret:
            continue

        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            df.append(landmarks + [label])
            count += 1

        cv2.putText(frame, f"{gesture_name}: {count}/200", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Collecting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Finished capturing: {gesture_name}!\n")

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(df)
df.to_csv("gesture_data.csv", index=False)
print("Saved dataset to gesture_data.csv")

