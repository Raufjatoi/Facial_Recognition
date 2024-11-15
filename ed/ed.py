import cv2
import mediapipe as mp

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to count fingers
def count_fingers(hand_landmarks):
    """Count the number of fingers shown based on hand landmarks."""
    finger_count = 0
    # For each finger (tip landmarks 4, 8, 12, 16, 20)
    for finger_tip in [4, 8, 12, 16, 20]:
        # If the finger tip is above the adjacent joint (finger curled up), it's extended
        if hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_tip - 2].y:
            finger_count += 1
    return finger_count

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Draw the landmarks and connections if hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the fingers and display the count
            finger_count = count_fingers(hand_landmarks)
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame with the finger count
    cv2.imshow("Finger Count", frame)

    # Exit when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
