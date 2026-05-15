from ultralytics import YOLO
import cv2
import pyttsx3
import math
import time
import threading
import queue

# ============================================
# TEXT TO SPEECH SETUP
# ============================================

speech_queue = queue.Queue()

engine = pyttsx3.init()

# Speech speed
engine.setProperty('rate', 150)


def speak_worker():

    while True:

        text = speech_queue.get()

        if text is None:
            break

        engine.say(text)
        engine.runAndWait()

        speech_queue.task_done()


# Start speech thread
speech_thread = threading.Thread(
    target=speak_worker,
    daemon=True
)

speech_thread.start()

# ============================================
# LOAD YOLO MODEL
# ============================================

model = YOLO("yolov8n.pt")

# ============================================
# VIDEO SOURCE
# ============================================

cap = cv2.VideoCapture("test_video.mp4")

# Webcam:
# cap = cv2.VideoCapture(0)

# ============================================
# IMPORTANT OBJECTS
# ============================================

important_objects = [
    "person",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "dog"
]

# ============================================
# STORAGE
# ============================================

previous_positions = {}

last_spoken_time = {}

# ============================================
# SETTINGS
# ============================================

# Repeat speech every X seconds
SPEAK_DELAY = 2.5

# Motion sensitivity
MOVEMENT_THRESHOLD = 3

# ============================================
# MAIN LOOP
# ============================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------------------------
    # TRACK OBJECTS
    # ----------------------------------------

    results = model.track(frame, persist=True)

    current_positions = {}

    best_alert = None

    boxes = results[0].boxes

    # Make sure IDs exist
    if boxes.id is not None:

        ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):

            class_id = int(box.cls[0])

            object_name = model.names[class_id]

            confidence = float(box.conf[0])

            # Filter objects
            if confidence > 0.5 and object_name in important_objects:

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Center point
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                # Unique object ID
                object_key = f"{object_name}_{track_id}"

                # Save current position
                current_positions[object_key] = (
                    x_center,
                    y_center
                )

                moving = False

                # ----------------------------------------
                # CHECK MOVEMENT
                # ----------------------------------------

                if object_key in previous_positions:

                    prev_x, prev_y = previous_positions[object_key]

                    distance = math.sqrt(
                        (x_center - prev_x) ** 2 +
                        (y_center - prev_y) ** 2
                    )

                    if distance > MOVEMENT_THRESHOLD:
                        moving = True

                # ----------------------------------------
                # DETERMINE DIRECTION
                # ----------------------------------------

                frame_width = frame.shape[1]

                if x_center < frame_width / 3:

                    direction = "left"
                    priority = 2

                elif x_center < 2 * frame_width / 3:

                    direction = "ahead"
                    priority = 1

                else:

                    direction = "right"
                    priority = 2

                # ----------------------------------------
                # MOVING OBJECT ALERT
                # ----------------------------------------

                if moving:

                    sentence = f"{object_name} {direction}"

                    print("Moving:", sentence)

                    # Keep highest priority alert
                    if (
                        best_alert is None
                        or priority < best_alert[0]
                    ):

                        best_alert = (
                            priority,
                            sentence,
                            object_key
                        )

                # ----------------------------------------
                # DRAW BOX
                # ----------------------------------------

                if moving:

                    color = (0, 255, 0)
                    label = "MOVING"

                else:

                    color = (100, 100, 100)
                    label = "STATIONARY"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{object_name} {label}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    # ----------------------------------------
    # SPEAK BEST ALERT ONLY
    # ----------------------------------------

    if best_alert is not None:

        _, sentence, object_key = best_alert

        current_time = time.time()

        if (
            current_time -
            last_spoken_time.get(object_key, 0)
            > SPEAK_DELAY
        ):

            print("SPEAK:", sentence)

            speech_queue.put(sentence)

            last_spoken_time[object_key] = current_time

    # ----------------------------------------
    # UPDATE POSITIONS
    # ----------------------------------------

    previous_positions = current_positions

    # ----------------------------------------
    # SHOW OUTPUT
    # ----------------------------------------

    cv2.imshow("Smart Cane AI", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# CLEANUP
# ============================================

speech_queue.put(None)

cap.release()

cv2.destroyAllWindows()