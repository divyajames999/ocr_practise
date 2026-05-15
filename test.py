from ultralytics import YOLO
import cv2
import pyttsx3

# Voice engine
engine = pyttsx3.init()

# Load model
model = YOLO("yolov8n.pt")

# Load image
image = cv2.imread("images/test1.jpg")

# Detect objects
results = model(image)

# Important objects only
important_objects = [
    "person",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "chair",
    "dog"
]

spoken_objects = []

# Process detections
for box in results[0].boxes:

    class_id = int(box.cls[0])

    object_name = model.names[class_id]

    confidence = float(box.conf[0])

    # Confidence filter
    if confidence > 0.50:

        # Important object filter
        if object_name in important_objects:

            # Avoid repeating same object
            if object_name not in spoken_objects:

                spoken_objects.append(object_name)

                sentence = f"{object_name} detected ahead"

                print(sentence)

                engine.say(sentence)

# Speak everything
engine.runAndWait()

# Show image
annotated_image = results[0].plot()

cv2.imshow("Detection", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()