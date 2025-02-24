from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import os
import time

app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO("yolov8_trained_model.pt")
print("Model classes:", model.names)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Directory to save detected frames
save_directory = "cheating_frames"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

confidence_threshold = 0.4  # Confidence threshold for "cheating" detection

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)

        # Detect cheating (person detected with high confidence)
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())
            class_name = model.names[class_id]
            
            if class_name != "normal" and confidence > confidence_threshold:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_directory, f"{class_name}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Cheating detected! Frame saved as {filename}")

        # Draw bounding boxes on the frame
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            class_id = int(result.cls[0].item())
            class_name = model.names[class_id]
            confidence = result.conf[0].item()

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the frame to JPEG format and send it to the frontend
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('live_stream.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
