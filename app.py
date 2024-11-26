from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

app = Flask(__name__)

# Paths and global variables
path = 'photos'
image_paths = []
classNames = []
attendance_record = set()
attendance_data = []  # To store attendance data for real-time updates

# Flag to control camera
camera_active = False

# Load images and create encodings
myList = os.listdir(path)
for cl in myList:
    imgPath = os.path.join(path, cl)
    if os.path.isfile(imgPath):
        image_paths.append(imgPath)
        classNames.append(os.path.splitext(cl)[0])

def findEncodings(image_paths):
    encodeList = []
    for imgPath in image_paths:
        img = cv2.imread(imgPath)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(image_paths)
print('Encoding Complete')

def markAttendance(name):
    global attendance_data
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    
    # Avoid duplicate entries
    if name not in attendance_record:
        attendance_record.add(name)
        attendance_data.append({'name': name, 'time': dtString})
        with open('Attendance.csv', 'a') as f:
            f.writelines(f'\n{name},{dtString}')

# Video streaming function
def generate_frames():
    global camera_active
    cap = cv2.VideoCapture(0)
    while camera_active:
        success, imgS = cap.read()
        if not success:
            break
        else:
            imgS = cv2.resize(imgS, (640, 480))
            rgb_frame = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(rgb_frame, model='hog')  # Faster model
            encodesCurFrame = face_recognition.face_encodings(rgb_frame, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    cv2.rectangle(imgS, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(imgS, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(imgS, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', imgS)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    global camera_active
    camera_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'status': 'Camera stopped'})

@app.route('/get_attendance')
def get_attendance():
    return jsonify(attendance_data)

if __name__ == "__main__":
    app.run(debug=True)
