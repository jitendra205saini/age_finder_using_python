import cv2
import dlib
import numpy as np

age_weights = "D:/Machine larning project/Age finder/age_model/age_net.caffemodel"
age_config = "D:/Machine larning project/Age finder/age_model/age_deploy.prototxt"

try:
    age_Net = cv2.dnn.readNet(age_config, age_weights)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

ageList = ['(0-3)', '(4-7)', '(8-12)', '(13-14)', '(15-20)', '(21-24)', '(25-32)', '(33-43)', '(44-59)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

face_detector = dlib.get_frontal_face_detector()

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_frame)
    
    for face in faces:
        x = face.left()
        y = face.top()
        x2 = face.right()
        y2 = face.bottom()

        if x < 0: x = 0
        if y < 0: y = 0
        if x2 > frame.shape[1]: x2 = frame.shape[1]
        if y2 > frame.shape[0]: y2 = frame.shape[0]

        face_image = frame[y:y2, x:x2]

        if face_image.size == 0:
            print("Detected an invalid face area, skipping...")
            continue

        try:
            blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), model_mean, swapRB=False)

            age_Net.setInput(blob)
            age_preds = age_Net.forward()
            
            print(f"Age Predictions: {age_preds}")

            age = ageList[age_preds[0].argmax()]

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 200, 200), 2)

            cv2.putText(frame, f'Age: {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error during processing: {e}")
            continue

    cv2.imshow("Webcam Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == 13:
        break

video_capture.release()
cv2.destroyAllWindows()
