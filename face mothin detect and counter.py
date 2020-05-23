import os
import cv2
import dlib

cap = cv2.VideoCapture(0)
detect = dlib.get_frontal_face_detector()
pradict = dlib.shape_predictor(os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat'))#'/home/ashrafi/Documents/eye desktop project/shape_predictor_68_face_landmarks.dat')

temp_font = 0
temp_left = 0
temp_right = 0

right = 0
left = 0

cal = lambda x1, x: x1-x

while cap.isOpened():
    ref, frame = cap.read()
    faces  = detect(frame)
    for face in faces:

        width = face.right()- face.left()

        landmarks = pradict(frame, face)
        #left eye
        x0 = landmarks.part(0).x
        x36 = landmarks.part(36).x
        # Right eye
        x16 = landmarks.part(16).x
        x45 = landmarks.part(45).x

        # Local calculation
        value = cal(x16, x45) - cal(x36, x0)

        if value > 15:
            cv2.putText(frame, 'Right', (45, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            temp_right += 1
            temp_left = 0
            temp_font = 0
        elif value < -15:
            cv2.putText(frame, 'Left', (45, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            temp_left += 1
            temp_right = 0
            temp_font = 0
        else:
            cv2.putText(frame, 'Font', (45, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            temp_font += 1
            temp_right = 0
            temp_left = 0

        if temp_right == 2:
            right +=1

        if temp_left == 2:
            left += 1


    cv2.putText(frame, 'Left : {}'.format(str(left)), (45,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
    cv2.putText(frame, 'Right : {}'.format(str(right)), (45,95), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)


    cv2.imshow('Window', frame)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()