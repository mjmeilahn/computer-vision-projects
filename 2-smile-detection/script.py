import cv2

# LOAD UNIVERSAL PATTERNS
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# DETECT THE PATTERNS
def detect (gray, frame):
    # .detectMultiScale(colorScale, reduceImageBy, threshhold)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # .rectangle(window, vectors, selectedAreaWithinVector, borderColor, borderThickness)
        cv2.rectangle(frame, (x, y), (x+y, w+h), (255, 0, 0), 2)
        grayRegion = gray[y:y+h, x: x+w]
        faceArea = frame[y:y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(grayRegion, 1.1, 3)
        smile = smile_cascade.detectMultiScale(grayRegion, 1.7, 22)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(faceArea, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(faceArea, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    return frame


video_capture = cv2.VideoCapture(0)

# INFINITE LOOP UNTIL WEBCAM IS TURNED OFF MANUALLY
while True:
    # TURN THE WEBCAM ON
    _, frame = video_capture.read()

    # CONVERT WEBCAM VIDEO TO GRAYSCALE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DETECT PATTERN ON THE VIDEO
    canvas = detect(gray, frame)

    # PAINT ON THE VIDEO
    cv2.imshow('Video', canvas)

    # LISTEN TO KEYBOARD OR PRESS "Q" TO QUIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# TURN OFF THE WEBCAM
video_capture.release()

# REMOVE ALL WEBCAM WINDOWS
cv2.destroyAllWindows()