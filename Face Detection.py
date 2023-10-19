import cv2

#Credit to https://www.datacamp.com/tutorial/face-detection-python-opencv

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
front_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

video = cv2.VideoCapture(0)

def create_face_box(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize = (40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

def create_front_box(image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fronts = front_classifier.detectMultiScale(gray_scale, 1.1, 5, minSize = (40, 40))
    for (x, y, w, h) in fronts:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return fronts

while True:
    result, vid = video.read()
    
    if result is False: break

    #creating boxes
    faces = create_face_box(vid)
    fronts = create_front_box(vid)
    
    cv2.imshow("Face Detection", vid)
    k = cv2.waitKey(1) & 0xFF
    #esc
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
