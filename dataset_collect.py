
"""This file using libraries"""
import cv2
import os

HARRCASECADE = "haarcascade_frontalface_default.xml"
CASECADE = cv2.CascadeClassifier(HARRCASECADE)
DATASET = 'dataset'
SUBDIR = input("Enter your face class name: ")
cam = cv2.VideoCapture(0)

PATH = os.path.join(DATASET, SUBDIR)
if not os.path.isdir(PATH):
    os.mkdir(PATH)
(width, height) = (130, 100)
LIMETS = 1

while LIMETS < 31:
    _, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = CASECADE.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (PATH,LIMETS), face_resize)
        LIMETS += 1
    cv2.imshow("Collecing Data", img)
    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()






