
"""This file using libraries"""
import cv2
import numpy
import os

CASECADE = "haarcascade_frontalface_default.xml"
CADE = cv2.CascadeClassifier(CASECADE)

Cam = cv2.VideoCapture(0)
datasets = "dataset"
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdirs in dirs:
        names[id] = subdirs
        subjectspath = os.path.join(datasets, subdirs)
        for files in os.listdir(subjectspath):
            path = subjectspath + '/' + files
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(images, labels) = [numpy.array(lis) for lis in (images, labels)]
print(images, labels)

(width, height) = (130, 100)

model = cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)

CNT = 1

while True:
    _, img = Cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = CADE.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        if prediction[1] < 800:
            cv2.putText(img, '%s - %.0f' % (names[prediction[0]], prediction[1]),
                        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            print(names[prediction[0]])
            CNT += 0
        else:
            CNT += 1
            cv2.putText(img, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            if (CNT>100):
                cv2.imwrite("unknown.jpg", img)
                CNT = 0


    cv2.imshow("Collecing Data", img)
    if cv2.waitKey(1) == 27:
        break


Cam.release()
cv2.destroyAllWindows()
