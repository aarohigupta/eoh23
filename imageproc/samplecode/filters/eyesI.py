import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# Capture video from the webcam
capture = cv2.VideoCapture(0)

# Create cascades
face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# PNG
illiniI = cv2.imread('illiniI.png', -1)

while 1:
    # Get the current frame
    ret, img = capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each face
    for(x,y,w,h) in faces:
        # Crop the face
        face = img[y:y+h, x:x+w]

        # Resize
        illiniI_resized = cv2.resize(illiniI, (w, h))

        # Foreground and background layer
        illiniI_fg = illiniI_resized[:,:,:3]
        illiniI_bg = cv2.bitwise_not(illiniI_resized[:,:,3])

        # Convert to FP rep
        face_float = face.astype(float)
        illiniI_fg_float = illiniI_fg.astype(float)

        # Normalize layers
        face_normalized = face_float / 255.0
        illiniI_fg_normalized = illiniI_fg_float / 255.0

        # Multiply by (1 - alpha)
        face_filtered = cv2.multiply(face_normalized, 1.0 - illiniI_fg_normalized)

        # Add layers
        face_filtered = (face_filtered * 255).astype(np.uint8)

        # Overlay onto original frame
        img[y:y+h, x:x+w] = face_filtered


        # Currently not working for Individual eyes, uncomment code
        # Breaking down face
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]

        # Individual Eyes
        # # Find eyes
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     # Crop the eyes
        #     eye = img[ey:ey+eh, ex:ex+eh]

        #     # Resize illini image to size of the eyes
        #     illiniI_resized = cv2.resize(illiniI, (ew, eh))


        #     #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #     # FG and BG layers
        #     illiniI_fg = illiniI_resized[:,:,:3]
        #     illiniI_bg = cv2.bitwise_not(illiniI_resized[:,:,3])

        #     # Convert eye and Illini layers to floating representation
        #     eye_float = eye.astype(float)
        #     illiniI_fg_float = illiniI_fg.astype(float)

        #     # Normalize layers
        #     eye_normalized = eye_float / 255.0
        #     illiniI_fg_normalized = illiniI_fg_float / 255.0

        #     # Mutiply normalized eye layer by (1 - alpha)
        #     eye_filtered = cv2.multiply(eye_normalized, 1.0 - illiniI_fg_normalized)

        #     # Add filtered eye and Illini layers
        #     eye_filtered = cv2.add(eye_filtered, illiniI_fg_normalized)

        #     # Convert the eye to 8-bit representation
        #     eye_filtered = (eye_filtered * 255).astype(np.uint8)

        #     #Overlay the filtered eyes on the original img
        #     img[y+ey:y+ey+eh, x+ex:x+ex+ew] = eye_filtered

    #Display the resulting image
    cv2.imshow('img',img)

    # Check if the user pressed the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the capture
capture.release()
cv2.destroyAllWindows()