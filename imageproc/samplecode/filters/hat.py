import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")


imghat = cv2.imread('bunnyears.png', -1)


imghatGray = cv2.cvtColor(imghat, cv2.COLOR_BGR2GRAY)

ret, orig_mask = cv2.threshold(imghatGray, 0, 255, cv2.THRESH_BINARY) #thresholds the image fg is sig
orig_mask_inv = cv2.bitwise_not(orig_mask) #inversion of theshold replaces significance of bg is sig

# Convert hat image to BGR
# and save the original image size (used later when re-sizing the image)
imghat = imghat[:,:,0:3]
origHatHeight, origHatWidth = imghat.shape[:2]

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read() #frame is the camera feed from VideoCapture, ret is a boolean var for the status of feed

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converts the frame to b/w
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # checks for faces in the b/w image

    for (x, y, w, h) in faces:#for the xy coordinates of the start of the face, and the width and height of face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) #draws a rectangle on face

        hatWidth = w 
        hatHeight = int(hatWidth * origHatHeight / origHatWidth)# scale the hat width appropriately

        # Center the hat
        x1 = x - 15
        y1 = y - h
        x2 = x + w +15
        y2 = y

        cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2) #draws rectangle for hat

        # Check for clipping
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > video_capture.get(3):
            x2 = video_capture.get(3)
        if y2 > h:
            y2 = h

        # Re-calculate the width and height of the hat image
        hatWidth = x2 - x1
        hatHeight = y2 - y1

        roi_gray = gray[y-hatHeight:y, x-15:x+w+15]#region of interest where we want to superimpose the hat
        roi_color = frame[y-hatHeight:y, x-15:x+w+15]#region of interest but in color

        # Re-size the original image and the masks to the hat sizes
        # calcualted above
        hat = cv2.resize(imghat, (hatWidth,hatHeight), interpolation = cv2.INTER_AREA) #resized image of the hat
        mask = cv2.resize(orig_mask, (hatWidth,hatHeight), interpolation = cv2.INTER_AREA) #resizes thesholded mask (fg)
        mask_inv = cv2.resize(orig_mask_inv, (hatWidth,hatHeight), interpolation = cv2.INTER_AREA) #resizes inverted thesholded mask (bg)

        #PROBLEM
        #Need to do a bitwise and between the frame and mask_inv to recreate the unaffected background
        #Need to perform bitwise AND between the frame and mask to create the addition of the rabbit ears
        #Merge these 2 frames^
       

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
