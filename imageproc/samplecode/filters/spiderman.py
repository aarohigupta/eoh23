# import the opencv library
import cv2
# define a video capture object
vid = cv2.VideoCapture(0)

mode = 'def'
while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = frame[0:768, 310:1078]
    # cropping frame

    # Display the resulting frame
    if mode == 'def':
        cv2.imshow('frame', spiderman(frame))

    # the 'q' button is set as the quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()

def spiderman(frame):
    
    return frame