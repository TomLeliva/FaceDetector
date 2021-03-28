import cv2
from random import randrange

# load some pre-trained data of face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
# img=cv2.imread('my_face2.PNG')

# To capture video from webcam
webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()
    # Convert to grayscale
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(frame)
    # Draw rectangles around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(0,256), randrange(0,256), randrange(0,256)), 2)
    # Show image
    cv2.imshow('Face Detector', frame)
    # Wait with execusion
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key==81 or key==113:
        break

# release the VideoCapture object
webcam.release()

print("Code completed")