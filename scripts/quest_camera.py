import cv2
import numpy as np
# Open camera 0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
# Get the default video size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('output.mp4', fourcc, 30, (1920, 320))

# Start capturing and processing frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is not available, break the loop
    if not ret:
        break
    # Write the frame to the output video filef
    
    
    # # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_frame = np.concatenate([frame[:, :320].transpose(1, 0, 2)[::-1, ], frame[:, 320:640].transpose(1, 0, 2)[::-1, ], frame[:, 640:960].transpose(1, 0, 2)[::-1, ], frame[:, 960:].transpose(1, 0, 2)[::-1, ]], axis = 1)
    
    # # out.write(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR))
    # out.write(gray_frame)
    
    # Display the resulting frame
    cv2.imshow('frame', frame[..., 1])
    
    # Wait for 1 millisecond for user to press 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and output objects, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()