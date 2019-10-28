import cv2
import numpy as np
import os
import threading
import time


def showDetails(frame):
    timeout = time.time() + 30   # 30 seconds from now
    while(True):
        cv2.rectangle(frame, (30, 30), (300, 200), (0, 255, 0), 5) 
        if  time.time() > timeout:
            break




MIN_MATCHES = 100
model = cv2.imread('a.jpg', 0)

cap = cv2.VideoCapture(0)

# ORB keypoint detector
orb = cv2.ORB_create()              
# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
# Compute model keypoints and its descriptors
kp_model, des_model = orb.detectAndCompute(model, None) 

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Compute scene keypoints and its descriptors
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    # Match frame descriptors with model descriptors
    matches = bf.match(des_model, des_frame)
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > MIN_MATCHES:
        #print(len(matches))

        #===============================================================================================
        # assuming matches stores the matches found and 
        # returned by bf.match(des_model, des_frame)
        # differenciate between source points and destination points
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # compute Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


        # Draw a rectangle that marks the found model in the frame
        h, w = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, M)  
        # connect them with lines
        img2 = cv2.polylines(frame, [np.int32(dst)], True, (255,255,255), 3, cv2.LINE_AA) 

        #draw matching points
        # cap2 = cv2.drawMatches(model, kp_model, img2, kp_frame,
        #                     matches[:len(matches)], 0, flags=2)


        # Creating rectangle 
        # cv2.rectangle(frame, (30, 30), (300, 200), (0, 255, 0), 5) 
        x = threading.Thread(target=showDetails, args=(frame,))
        # x.start()

        cv2.imshow('frame', img2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        cv2.imshow('frame', frame)
        # print ("Not enough matches have been found - " + str(len(matches)/MIN_MATCHES))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows() 
