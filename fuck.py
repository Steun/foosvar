from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import imutils
import time
import cv2
import itertools

from collections import deque

BUFFER_LEN = 32

ball = None
ball_dX = 0
goal_left = None
goal_right = None

ball_lower = (62, 30, 94)
ball_upper = (106, 255, 255)

goals_lower = (0, 94, 110)
goals_upper = (5, 236, 255)

pts = deque(maxlen=BUFFER_LEN)

camera = PiCamera()
camera.resolution = (1296, 976)
camera.framerate = 70
camera.iso = 300
camera.shutter_speed = camera.exposure_speed
#raw_capture = PiRGBArray(camera, size=(640, 480))
raw_capture = PiRGBArray(camera)

time.sleep(0.1)

for output in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
    frame = output.array
    ratio = frame.shape[0] / float(frame.shape[0])

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # goal markers mask
    goals_mask = cv2.inRange(hsv, goals_lower, goals_upper)
    goals_mark = cv2.erode(goals_mask, None, iterations=2)
    goals_mark = cv2.dilate(goals_mask, None, iterations=2)

    goals_cnts = cv2.findContours(goals_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    goals_cnts = imutils.grab_contours(goals_cnts)
    center = None

    if len(goals_cnts) > 0:
        permutations = []
        for i, _ in enumerate(goals_cnts):
            for j in range(i+1, len(goals_cnts)):
                permutations.append((goals_cnts[i], goals_cnts[j]))

        distances = []
        for a, b in permutations:
            M1 = cv2.moments(a)
            M2 = cv2.moments(b)
            x1 = M1['m10'] / (M1['m00'] + 1e-10)
            y1 = M1['m01'] / (M1['m00'] + 1e-10)
            x2 = M2['m10'] / (M2['m00'] + 1e-10)
            y2 = M2['m01'] / (M2['m00'] + 1e-10)

            distance = (x1 - x2)**2 + (y1 - y2)**2
            distances.append(((x1, y1), (x2, y2), distance))

        distances.sort(key=lambda x: x[-1])
        distances = list(dict.fromkeys(distances))

        for goal in distances[:2]:
            x1, y1 = goal[0]
            x2, y2 = goal[1]

            if x1 < frame.shape[0]/2 + 20 and x2 < frame.shape[0]/2 + 20:
                goal_left = goal
                color = (0, 255, 0)
            else:
                goal_right = goal
                color = (255, 0, 255)


            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=3, lineType=8)

    # ball mask
    ball_mask = cv2.inRange(hsv, ball_lower, ball_upper)
    ball_mask = cv2.erode(ball_mask, None, iterations=2)
    ball_mask = cv2.dilate(ball_mask, None, iterations=2)

    ball_cnts = cv2.findContours(ball_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_cnts = imutils.grab_contours(ball_cnts)
    center = None

    if len(ball_cnts) > 0:
        ball_in_play = True

        c = max(ball_cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        ball = M
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        if len(pts) >= 2:
            ball_dX = pts[1][0] - pts[0][0]

        if radius > 1:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 3)

            pts.appendleft(center)

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue

            thickness = int(np.sqrt(BUFFER_LEN / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    else:
        ball = None

    cv2.imshow('frame', frame)

    if ball is not None:
        ball_x = ball['m10'] / (ball['m00'] + 1e-10)
        ball_y = ball['m01'] / (ball['m00'] + 1e-10)

        if goal_left is not None:
            if ball_x <= goal_left[0][0]:
                print('goal left'*5)

        if goal_right is not None:
            if ball_x >= goal_right[0][0]:
                print('goal right'*5)

    raw_capture.truncate(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
