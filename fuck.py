import numpy as np
import imutils
import cv2
import itertools

from collections import deque

BUFFER_LEN = 32

ball_in_play = False

ball_lower = (92, 151, 36)
ball_upper = (176, 255, 255)

goals_lower = (13, 100, 158)
goals_upper = (47, 255, 255)

cap = cv2.VideoCapture(0)
pts = deque(maxlen=BUFFER_LEN)

while(True):
    ret, frame = cap.read()
    ratio = frame.shape[0] / float(frame.shape[0])

    if ret == True:
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
            permutations = itertools.permutations(goals_cnts, 2)
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

            for goal in distances[:1]:
                x1, y1 = goal[0]
                x2, y2 = goal[1]

                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=3, lineType=8)

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
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            if len(pts) >= 2:
                dX = pts[1][0] - pts[0][0]
                if dX < 0:
                    print('right')
                elif dX > 0:
                    print('left')
                else:
                    print('stationary')

            if radius > 3:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 3)

                pts.appendleft(center)

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue

                thickness = int(np.sqrt(BUFFER_LEN / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        else:
            ball_in_play = False

    cv2.imshow('frame', frame)

    print(ball_in_play)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
