import numpy as np
import imutils
import cv2

from collections import deque

BUFFER_LEN = 32

ball_in_play = False

ball_lower = (92, 151, 36)
ball_upper = (176, 255, 255)

field_lower = (0, 209, 44)
field_upper = (255, 255, 255)

cap = cv2.VideoCapture(0)
pts = deque(maxlen=BUFFER_LEN)

while True:
    ret, frame = cap.read()
    ratio = frame.shape[0] / float(frame.shape[0])

    if ret == True:
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

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

        # field mask
        field_mask = cv2.inRange(hsv, field_lower, field_upper)
        field_mask = cv2.erode(field_mask, None, iterations=2)
        field_mask = cv2.dilate(field_mask, None, iterations=2)

        field_cnts = cv2.findContours(field_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        field_cnts = imutils.grab_contours(field_cnts)

        for c in field_cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            if len(approx) == 4:
                c = c.astype('float')
                c *= ratio
                c = c.astype('int')
                cv2.drawContours(frame, [c], -1, (255, 255, 0), 2)

    cv2.imshow('frame', frame)

    print(ball_in_play)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
