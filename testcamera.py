import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)
for _ in range(30):
    cap.read()

ret, frame = cap.read()
print("ret:", ret)
print("frame shape:", frame.shape if ret else "None")
cap.release()