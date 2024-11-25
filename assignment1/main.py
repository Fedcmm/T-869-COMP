import cv2
import time
import sys

import numpy as np


def find_spots_cv2(frame):
    # Brightest spot
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, _, _, max_bright = cv2.minMaxLoc(gray)
    cv2.circle(frame, max_bright, 7, (255, 0, 0), 2)

    # Reddest spot
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.add(mask1, mask2)

    v_channel = hsv[:, :, 2]
    red_intensity = cv2.bitwise_and(v_channel, v_channel, mask=red_mask)
    _, _, _, max_red = cv2.minMaxLoc(red_intensity)
    cv2.circle(frame, max_red, 7, (0, 0, 255), 2)


def find_max_brightness_pixels(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    max_bright = (0, 0)
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            if gray[i, j] > gray[max_bright[0], max_bright[1]]:
                max_bright = (i, j)
    cv2.circle(frame, max_bright, 7, (255, 0, 0), 2)


def find_max_red_pixels(frame):
    red_ranges = [
        ((0, 100, 100), (10, 255, 255)),
        ((170, 100, 100), (180, 255, 255))
    ]
    max_red = 0
    max_red_loc = (0, 0)
    for i in range(0, frame.shape[0]):
        for j in range(0, frame.shape[1]):
            h, s, v = frame[i, j]
            in_lower_red = red_ranges[0][0][0] <= h <= red_ranges[0][1][0]
            in_upper_red = red_ranges[1][0][0] <= h <= red_ranges[1][1][0]
            if (in_lower_red or in_upper_red) and v > max_red:
                max_red = v
                max_red_loc = (i, j)
    cv2.circle(frame, max_red_loc, 7, (0, 0, 255), 2)


def main():
    while True:
        ts = time.time()
        ret, frame = cap.read()

        if method == 'frame':
            find_spots_cv2(frame)
        elif method == 'pixel':
            find_max_brightness_pixels(frame)
            find_max_red_pixels(frame)

        elapsed = time.time() - ts
        cv2.putText(frame, f"{1 / elapsed:.0f} FPS", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
        print(elapsed)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ['frame', 'pixel']:
        sys.exit('Usage: python main.py <frame/pixel> [device]')

    method = sys.argv[1]
    cap = cv2.VideoCapture(0 if len(sys.argv) < 3 else int(sys.argv[2]))
    main()

    cap.release()
    cv2.destroyAllWindows()