import cv2
import time

import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

cap = cv2.VideoCapture(0)

while True:
    ts = time.time()
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(frame, 100, 200, L2gradient=True)

    points = np.column_stack(np.where(edges > 0))

    x = points[:, 1].reshape(-1, 1)
    y = points[:, 0]

    ransac = RANSACRegressor(LinearRegression(), residual_threshold=20, max_trials=100)
    ransac.fit(x, y)

    line_x = np.array([x.min(), x.max()]).reshape(-1, 1)
    line_y = ransac.predict(line_x)

    pt1 = (int(line_x[0][0]), int(line_y[0]))
    pt2 = (int(line_x[1][0]), int(line_y[1]))
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    elapsed = time.time() - ts
    cv2.putText(frame, f"{1 / elapsed:.0f} FPS", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    print(elapsed)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()