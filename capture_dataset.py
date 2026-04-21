import time
from pathlib import Path

import cv2
import numpy as np
from mss import mss


SAVE_DIR = Path("screenshots/raw")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CAPTURE_REGION = None
# Example:
# CAPTURE_REGION = {"top": 100, "left": 100, "width": 1280, "height": 720}

MONITOR_INDEX = 1
CAPTURE_EVERY_N_SECONDS = 0.5


def main():
    with mss() as sct:
        region = CAPTURE_REGION if CAPTURE_REGION else sct.monitors[MONITOR_INDEX]
        last_save = 0.0

        while True:
            shot = np.array(sct.grab(region))
            frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

            now = time.time()
            if now - last_save >= CAPTURE_EVERY_N_SECONDS:
                filename = SAVE_DIR / f"frame_{int(now * 1000)}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"Saved {filename}")
                last_save = now

            preview = frame.copy()
            cv2.putText(
                preview,
                "Capturing screenshots - press Q to quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Capture Dataset", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()