import os
import cv2

class Capture:

    @staticmethod
    def SaveImage(frame, filename="../Cache/screenshot.png"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, filename)
        file_path = os.path.normpath(file_path)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        cv2.imwrite(file_path, small_frame)
        print(f"Image saved as {file_path}")
        