import cv2
import os
import time
from Core.MotionDetector.MotionDetector import MotionDetector
from Core.MotionTracker.MotionTracker import MotionTracker
from Core.FaceRecognition.FaceRecognition import FaceRecognition
from Manager.KeyboardManager import KeyboardManager, KeyboardLayoutCode
from Manager.HttpManager import HttpManager, httpManager
from Manager.CrossLineManager import CrossLineManager
from Utils.Capture import Capture
from dotenv import load_dotenv


class CarmeraProcessor:
    def __init__(self, camera_index=0, width=640, height=480, headless=False):
        self.headless = headless
        self.frame_size = (width, height)
        self.flip_frame = os.getenv("FLIP_FRAME", "false").lower() in ("true", "1", "yes", "on")
        self.frame_times = []
        self.cap = cv2.VideoCapture(camera_index)
        self.setup_camera_properties()
        
        self.motion_detector = MotionDetector(self.headless)
        self.motion_tracker = MotionTracker(self.headless)
        self.face_recognizer = FaceRecognition(self.headless)
        self.crossLineMgr = CrossLineManager(cv_window_name = "Face Recognition", headless = self.headless)
        
        # http parameter
        self.motion_enable = False
        self.face_enable = True
        self.crossLine_enable = False

        # camera reconnection
        self.consecutive_failures = 0
        self.max_failures = 10

    def setup_camera_properties(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        if self.headless:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)

    def get_fps(self):
        now = time.time()
        self.frame_times.append(now)
        # 只保留最近 1 秒的時間戳
        self.frame_times = [t for t in self.frame_times if now - t < 1.0]
        return len(self.frame_times)

    def Process(self, frame):
        try:
            if self.motion_enable:
                motion_detected, motion_frame, thresh = self.motion_detector.detect(frame.copy())
                httpManager.update_frame('motion', motion_frame)
                httpManager.update_motion_info(motion_detected)

            if self.face_enable or self.crossLine_enable:
                face_frame, face_info = self.face_recognizer.start(frame.copy())
                httpManager.update_frame('face', face_frame)
                httpManager.update_face_info(face_info)

                if self.crossLine_enable:
                    #Cross Line Detection
                    crossline_frame = frame.copy()
                    for (x1, y1, x2, y2, track_id, name) in face_info:
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        cv2.circle(crossline_frame, center, 5, (255, 0, 0), -1)
                        if(self.crossLineMgr.is_cross_line(center, track_id)):
                            print(f"{track_id} {name}: Cross Line Detected! {center}")
                            httpManager.update_crossline_info(f"{name}")
                    self.crossLineMgr.draw_line(crossline_frame)
                    httpManager.update_frame('crossline', crossline_frame)
                    
            # tracker_frame = self.motion_tracker.start(frame.copy())
            httpManager.update_frame('current', frame)
            cv2.putText(frame, f"FPS: {self.get_fps()}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if not self.headless: cv2.imshow("Camera", frame)
        except Exception as e:
            print(f"Process frame error: {e}")

    def keyHandler(self, frame):
        if self.headless:
            return True  # 無頭模式下不處理鍵盤事件
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): 
            KeyboardManager.changeKeyboardLayout(KeyboardLayoutCode.CHINESE.value)
            return False
        if key == ord("s"):
            Capture.SaveImage(frame)
        KeyboardManager.changeKeyboardLayout(KeyboardLayoutCode.ENGLISH.value)
        return True

    def reconnectionChecker(self, ret):
        if not ret:
            self.consecutive_failures += 1
            print(f"讀取幀失敗 ({self.consecutive_failures}/{self.max_failures})")
            
            if self.consecutive_failures >= self.max_failures:
                print("連續讀取失敗過多，嘗試重新連接...")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(os.environ.get("CAMERA_INDEX", 0))
                self._setup_camera_properties()
                self.consecutive_failures = 0
                return True
        else:
            self.consecutive_failures = 0
            
        return False

    def start(self):
        while True:
            try:
                ret, frame = self.cap.read()
                if self.reconnectionChecker(ret):
                    continue

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if not self.keyHandler(frame):
                    break

                self.Process(frame)

            except Exception as e:
                print(f"攝影機讀取錯誤: {e}")
                self.consecutive_failures += 1
                time.sleep(0.1)  # 短暫等待後重試
                continue
            
        self.cap.release()
        if not self.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    load_dotenv()
    processor = CarmeraProcessor(camera_index=os.environ.get("CAMERA_INDEX", 0))
    processor.start()