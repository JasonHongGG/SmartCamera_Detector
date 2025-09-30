
from polars import Enum
import cv2
from Core.MotionDetector.MotionDetector import MotionDetector
from Manager.YoloManager import YoloManager
from Manager.OCSortManager import OCSortManager
from Manager.FontManager import fontMgr
from Core.FaceRecognition.FaceManager import faceMgr

class State(Enum):
    MOTION_DETECTED = 0
    PERSON_DETECTED = 1

class MotionTriggeredRecognition:
    def __init__(self, headless=False):
        self.headless = headless
        self.state = State.MOTION_DETECTED
        self.motion_detector = MotionDetector(self.headless)
        self.objectDetectionMgr = YoloManager("yolo11m.pt")
        self.trackerMgr = OCSortManager()

        self.motion_flag = False
        self.person_flag = False
        self.face_flag = False

        self.cache = {}  # {track_id: {"name": str}} # 快取已識別的人臉
        
    def detect(self, frame):
        info = []
        self.motion_flag = False
        self.person_flag = False
        self.face_flag = False

        # Motion Detection
        if self.state == State.MOTION_DETECTED:
            motion_detected, motion_frame, thresh = self.motion_detector.start(frame.copy())

            if motion_detected:
                self.state = State.PERSON_DETECTED

        # 有動靜後開始做人物偵測
        elif self.state == State.PERSON_DETECTED:
            self.motion_flag = True
            
            bboxes, class_ids, scores = self.objectDetectionMgr.objectDetect(frame.copy())

            # 有偵測到人物則持續做追蹤
            if bboxes:
                self.person_flag = True
                tracks = self.trackerMgr.start(frame.copy(), bboxes, scores)

                # 有偵測到人則進入人臉辨識
                name = "Unknown"
                for (x1, y1, x2, y2, track_id) in tracks:
                    crop = frame[y1:y2, x1:x2]
                    small_crop = cv2.resize(crop, (0,0), fx=0.5, fy=0.5)
                    face = faceMgr.face_app.get(crop)
                    if face:
                        self.face_flag = True
                        name = faceMgr.recognizeFaces(small_crop, crop, track_id)
                
                    info.append({
                        "track_id": track_id,
                        "name": self.cache.get(track_id, {"name": name})["name"],  # 如果已經追中到就用快取的
                        "bbox": (x1, y1, x2, y2)
                    })

                    # 有辨識出且未在 cache
                    if track_id not in self.cache and name != "Unknown" and "學習中" not in name:
                        self.cache[track_id] = {"name": name}
                    
                    
            # 沒有偵測到則回到 Motion Detection
            else:
                self.state = State.MOTION_DETECTED

        return info
    
    def draw(self, frame, info):
        cv2.putText(frame, f"Motion Flag: {self.motion_flag}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Person Flag: {self.person_flag}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Face Flag: {self.face_flag}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for person in info:
            x1, y1, x2, y2 = person["bbox"]
            track_id = person["track_id"]
            name = person["name"]
            label = f"ID:{track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1 + 2 , y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = fontMgr.cv2AddChineseText(frame, name, (x1 + 2 , y1 - 40))
       
        return frame
    
    def start(self, frame):
        info = self.detect(frame.copy())
        frame = self.draw(frame, info)

        if not self.headless:
            cv2.imshow("MotionTriggeredRecognition", frame)

        return frame, info