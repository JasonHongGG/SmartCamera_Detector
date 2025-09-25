import cv2
from Manager.YoloManager import YoloManager
from Manager.OCSortManager import OCSortManager
from Manager.CrossLineManager import CrossLineManager

class MotionTracker:
    def __init__(self, headless=False):
        self.headless = headless
        self.objectDetectionMgr = YoloManager("yolo11m.pt")
        self.trackerMgr = OCSortManager()
        self.crossLineMgr = CrossLineManager(cv_window_name = "Camera", headless = self.headless)

    def start(self, frame):
        # Object Detection
        bboxes, class_ids, scores = self.objectDetectionMgr.objectDetect(frame)
        self.objectDetectionMgr.draw(frame, bboxes, class_ids, scores)

        # Object Tracking
        tracks = self.trackerMgr.objectTrack(frame, bboxes, scores)
        self.trackerMgr.draw(frame, tracks)

        # Cross Line Detection
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            if(self.crossLineMgr.is_cross_line(center, track_id)):
                print(f"{track_id} : Cross Line Detected! {center}")
        self.crossLineMgr.draw_door_line(frame)

        # show results
        if not self.headless:
            cv2.imshow("Motion Tracking", frame)
            
        return frame
        
        
        
