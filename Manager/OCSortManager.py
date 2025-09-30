import cv2
import numpy as np
from .OCSortTracker.ocsort import OCSort

class OCSortManager:
    def __init__(self):
        self.tracker = OCSort(det_thresh=0.45, iou_threshold=0.3)
        self.track_paths = {}  # {track_id: [points]}

    def objectTrack(self, frame, bboxes, scores):
        results = []
        if len(bboxes) > 0:
            # Convert bboxes and scores to numpy array
            detections = []
            for bbox, score in zip(bboxes, scores):
                x1, y1, x2, y2 = bbox
                if hasattr(score, "cpu"):
                    score = float(score.cpu().numpy())
                else:
                    score = float(score)
                detections.append([x1, y1, x2, y2, score])

            detections = np.array(detections)

            # update tracker
            height, width = frame.shape[:2]
            img_info = (height, width, 0)
            img_size = (height, width)
            tracks = self.tracker.update(detections, img_info, img_size)
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track)
    
                # 更新路徑
                if track_id not in self.track_paths:
                    self.track_paths[track_id] = []
                self.track_paths[track_id].append((x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2))

                # 只保留最近 100 個點
                if len(self.track_paths[track_id]) > 100:
                    self.track_paths[track_id] = self.track_paths[track_id][-100:]
                    
                results.append((x1, y1, x2, y2, track_id))
        return results
    
    def draw(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            label = f"ID: {track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2), radius=2, color=(0, 0, 255), thickness=-1)
            if track_id in self.track_paths and len(self.track_paths[track_id]) > 1:
                pts = np.array(self.track_paths[track_id], dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=False, color=(0,255,255), thickness=2)
            cv2.putText(frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    def start(self, frame, bboxes, scores):
        tracks = self.objectTrack(frame, bboxes, scores)
        self.draw(frame, tracks)
        return tracks