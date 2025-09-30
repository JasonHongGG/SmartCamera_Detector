import cv2
import os
from ultralytics import YOLO

class YoloManager:
    def __init__(self, model_name = "yolo11m.pt"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "Models", model_name)
        model_path = os.path.normpath(model_path) # 先處理 .. 和 . 等相對路徑
        self.model = YOLO(model_path)

    def objectDetect(self, frame):
        bboxes, class_ids, scores = [], [], []
        results = self.model(frame, verbose=False)[0]
        for result in results.boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = result.conf[0]
            cls = int(result.cls[0]) # self.objectDetectionModel.names[cls] 取 name
            if conf > 0.5:  # 只顯示置信度大於 0.5 的物體
                bboxes.append((x1, y1, x2, y2))
                class_ids.append(cls)
                scores.append(conf)

        return bboxes, class_ids, scores
    
    def draw(self, frame, bboxes, class_ids, scores):
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            x1, y1, x2, y2 = bbox
            label = f"{self.model.names[class_id]}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)