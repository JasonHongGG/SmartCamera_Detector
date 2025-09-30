import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from Core.FaceRecognition.FaceManager import faceMgr
from Manager.OCSortManager import OCSortManager
from Manager.FontManager import fontMgr

# 需要和 Font 資料夾放在一起
class FaceRecognition:
    def __init__(self, headless=False):
        self.headless = headless
        self.trackerMgr = OCSortManager()
        self.face_cache = {}  # {track_id: {"name": str, "embedding": np.array}} # 快取已識別的人臉
        
        self.frame_resize = 0.5    # 為了速度，把影格縮小 
        self.CurFilePath = os.path.dirname(os.path.abspath(__file__))
        
    def getCrop(self, x1, y1, x2, y2, frame, small_frame, frame_resize):
        # small
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > small_frame.shape[1]: x2 = small_frame.shape[1]
        if y2 > small_frame.shape[0]: y2 = small_frame.shape[0]
        
        small_crop = small_frame[y1:y2, x1:x2]
        
        # origin (座標需要還原到原始比例)
        orig_x1 = int(x1 / frame_resize)
        orig_y1 = int(y1 / frame_resize)
        orig_x2 = int(x2 / frame_resize)
        orig_y2 = int(y2 / frame_resize)
        
        if orig_x1 < 0: orig_x1 = 0
        if orig_y1 < 0: orig_y1 = 0
        if orig_x2 > frame.shape[1]: orig_x2 = frame.shape[1]
        if orig_y2 > frame.shape[0]: orig_y2 = frame.shape[0]
        
        crop = frame[orig_y1:orig_y2, orig_x1:orig_x2]
        
        if not self.headless:
            cv2.imshow("Crop", crop)
            cv2.imshow("Frame Crop", crop)
        
        return small_crop, crop

    def recognizeFaces(self, frame):
        small_frame = cv2.resize(frame, (0,0), fx=self.frame_resize, fy=self.frame_resize)

        # 人臉偵測
        faces = faceMgr.face_app.get(small_frame)  # 偵測 + 對齊 + 抽特徵
        bboxes = []
        scores = []
        for f in faces:
            bboxes.append(f.bbox)  # [x1, y1, x2, y2]
            scores.append(f.det_score)

        # 追蹤
        tracks = self.trackerMgr.objectTrack(small_frame, bboxes, scores)

        results = []
        for (x1, y1, x2, y2, track_id) in tracks:
            small_crop, crop = self.getCrop(int(x1), int(y1), int(x2), int(y2), frame, small_frame, self.frame_resize)
            name = faceMgr.recognizeFaces(small_crop, crop, track_id)

            # resize 回原大小
            results.append((
                int(x1 / self.frame_resize),
                int(y1 / self.frame_resize),
                int(x2 / self.frame_resize),
                int(y2 / self.frame_resize),
                track_id,
                name
            ))

        return results
    
    def draw(self, frame, face_info):
        # 先畫所有的方框和 ID
        for left, top, right, bottom, track_id, name in face_info:
            line_width = (right - left) // 3
            line_height = (bottom - top) // 3
            cv2.line(frame, (left, top), (left + line_width, top), (0, 255, 0), 2)
            cv2.line(frame, (left, bottom), (left + line_width, bottom), (0, 255, 0), 2)
            cv2.line(frame, (right, top), (right - line_width, top), (0, 255, 0), 2)
            cv2.line(frame, (right, bottom), (right - line_width, bottom), (0, 255, 0), 2)

            cv2.line(frame, (left, top), (left, top + line_height), (0, 255, 0), 2)
            cv2.line(frame, (right, top), (right, top + line_height), (0, 255, 0), 2)
            cv2.line(frame, (left, bottom), (left, bottom - line_height), (0, 255, 0), 2)
            cv2.line(frame, (right, bottom), (right, bottom - line_height), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (left + 2 , top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = fontMgr.cv2AddChineseText(frame, name, (left + 2 , top - 40))
        
        return frame

    def start(self, frame):
        # Recognize Faces
        face_info = self.recognizeFaces(frame)
        frame = self.draw(frame, face_info)

        # show results
        if not self.headless:
            cv2.imshow("Face Recognition", frame)

        return frame, face_info

