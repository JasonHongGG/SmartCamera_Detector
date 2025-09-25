import os
import threading
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import insightface
import faiss

from Manager.OCSortManager import OCSortManager
from Manager.LineAlarmManager import LineAlarmManager
from Core.FaceRecognition.FaceSelfLearning import FaceSelfLearning

class FaceRecognition:
    def __init__(self, headless=False):
        self.headless = headless
        self.trackerMgr = OCSortManager()
        self.face_app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))  # ctx_id=0: GPU, -1: CPU
        self.faiss_index = faiss.IndexFlatL2(512)  # L2 距離度量
        self.face_cache = {}  # {track_id: {"name": str, "embedding": np.array}} # 快取已識別的人臉
        
        self.known_encodings = []
        self.known_names = []
        self.faceRecognition_threshold = 1.1        # 越小越嚴格 (0.-0.6 常見)
        self.frame_resize = 0.25    # 為了速度，把影格縮小
        self.CurFilePath = os.path.dirname(os.path.abspath(__file__))
        self.fontSize = 25
        self.font = ImageFont.truetype(os.path.join(self.CurFilePath, "Font/NotoSansTC-Medium.ttf"), self.fontSize, encoding="utf-8")
        
        # 自我學習機制相關變數
        self.faceSelfLearning = FaceSelfLearning(self.CurFilePath, self.frame_resize)
        self.learning_threshold = 2.0  # 最大距離閾值
        
        self.known_dir = "KnownFaces"
        self.loadKnownFaces(self.known_dir)
        

    def loadKnownFaces(self, known_dir):
        self.known_embeddings = []
        self.known_names = []

        known_path = Path(os.path.join(self.CurFilePath, known_dir)).resolve()
        print(f"載入已知人臉資料夾: {known_path}")

        for person_dir in known_path.iterdir():
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for img_path in person_dir.glob("*"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"[錯誤] 無法讀取圖片: {img_path}")
                        continue
                    faces = self.face_app.get(img)
                    if len(faces) != 1:
                        print(f"[警告] {img_path} 找到 {len(faces)} 張臉，跳過")
                        continue
                    embedding = faces[0].normed_embedding.astype(np.float32)
                    self.known_embeddings.append(embedding)
                    self.known_names.append(name)
                    print(f"已載入 {img_path} -> {name}")
                except Exception as e:
                    print(f"[錯誤] 讀取 {img_path} 時發生問題：{e}")

        # 建立 FAISS 索引
        if len(self.known_embeddings) > 0:
            self.faiss_index = faiss.IndexFlatL2(512)  # 重新創建
            self.faiss_index.add(np.array(self.known_embeddings, dtype=np.float32))
            print(f"[完成] 已建立 FAISS 索引，共 {len(self.known_embeddings)} 個人臉")

    def recognizeFaces(self, frame):
        small_frame = cv2.resize(frame, (0,0), fx=self.frame_resize, fy=self.frame_resize)

        # 人臉偵測
        faces = self.face_app.get(small_frame)  # 偵測 + 對齊 + 抽特徵
        bboxes = []
        scores = []
        for f in faces:
            bboxes.append(f.bbox)  # [x1, y1, x2, y2]
            scores.append(f.det_score)

        # 追蹤
        tracks = self.trackerMgr.objectTrack(small_frame, bboxes, scores)

        results = []
        for (x1, y1, x2, y2, track_id) in tracks:
            # 如果沒有 cache 或是 name 為 Unknown，才進行比對
            if track_id not in self.face_cache or self.face_cache[track_id]["name"] == "Unknown":
                # 避免超過邊界
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 > small_frame.shape[1]: x2 = small_frame.shape[1]
                if y2 > small_frame.shape[0]: y2 = small_frame.shape[0]
                
                # 裁切人臉
                crop = small_frame[y1:y2, x1:x2]
                if not self.headless:
                    cv2.imshow("Crop", crop)
                
                # 抽特徵 & 比對
                face = self.face_app.get(crop)
                if len(face) > 0:
                    emb = face[0].normed_embedding.astype('float32')
                    name = "Unknown"
                    if len(self.known_embeddings) > 0:
                        D, I = self.faiss_index.search(np.expand_dims(emb, axis=0), k=1)      # D[0][0] : best_distance & I[0][0] : best_index
                        
                        if I[0][0] < len(self.known_names):
                            print(f"Track ID {track_id} best match: {self.known_names[I[0][0]]} with distance {D[0][0]:.4f}")
                            if D[0][0] < self.faceRecognition_threshold:  # 1.1 閾值要自己調，L2距離越小越像
                                name = self.known_names[I[0][0]]
                                threading.Thread(target=LineAlarmManager.triggerAlarm, args=(frame.copy(), name)).start() # alarm 傳遞的是全新的 frame，避免被之後的繪畫影響
                            elif D[0][0] < self.learning_threshold:  # 2.0 在學習範圍內
                                candidate_name = self.known_names[I[0][0]]
                                self.faceSelfLearning.learning(self, self.known_dir, track_id, candidate_name, D[0][0], emb, crop)
                                
                        else:
                            print(f"[錯誤] 無效的索引 {I[0][0]}，超出已知人臉數量 {len(self.known_names)}。正在重建 FAISS 索引...")

                    self.face_cache[track_id] = {"name": name, "embedding": emb}
                
            name = self.face_cache.get(track_id, {"name": "Unknown"})["name"]
            
            # 處理已知為 Unknown 但有 cache 的情況（可能正在學習中）
            if name == "Unknown" and track_id in self.face_cache:
                if self.faceSelfLearning.isLearning(track_id):
                    candidate_name = self.faceSelfLearning.getLearningFaceName(track_id)
                    name = f"學習中-{candidate_name}"

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
    
    def cv2AddChineseText(self, img, text, position, textColor=(0,255,0), textSize=30):
        if(isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img)
        fontStyle = self.font
        if textSize != self.fontSize:
            fontStyle = ImageFont.truetype(os.path.join(self.CurFilePath, "Font/NotoSansTC-Medium.ttf"), textSize, encoding="utf-8")
        draw.text(position, text, textColor, font=fontStyle)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    
    def drawFaces(self, frame, face_info):
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
            frame = self.cv2AddChineseText(frame, name, (left + 2 , top - 40))
        
        return frame

    def start(self, frame):
        # Recognize Faces
        face_info = self.recognizeFaces(frame)
        frame = self.drawFaces(frame, face_info)

        # show results
        if not self.headless:
            cv2.imshow("Face Recognition", frame)

        return frame, face_info

