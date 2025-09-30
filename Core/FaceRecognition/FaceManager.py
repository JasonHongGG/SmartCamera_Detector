import os
import threading
import cv2
import numpy as np
from pathlib import Path
import insightface
import faiss
from PIL import ImageFont

from Manager.LineAlarmManager import LineAlarmManager
from Core.FaceRecognition.FaceSelfLearning import FaceSelfLearning


# 需要和 KnownFaces 資料夾放在一起 (更改 self.known_dir)
class FaceManager:
    def __init__(self):
        self.known_dir = "KnownFaces"
        self.CurFilePath = os.path.dirname(os.path.abspath(__file__))
        self.known_path = Path(os.path.join(self.CurFilePath, self.known_dir)).resolve()

        self.face_app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))  # ctx_id=0: GPU, -1: CPU
        self.faiss_index = faiss.IndexFlatL2(512)  # L2 距離度量
        self.known_embeddings = []
        self.known_names = []
        self.faceRecognition_threshold = 1.1        # 越小越嚴格 (0.-0.6 常見)
        self.face_cache = {}  # {track_id: {"name": str, "embedding": np.array}} # 快取已識別的人臉
        self.loadKnownFaces()

        # 自我學習機制相關變數 (只有先做追中才能使用自我學習)
        self.faceSelfLearning = FaceSelfLearning(self.known_path)
        self.learning_threshold = 2.0  # 最大距離閾值

    def loadKnownFaces(self):
        print(f"載入已知人臉資料夾: {self.known_path}")
        self.known_embeddings = []
        self.known_names = []
        for person_dir in self.known_path.iterdir():
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for img_path in person_dir.glob("*"):
                try:
                    # img = cv2.imread(str(img_path))
                    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
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


    def compareFaces(self, small_crop, crop, track_id):
        # small_crop 是做比較的 (為了加速 resize 過)， crop 是做學習的
        face = self.face_app.get(small_crop)
        if len(face) > 0:
            emb = face[0].normed_embedding.astype('float32')
            name = "Unknown"
            if len(self.known_embeddings) > 0:
                D, I = self.faiss_index.search(np.expand_dims(emb, axis=0), k=1)      # D[0][0] : best_distance & I[0][0] : best_index
                
                if I[0][0] < len(self.known_names):
                    print(f"Track ID {track_id} best match: {self.known_names[I[0][0]]} with distance {D[0][0]:.4f}")
                    if D[0][0] < self.faceRecognition_threshold:  # 1.1 閾值要自己調，L2距離越小越像
                        name = self.known_names[I[0][0]]
                        threading.Thread(target=LineAlarmManager.triggerAlarm, args=(crop.copy(), name)).start() # alarm 傳遞的是全新的 frame，避免被之後的繪畫影響
                    elif D[0][0] < self.learning_threshold:  # 2.0 在學習範圍內
                        candidate_name = self.known_names[I[0][0]]
                        self.faceSelfLearning.learning(self, self.known_dir, track_id, candidate_name, D[0][0], crop)
                        
                else:
                    print(f"[錯誤] 無效的索引 {I[0][0]}，超出已知人臉數量 {len(self.known_names)}。正在重建 FAISS 索引...")

            self.face_cache[track_id] = {"name": name, "embedding": emb}
            return self.face_cache[track_id]
        

    def recognizeFaces(self, small_crop, crop, track_id):
        # 如果沒有 cache 或是 name 為 Unknown，才進行比對
        if track_id not in self.face_cache or self.face_cache[track_id]["name"] == "Unknown":
            
            # 抽特徵 & 比對
            self.compareFaces(small_crop, crop, track_id)
            
        name = self.face_cache.get(track_id, {"name": "Unknown"})["name"]
        
        # 處理已知為 Unknown 但有 cache 的情況（可能正在學習中）
        if name == "Unknown" and track_id in self.face_cache:
            if self.faceSelfLearning.isLearning(track_id):
                candidate_name = self.faceSelfLearning.getLearningFaceName(track_id)
                name = f"學習中-{candidate_name}"

        return name
    
faceMgr = FaceManager() 
