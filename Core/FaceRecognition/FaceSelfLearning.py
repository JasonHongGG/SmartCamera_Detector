import datetime
import os
import cv2
import numpy as np
from pathlib import Path

class FaceSelfLearning:
    def __init__(self, known_path = ""):
        if known_path == "":
            raise ValueError("known_path 不能為空字串！請提供有效的人臉資料庫路徑。")
            
        self.known_path = known_path
        self.learning_cache = {}  # {track_id: {"candidate_name": str, "distances": [float] "crops": [np.array]}}
        self.learning_threshold_min = 1.1  # 最小距離閾值（原本的識別閾值）
        self.learning_threshold_max = 2.0  # 最大距離閾值
        self.learning_consecutive_count = 5  # 需要連續匹配的次數

    def isLearning(self, track_id):
        return track_id in self.learning_cache
    
    def getLearningFaceName(self, track_id):
        if track_id in self.learning_cache:
            return self.learning_cache[track_id]["candidate_name"]
        return "None"

    def learning(self, super, known_dir, track_id, candidate_name, distance, crop_frame):
        if track_id not in self.learning_cache:
            # 初始化學習記錄
            self.learning_cache[track_id] = {
                "candidate_name": candidate_name,
                "distances": [distance],
                "crops": [crop_frame.copy()]
            }
            print(f"[學習] Track {track_id} 開始學習候選人: {candidate_name}, 距離: {distance:.4f}")
        else:
            # 檢查是否是同一個候選人
            if self.learning_cache[track_id]["candidate_name"] == candidate_name:
                # 累積證據
                self.learning_cache[track_id]["distances"].append(distance)
                self.learning_cache[track_id]["crops"].append(crop_frame.copy())
                
                count = len(self.learning_cache[track_id]["distances"])
                avg_distance = np.mean(self.learning_cache[track_id]["distances"])
                
                #print(f"[學習] Track {track_id} 累積 {count}/{self.learning_consecutive_count} 次匹配到 {candidate_name}, 平均距離: {avg_distance:.4f}")
                
                # 檢查是否達到學習條件
                if count >= self.learning_consecutive_count:
                    self.addKnownFace(track_id)

                    super.loadKnownFaces(known_dir)
                    print("[學習完成] 人臉資料庫更新完成!")

            else:
                # 候選人改變，重新開始
                print(f"[學習] Track {track_id} 候選人從 {self.learning_cache[track_id]['candidate_name']} 變為 {candidate_name}, 重新開始學習")
                self.learning_cache[track_id] = {
                    "candidate_name": candidate_name,
                    "distances": [distance],
                    "crops": [crop_frame.copy()]
                }

    def addKnownFace(self, track_id):
        if track_id not in self.learning_cache:
            return
            
        learning_data = self.learning_cache[track_id]
        candidate_name = learning_data["candidate_name"]
        
        # 選擇最好的樣本（距離最小的）
        best_idx = np.argmin(learning_data["distances"])
        best_crop = learning_data["crops"][best_idx]
        best_distance = learning_data["distances"][best_idx]
        
        # 保存圖片到對應人名資料夾
        person_dir = Path(self.known_path) / candidate_name
        person_dir.mkdir(exist_ok=True)
        
        # 生成唯一檔案名
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"learned_{track_id}_{timestamp}.png"
        save_path = person_dir / filename
        
        # 儲存
        cv2.imwrite(str(save_path), best_crop)
        
        print(f"[學習完成] 已保存新樣本到: {save_path}")
        print(f"[學習完成] Track {track_id} 學習到 {candidate_name}, 最佳距離: {best_distance:.4f}")
        
        # 清除學習記錄
        del self.learning_cache[track_id]
    