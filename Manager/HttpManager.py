
import cv2
import threading
import time
import os
import base64
from PIL import Image

class HttpManager:
    def __init__(self, camera_index=0, width=640, height=480):
        # 分別的鎖定機制
        self.frames = {
            'current': None,
            'motion': None,
            'face': None,
            'crossline': None
        }
        
        # 為每個流創建獨立的鎖
        self.locks = {
            'current': threading.Lock(),
            'motion': threading.Lock(),
            'face': threading.Lock(),
            'crossline': threading.Lock()
        }

        # motion 資訊
        self.motion_info = {
            'lastDetection': None
        }
        self.motion_info_lock = threading.Lock()
        
        # 人臉辨識資訊
        self.face_info = {
            'faceCount': 0,
            'faceNames': "", # name, name, name, ....
            'lastDetection': "",  # current timestamp 
        }
        self.face_info_lock = threading.Lock()

        # crossline 資訊
        self.crossline_info = {
            'crossingEvent': "",  # 誰穿過哪條線
            'lastDetection': "",  # current timestamp 
        }
        self.crossline_info_lock = threading.Lock()
        
        
        self.frame_times = []

    # opencv 處理完後更新畫面
    def update_frame(self, stream_type, frame):
        with self.locks[stream_type]:
            self.frames[stream_type] = frame.copy()

    # 為 Http 提供畫面
    def get_frame(self, stream_type='original'):
        with self.locks[stream_type]:
            frame = self.frames.get(stream_type)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    return buffer.tobytes()
        return None

    def generate_frames(self, stream_type='original'):
        while True:
            frame_bytes = self.get_frame(stream_type)
            if frame_bytes:
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # 約30fps

    def update_motion_info(self, motion_detected):
        with self.motion_info_lock:
            if motion_detected:
                self.motion_info['lastDetection'] = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    
    def get_motion_info(self):
        with self.motion_info_lock:
            return self.motion_info.copy()

    # 更新人臉辨識資訊
    def update_face_info(self, face_info_list):
        with self.face_info_lock:
            self.face_info['faceCount'] = len(face_info_list)
            self.face_info['faceNames'] = ", ".join(name for (x1, y1, x2, y2, track_id, name) in face_info_list)
            self.face_info['lastDetection'] = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())

    # 取得人臉辨識資訊
    def get_face_info(self):
        with self.face_info_lock:
            return self.face_info.copy()
        
    def update_crossline_info(self, crossing_event):
        with self.crossline_info_lock:
            if crossing_event:
                self.crossline_info['crossingEvent'] = crossing_event
                self.crossline_info['lastDetection'] = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        
    def get_crossline_info(self):
        with self.crossline_info_lock:
            return self.crossline_info.copy()
        
    # 取得儲存的圖片
    def _format_file_size(self, size_bytes):
        """將位元組大小轉換為適當的單位"""
        if size_bytes >= 1024 * 1024:  # >= 1MB
            size_mb = size_bytes / (1024 * 1024)
            return f"{size_mb:.2f} MB"
        elif size_bytes >= 1024:  # >= 1KB
            size_kb = size_bytes / 1024
            return f"{size_kb:.2f} KB"
        else:
            return f"{size_bytes} bytes"
        
    def get_image(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(current_dir, "../Storage/Image")
        filepath = os.path.join(image_dir, filename)
        
        if os.path.exists(filepath):
            try:
                # 取得圖片尺寸和格式
                with Image.open(filepath) as img:
                    image_width, image_height = img.size
                    image_format = img.format.lower() if img.format else None
                
                # 從檔名取得副檔名作為備用
                file_extension = os.path.splitext(filename)[1][1:].lower() if '.' in filename else 'unknown'
                
                # 優先使用PIL檢測的格式，若無則使用副檔名
                image_type = image_format if image_format else file_extension
                
                # 讀取圖片數據
                with open(filepath, "rb") as f:
                    image_data = f.read()
                    encoded_image = base64.b64encode(image_data).decode('utf-8')
                
                # 轉換檔案大小
                file_size_bytes = os.path.getsize(filepath)
                file_size_str = self._format_file_size(file_size_bytes)
                
                return {
                    'type': image_type,
                    'filename': filename,
                    'data': encoded_image,
                    'image_size': f"{image_width}x{image_height}",
                    'file_size': file_size_str
                }
            except Exception as e:
                print(f"讀取圖片時發生錯誤: {e}")
                return None
        else:
            return None
        
    def get_all_images(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(current_dir, "../Storage/Image")
        
        if os.path.exists(image_dir):
            images = []
            for filename in os.listdir(image_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_info = self.get_image(filename)
                    if image_info:
                        images.append(image_info)
            return images
        else:
            print(f"圖片目錄不存在: {image_dir}")
            return []

httpManager = HttpManager()
