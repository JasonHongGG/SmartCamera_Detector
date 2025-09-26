
import cv2
import threading
import time

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
    def get_frame(self, stream_type='current'):
        with self.locks[stream_type]:
            frame = self.frames.get(stream_type)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    return buffer.tobytes()
        return None

    def generate_frames(self, stream_type='current'):
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


httpManager = HttpManager()
