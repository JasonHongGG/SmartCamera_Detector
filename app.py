import os
import time
import threading
from dotenv import load_dotenv

from CameraProcessor import CarmeraProcessor
from Server.CameraServer import CameraServer

class App:
    def __init__(self):
        self.processor = CarmeraProcessor(camera_index=os.environ.get("CAMERA_INDEX", 0), headless=True)  # headless模式

        # 把功能都關閉
        self.processor.motion_enable = False
        self.processor.face_enable = False
        self.processor.crossLine_enable = False

    def printPrefixInfo(self):
        print("="*50)
        print(f"攝像頭索引: {os.environ.get('CAMERA_INDEX', 0)}")
        print(f"Web 服務地址: http://127.0.0.1:self.")
        print("按 Ctrl+C 停止系統")
        print("="*50)

    def start(self):
        # 啟動攝像頭處理線程
        camera_thread = threading.Thread(target=self.processor.start, daemon=True)
        camera_thread.start()
        
        # 等待攝像頭初始化
        time.sleep(2)
        
        # 啟動 Flask 服務器
        cameraServer = CameraServer(self.processor)
        cameraServer.app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    # 載入環境變數
    load_dotenv()
    
    app = App()
    app.printPrefixInfo()
    app.start()