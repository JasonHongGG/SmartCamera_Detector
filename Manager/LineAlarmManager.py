import cv2  
import os
import requests
from Storage.Storage import Storage
from dotenv import load_dotenv
load_dotenv()

class LineAlarmManager:
    alarm_flag = os.getenv("ALARM_FLAG", "false").lower() in ("true", "1", "yes", "on")

    def __init__(self):
        pass
    
    @staticmethod
    def triggerAlarm(frame, msg = "", counter=-1):
        print(f"Alarm Triggered! {counter if counter != -1 else ''}")
        if not LineAlarmManager.alarm_flag: return
        # 儲存並上傳圖片
        success, encoded_image = cv2.imencode('.png', frame)
        if success: 
            image_url = Storage.upload(encoded_image)
    
            requests.post(
                os.environ.get("LINE_IP") + "/triggerAlarm", 
                json={"image_url": image_url, "msg": msg}
                )
        