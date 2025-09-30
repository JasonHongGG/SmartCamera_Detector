import os
import io
import datetime
import cv2
from dotenv import load_dotenv
load_dotenv()

from .CloudinaryStorage import CloudinaryStorage

class Storage:
    init_flag = False
    image_dir = None

    @staticmethod
    def init():
        CloudinaryStorage.init()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        Storage.image_dir = os.path.join(current_dir, "Image")
        os.makedirs(Storage.image_dir, exist_ok=True)

    def saved_frame(frame, filename):
        if not Storage.init_flag:
            Storage.init_flag = True
            Storage.init()

        filepath = os.path.join(Storage.image_dir, filename)
        success, encoded_image = cv2.imencode('.png', frame)
        if success: 
            with open(filepath, "wb") as f:
                f.write(encoded_image.tobytes())

    @staticmethod
    def upload(encoded_image):
        if not Storage.init_flag:
            Storage.init_flag = True
            Storage.init()

        storage_type = os.getenv("STORAGE_TYPE", "local").lower()
        if storage_type == "local":
            now = datetime.datetime.now()
            filename = now.strftime("alarm_%Y-%m-%d_%H-%M-%S.png")
            filepath = os.path.join(Storage.image_dir, filename)
            with open(filepath, "wb") as f:
                f.write(encoded_image.tobytes())
            return "" # TODO : https 本地圖片，現在先暫且返回空
        elif storage_type == "cloud":
            file_bytes = io.BytesIO(encoded_image.tobytes())
            return CloudinaryStorage.upload(file_bytes)