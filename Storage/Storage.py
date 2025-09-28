import os
import io
import datetime
from dotenv import load_dotenv
load_dotenv()

from .CloudinaryStorage import CloudinaryStorage

class Storage:
    init_flag = False

    @staticmethod
    def init():
        CloudinaryStorage.init() 

    @staticmethod
    def upload(encoded_image):
        if not Storage.init_flag:
            Storage.init_flag = True
            Storage.init()

        storage_type = os.getenv("STORAGE_TYPE", "local").lower()
        if storage_type == "local":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image_dir = os.path.join(current_dir, "Image")
            os.makedirs(image_dir, exist_ok=True)
            now = datetime.datetime.now()
            filename = now.strftime("alarm_%Y-%m-%d_%H-%M-%S.png")
            filepath = os.path.join(image_dir, filename)
            with open(filepath, "wb") as f:
                f.write(encoded_image.tobytes())
            return "" # TODO : https 本地圖片，現在先暫且返回空
        elif storage_type == "cloud":
            file_bytes = io.BytesIO(encoded_image.tobytes())
            return CloudinaryStorage.upload(file_bytes)