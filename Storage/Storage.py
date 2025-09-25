import io
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

        file_bytes = io.BytesIO(encoded_image.tobytes())
        return CloudinaryStorage.upload(file_bytes)