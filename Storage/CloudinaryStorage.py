import os
import time
import cloudinary
import cloudinary.uploader

class CloudinaryStorage:
    @staticmethod
    def init():
        cloudinary.config(
            cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key = os.getenv("CLOUDINARY_API_KEY"),
            api_secret = os.getenv("CLOUDINARY_API_SECRET")
        )

    @staticmethod
    def upload(file_bytes):
        try:
            response = cloudinary.uploader.upload(
                file_bytes, 
                public_id="latest_photo",
                overwrite=True,
                resource_type="image"
            )
            print("最新圖片 URL:", response['secure_url'])
            return response.get('secure_url')
        except Exception as e:
            print(f"Upload Error: {e}")
            return None