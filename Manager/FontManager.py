import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class FontManager:
    def __init__(self):
        self.fontSize = 25
        self.CurFilePath = os.path.dirname(os.path.abspath(__file__))
        self.font = ImageFont.truetype(os.path.join(self.CurFilePath, "Font/NotoSansTC-Medium.ttf"), self.fontSize, encoding="utf-8")
        

    def cv2AddChineseText(self, img, text, position, textColor=(0,255,0), textSize=30):
        if(isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img)
        fontStyle = self.font
        if textSize != self.fontSize:
            fontStyle = ImageFont.truetype(os.path.join(self.CurFilePath, "Font/NotoSansTC-Medium.ttf"), textSize, encoding="utf-8")
        draw.text(position, text, textColor, font=fontStyle)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    
fontMgr = FontManager()