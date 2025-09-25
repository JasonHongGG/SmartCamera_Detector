from enum import Enum
import ctypes

class KeyboardLayoutCode(Enum):
    ENGLISH = "00000409"
    CHINESE = "00000804"

class KeyboardManager:
    def __init__(self):
        pass

    @staticmethod
    def changeKeyboardLayout(layout_code):
        # 00000409 是美式英文鍵盤
        eng_layout = ctypes.windll.user32.LoadKeyboardLayoutW(layout_code, 1)
        ctypes.windll.user32.ActivateKeyboardLayout(eng_layout, 0)