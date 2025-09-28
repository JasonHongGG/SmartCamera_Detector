import cv2
import imutils
import threading
from Manager.LineAlarmManager import LineAlarmManager

class MotionDetector:
    def __init__(self, headless=False, color_threshold=25, motion_threshold=10000, alarm_threshold=20, min_area=500):
        self.headless = headless
        self.color_threshold = color_threshold # 閾值 (threshold-255)
        self.motion_threshold = motion_threshold # 動作強度 (>motion_threshold) 才會被視為有動作
        self.alarm_threshold = alarm_threshold # 動作持續時間 (>alarm_threshold) 才會觸發警報
        self.min_area = min_area # 最小面積 (>min_area) 才會被視為移動物體
        self.pre_frame = None
        self.alarmCounter = 0
        self.alarmTriggerCounter = 0

    def detect(self, frame):
        motion_detected_flag = False
        # frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 初始化前一幀 (才能比較)
        if self.pre_frame is None:
            self.pre_frame = gray
            return False, frame, None

        # 偵測
        delta_frame = cv2.absdiff(self.pre_frame, gray)
        thresh = cv2.threshold(delta_frame, self.color_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2) # 擴大白色區域、讓物體輪廓更明顯、填補小洞
        self.pre_frame = gray

        # 動作的強度
        cv2.putText(frame, f"Threshold: {thresh.sum()}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if thresh.sum() > self.motion_threshold: 
            self.alarmCounter += 1
        else:
            if self.alarmCounter > 0:
                self.alarmCounter -= 1

        # Draw
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            # draw bounding box
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not self.headless:
            cv2.imshow("Motion Detection Threshold", thresh)
            cv2.imshow("Motion Detection", frame)


        # 動作的持續時間
        cv2.putText(frame, f"Counter: {self.alarmCounter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if self.alarmCounter > self.alarm_threshold:
            motion_detected_flag = True
            self.alarmCounter = 0
            threading.Thread(target=LineAlarmManager.triggerAlarm, args=(frame.copy(), "Motion Detector : 有動靜!!!", self.alarmTriggerCounter)).start()
            self.alarmTriggerCounter += 1

        return motion_detected_flag, frame, thresh
