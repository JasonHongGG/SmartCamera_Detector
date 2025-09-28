import cv2

class CrossLineManager:
    def __init__(self, cv_window_name = "CrossLine", headless = False):
        self.headless = headless
        self.lines = [[(0, 0), (0, 0)]]  # 支援多條線，預設一條線
        self.drawing = False
        self.prev_center = {} #存儲每個 track_id 的前一個位置

        if self.headless:
            self.lines = []  # 初始化是為了給 opencv mouse callback 使用，否則預設沒有任何線段
            cv2.namedWindow(cv_window_name)
            cv2.setMouseCallback(cv_window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        # 只操作第一條線 (index 0)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.lines[0][0] = (x, y)
            self.lines[0][1] = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.lines[0][1] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.lines[0][1] = (x, y)
            self.drawing = False

    def is_cross_line(self, center, track_id):
        if track_id not in self.prev_center:
            self.prev_center[track_id] = center
            return False

        def side(p, a, b):
            return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

        crossed = False
        # 檢查每一條線
        for i, line in enumerate(self.lines):
            # 計算前後兩個 frame 中人物相對於線所在的左邊還是右邊
            side_prev = side(self.prev_center[track_id], line[0], line[1])
            side_curr = side(center, line[0], line[1])
            
            # 如果符號不同，代表跨線
            if side_prev * side_curr < 0:  # 跨線
                crossed = True
                if side_prev > 0 and side_curr < 0:
                    print(f"Line {i}: A→B")  # 從線的一側到另一側
                elif side_prev < 0 and side_curr > 0:
                    print(f"Line {i}: B→A")  # 從另一側回到線的一側
        
        self.prev_center[track_id] = center
        return crossed

    def draw_line(self, frame):
        for i, line in enumerate(self.lines):
            # 不同線條使用不同顏色
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[i % len(colors)]
            cv2.line(frame, line[0], line[1], color, 2)
            # 在線條旁邊標註編號
            mid_point = ((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2)
            cv2.putText(frame, f"L{i}", (mid_point[0] + 5, mid_point[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def add_line(self, start_point, end_point):
        self.lines.append([start_point, end_point])
        return len(self.lines) - 1  # 回傳新線的索引

    def clear_lines(self):
        self.lines = [] 