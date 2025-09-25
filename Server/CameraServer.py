from flask import Flask, Response, request, jsonify
from Manager.HttpManager import HttpManager, httpManager
from CameraProcessor import CarmeraProcessor

class CameraServer:
    def __init__(self, processor : CarmeraProcessor):
        self.processor = processor
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):

        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response
        
        @self.app.route('/status')
        def status():
            """返回系統狀態"""
            return {
                'running': True,
                'camera_connected': False
            }
        
        @self.app.route('/capture')
        def capture():
            return Response(httpManager.get_frame("current"), mimetype='image/jpeg')
               
        @self.app.route('/stream')
        def stream():
            return Response(httpManager.generate_frames("current"),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
            
        @self.app.route('/<type>/stream')
        def video_feed(type):
            return Response(httpManager.generate_frames(type),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/detection/<type>', methods=['POST'])
        def toggle_detection(type):
            data = request.get_json()
            enabled = data.get('enabled', False)

            if type == 'motion':
                self.processor.motion_enable = enabled
                return jsonify({'status': 'success', 'message': f'Motion detection {"enabled" if enabled else "disabled"}'})
            elif type == 'face':
                self.processor.face_enable = enabled
                return jsonify({'status': 'success', 'message': f'Face recognition {"enabled" if enabled else "disabled"}'})
            elif type == 'crossline':
                self.processor.crossLine_enable = enabled
                return jsonify({'status': 'success', 'message': f'Cross line detection {"enabled" if enabled else "disabled"}'})

        @self.app.route('/motion/info')
        def get_motion_info():
            motion_info = httpManager.get_motion_info()
            return jsonify(motion_info)
        
        @self.app.route('/motion/sensitivity', methods=['POST'])
        def set_motion_sensitivity():
            data = request.get_json()
            motion_threshold = data.get('motion_threshold', 10000)
            alarm_threshold = data.get('alarm_threshold', 20)
            self.processor.motion_detector.motion_threshold = motion_threshold
            self.processor.motion_detector.alarm_threshold = alarm_threshold
            print(f"Motion sensitivity set to {motion_threshold}, alarm threshold set to {alarm_threshold}")
            return jsonify({'status': 'success', 'message': f'Motion sensitivity set to {self.processor.motion_detector.motion_threshold} and alarm threshold set to {self.processor.motion_detector.alarm_threshold}'})

        @self.app.route('/face/info')
        def get_face_info():
            face_info = httpManager.get_face_info()
            return jsonify(face_info)
        
        @self.app.route('/crossline/lines', methods=['POST'])
        def set_crossline_lines():
            data = request.get_json()
            lines_data = data.get('lines', [])
            image_width = data.get('image_width', 640)
            image_height = data.get('image_height', 480)
            
            # 清空現有線條
            self.processor.crossLineMgr.clear_lines()

            # 如果沒有線段，表示清除所有線段
            if not lines_data:
                return jsonify({'status': 'success', 'message': f'清除所有線段'})
            
            # 添加新線條
            added_lines = []
            for i, line_data in enumerate(lines_data):
                start_x = line_data.get('startX', 0) 
                start_y = line_data.get('startY', 0)
                end_x = line_data.get('endX', 0) 
                end_y = line_data.get('endY', 0)
                
                # 驗證座標範圍
                if (0 <= start_x <= image_width and 0 <= start_y <= image_height and
                    0 <= end_x <= image_width and 0 <= end_y <= image_height):
                        
                    self.processor.crossLineMgr.add_line((start_x, start_y), (end_x, end_y))
                    
                    added_lines.append({
                        'index': i,
                        'start': (start_x, start_y),
                        'end': (end_x, end_y)
                    })
                else:
                    print(f"警告: 線條 {i} 座標超出範圍，已跳過")
            
            print(f"跨線檢測設定完成，共 {len(added_lines)} 條線")
            for line in added_lines:
                print(f"  線條 {line['index']}: {line['start']} -> {line['end']}")
            
            return jsonify({
                'status': 'success', 
                'message': f'設定 {len(added_lines)} 條跨線檢測線',
                'lines_count': len(added_lines),
                'lines': added_lines
            })
        
        @self.app.route('/crossline/info')
        def get_crossline_info():
            crossline_info = httpManager.get_crossline_info()
            return jsonify(crossline_info)
            