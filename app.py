from flask import Flask, render_template, url_for, redirect, request, send_file
from flask_restful import Api, Resource
import time
import base64
import process_files
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
api = Api(app)


@app.route('/')
def index():
    return redirect(url_for('image_page'))


@app.route('/image')
def image_page():
    return render_template('image_page.html')


@app.route('/video')
def video_page():  # put application's code here
    return render_template('video_page.html')


class GenerateVideoResource(Resource):
    def post(self):

        # valid data
        # required
        if 'target_video' not in request.files:
            return {'error': 'target image: ' + 'Missing required fields'}, 400

        if 'faceswap_image' not in request.files:
            return {'error': 'original video: ' + 'Missing required fields'}, 400

        if 'des_image' not in request.files:
            return {'error': 'destination video: ' + 'Missing required fields'}, 400

        # Xử lý dữ liệu từ biểu mẫu
        target_video = request.files.get('target_video')
        faceswap_image = request.files.get('faceswap_image')
        des_image = request.files.get('des_image')

        # valid type
        if not target_video:
            return {'error': 'target video: ' + 'Invalid data'}, 400

        if not faceswap_image:
            return {'error': 'original image: ' + 'Invalid data'}, 400

        if not des_image:
            return {'error': 'destination image: ' + 'Invalid data'}, 400

        # valid extension
        if not target_video.filename.lower().rsplit('.', 1)[1] in {'mp4'}:
            return {'error': target_video.filename + ': Invalid file type. Type must be *.mp4'}, 400

        if not faceswap_image.filename.lower().rsplit('.', 1)[1] in {'jpg', 'jpeg', 'png', 'gif'}:
            return {'error': faceswap_image.filename + ': Invalid file type. Type must be jpg,jpeg,png,gif'}, 400

        if not des_image.filename.lower().rsplit('.', 1)[1] in {'jpg', 'jpeg', 'png', 'gif'}:
            return {'error': des_image.filename + ': Invalid file type. Type must be jpg,jpeg,png,gif'}, 400
        # # Accessing file data as bytes
        # target_video_data = target_video.read()
        # faceswap_image_data = faceswap_image.read()
        # # valid size
        # if len(faceswap_image_data) > 3 * 1024 * 1024:  # 5 MB
        #     return {'error': 'File ' + target_video.filename + ': size exceeds the limit. File size must be less than '
        #                                                        '3MB'}, 400
        #
        # if len(target_video_data) > 50 * 1024 * 1024:  # 50 MB
        #     return {'error': 'File ' + faceswap_image.filename + ': size exceeds the limit. File size must be less '
        #                                                          'than 50MB'}, 400

        # Giả lập việc xử lý mất thời gian
        time.sleep(5)  # Ngủ trong 5 giây
        # Thực hiện xử lý dữ liệu tại đây, ví dụ:
        # - Lưu các tệp vào thư mục tạm thời hoặc thư mục nào đó
        target_video.save('uploads/' + target_video.filename)
        faceswap_image.save('uploads/' + faceswap_image.filename)
        des_image.save('uploads/' + des_image.filename)
        # - Gọi hàm xử lý video
        # - Trả về kết quả

        # Thay vì render_template, bạn có thể trả về JSON response chứa kết quả
        return send_file(target_video.filename, as_attachment=True)


class GenerateImageResource(Resource):
    def post(self):
        # valid data
        # required
        if 'target_image' not in request.files:
            return {'error': 'target image: ' + 'Missing required fields'}, 400

        if 'faceswap_image' not in request.files:
            return {'error': 'original image: ' + 'Missing required fields'}, 400

        # Xử lý dữ liệu từ biểu mẫu
        target_image = request.files.get('target_image')
        faceswap_image = request.files.get('faceswap_image')

        # valid type
        if not target_image:
            return {'error': 'target image: ' + 'Invalid data'}, 400

        if not faceswap_image:
            return {'error': 'original image: ' + 'Invalid data'}, 400

        # valid extension
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if not target_image.filename.lower().rsplit('.', 1)[1] in allowed_extensions:
            return {'error': target_image.filename + ': Invalid file type. Type must be jpg,jpeg,png,gif'}, 400

        if not faceswap_image.filename.lower().rsplit('.', 1)[1] in allowed_extensions:
            return {'error': faceswap_image.filename + ': Invalid file type. Type must be jpg,jpeg,png,gif'}, 400

        # Accessing file data as bytes
        # target_image_data = target_image.read()
        # faceswap_image_data = faceswap_image.read()
        # # valid size
        # if len(target_image_data) > 3 * 1024 * 1024:  # 3 MB
        #     return {'error': 'File ' + target_image.filename + ': size exceeds the limit. File size must be less '
        #                                                        'than 3MB'}, 400
        #
        # if len(faceswap_image_data) > 3 * 1024 * 1024:  # 3 MB
        #     return {'error': 'File ' + faceswap_image.filename + ': size exceeds the limit. File size must be less '
        #                                                          'than 3MB'}, 400

        # Giả lập việc xử lý mất thời gian
        # time.sleep(5)  # Ngủ trong 5 giây
        # Thực hiện xử lý dữ liệu tại đây, ví dụ:
        # - Lưu các tệp vào thư mục tạm thời hoặc thư mục nào đó
        faceswap_image.save('./uploads/' + faceswap_image.filename)
        target_image.save('./uploads/' + target_image.filename)
        # - Gọi hàm xử lý ảnh
        result_path=process_files.swap_image('./uploads/' + faceswap_image.filename,'./uploads/' + target_image.filename)
        # - Trả về kết quả
        # Đọc nội dung của file vào dạng base64
        with open(result_path, 'rb') as video_file:
            encoded_video = base64.b64encode(video_file.read()).decode('utf-8')
        # Thay vì render_template, bạn có thể trả về JSON response chứa kết quả
        return {'message':'success', 'image':encoded_video}


# Đăng ký resource với đường dẫn '/api/generate-image'
api.add_resource(GenerateImageResource, '/api/generate-image')
# Đăng ký resource với đường dẫn '/api/generate-video'
api.add_resource(GenerateVideoResource, '/api/generate-video')

if __name__ == '__main__':
    app.run()
