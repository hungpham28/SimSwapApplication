from flask import Flask, render_template, url_for, redirect, request, send_file
from flask_restful import Api, Resource
import time
import base64

import process_files

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config["CACHE_TYPE"] = "null"
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


@app.route('/download/<filename>')
def download_file(filename):
    # Đường dẫn tới file cần download (điều chỉnh theo cấu trúc thư mục của bạn)
    file_path = './output/' + filename

    # Trả file về client
    return send_file(file_path, as_attachment=True)


class GenerateVideoResource(Resource):
    def post(self):

        # valid data
        # required
        if 'target_video' not in request.files:
            return {'error': 'target image: ' + 'Missing required fields'}, 400

        if 'faceswap_image' not in request.files:
            return {'error': 'original video: ' + 'Missing required fields'}, 400

        # Xử lý dữ liệu từ biểu mẫu
        target_video = request.files.get('target_video')
        faceswap_image = request.files.get('faceswap_image')

        # valid type
        if not target_video:
            return {'error': 'target video: ' + 'Invalid data'}, 400

        if not faceswap_image:
            return {'error': 'original image: ' + 'Invalid data'}, 400

        # valid extension
        if not target_video.filename.lower().rsplit('.', 1)[1] in {'mp4'}:
            return {'error': target_video.filename + ': Invalid file type. Type must be *.mp4'}, 400

        if not faceswap_image.filename.lower().rsplit('.', 1)[1] in {'jpg', 'jpeg', 'png', 'gif'}:
            return {'error': faceswap_image.filename + ': Invalid file type. Type must be jpg,jpeg,png,gif'}, 400

        # - Lưu các tệp vào thư mục tạm thời hoặc thư mục nào đó
        target_video.save('./uploads/' + target_video.filename)
        faceswap_image.save('./uploads/' + faceswap_image.filename)
        # - Gọi hàm xử lý video
        # result_path = 'D:\Project_AI\SimSwap\demo_file\multi_people_1080p.mp4'
        result_path = process_files.swap_video('./uploads/' + target_video.filename,
                                               './uploads/' + faceswap_image.filename)
        # - Trả về kết quả
        with open(result_path, 'rb') as video_file:
            encoded_video = base64.b64encode(video_file.read()).decode('utf-8')
        # Thay vì render_template, bạn có thể trả về JSON response chứa kết quả
        return {'message': 'success', 'video': encoded_video, 'link': 'multi_people_1080p.mp4'}


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

        # - Lưu các tệp vào thư mục tạm thời hoặc thư mục nào đó
        faceswap_image.save('./uploads/' + faceswap_image.filename)
        target_image.save('./uploads/' + target_image.filename)
        # - Gọi hàm xử lý ảnh
        result_path = process_files.swap_image('./uploads/' + faceswap_image.filename,
                                               './uploads/' + target_image.filename)
        # result_path = './output/Iron_man.jpg'
        # - Trả về kết quả
        # Đọc nội dung của file vào dạng base64
        with open(result_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        # Thay vì render_template, bạn có thể trả về JSON response chứa kết quả
        return {'message': 'success', 'image': encoded_image, 'link': 'Iron_man.jpg'}


# Đăng ký resource với đường dẫn '/api/generate-image'
api.add_resource(GenerateImageResource, '/api/generate-image')
# Đăng ký resource với đường dẫn '/api/generate-video'
api.add_resource(GenerateVideoResource, '/api/generate-video')

if __name__ == '__main__':
    app.run()
