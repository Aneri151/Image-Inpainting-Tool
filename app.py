import os
import uuid
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from inpaint_engine import inpaint_custom

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')


@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    session_id = uuid.uuid4().hex
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_folder, exist_ok=True)

    upload_path = os.path.join(session_folder, filename)
    file.save(upload_path)

    image_data = cv2.imdecode(np.fromfile(upload_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image_data is None:
        return redirect(url_for('index'))

    outputs = inpaint_custom(image_data, output_dir=session_folder)
    image_urls = {
        label: url_for('static', filename=os.path.join('uploads', session_id, fname).replace('\\', '/'))
        for label, fname in outputs.items()
    }
    return render_template('result.html', images=image_urls, original=image_urls.get('original'))

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
