# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import subprocess
import uuid
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import subprocess
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='../frontend')
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(session_folder, filename))
    
    return jsonify({'session_id': session_id, 'status': 'uploaded'})

@app.route('/process/<session_id>', methods=['POST'])
def process_nerf(session_id):
    input_folder = os.path.join(UPLOAD_FOLDER, session_id)
    
    # 실제 instant-ngp 경로로 수정
    nerf_path = r"C:\Users\Yong\Desktop\instant-ngp\nerf_project\instant-ngp"
    cmd = f"cd {nerf_path} && python scripts/run.py --scene {os.path.abspath(input_folder)} --n_steps 1000"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return jsonify({'status': 'success', 'output': result.stdout})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000) 
