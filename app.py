import os
import uuid
import json
import shutil
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from werkzeug.utils import secure_filename

# 설정 파일 및 파이프라인 함수들을 임포트합니다.
import config
from nerf_pipeline import (
    validate_and_optimize_images,
    run_colmap_processing,
    run_nerf_training,
    export_to_mesh
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 최대 업로드 용량 200MB

# 업로드 폴더가 없으면 생성합니다.
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """허용된 이미지 파일 확장자인지 확인합니다."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """이미지 파일을 업로드하고 검증합니다."""
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    images_folder = os.path.join(session_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)
    
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'error': '업로드된 파일이 없습니다.'}), 400

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(images_folder, filename))
    
    # 이미지 검증 및 최적화
    is_valid, message = validate_and_optimize_images(images_folder)
    if not is_valid:
        shutil.rmtree(session_folder) # 유효하지 않으면 생성된 폴더를 삭제
        return jsonify({'success': False, 'error': message}), 400
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': message
    })

@app.route('/process/<session_id>')
def process_session(session_id):
    """COLMAP과 NeRF 처리 과정을 실시간 스트리밍으로 진행합니다."""
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.isdir(session_folder):
        return Response("잘못된 세션 ID 입니다.", status=404)

    def event_stream():
        """Server-Sent Events(SSE)를 사용하여 클라이언트에 진행 상황을 실시간 전송합니다."""
        def stream_event(event_type, data):
            return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"
        
        try:
            # 1단계: COLMAP으로 카메라 위치 추정
            yield stream_event("progress", {"percent": 5, "text": "📸 COLMAP 카메라 위치 추정 중..."})
            for log_line in run_colmap_processing(session_folder):
                 yield stream_event("log", log_line)
            
            # 2단계: NeRF 모델 학습
            yield stream_event("progress", {"percent": 50, "text": "🧠 NeRF 모델 학습 중..."})
            for log_line in run_nerf_training(session_folder):
                yield stream_event("log", log_line)
            
            # 3단계: 3D 메쉬 추출
            yield stream_event("progress", {"percent": 90, "text": "✨ 3D 모델(메쉬) 추출 중..."})
            for log_line in export_to_mesh(session_folder):
                yield stream_event("log", log_line)
                
            # 4단계: 완료
            yield stream_event("progress", {"percent": 100, "text": "🎉 완료! 뷰어 페이지로 이동합니다.", "status": "success"})

        except Exception as e:
            # 어떤 단계에서든 에러가 발생하면 클라이언트에 에러 메시지 전송
            import traceback
            error_message = f"❌ 치명적인 오류 발생: {str(e)}"
            yield stream_event("log", error_message)
            yield stream_event("log", traceback.format_exc()) # 상세 에러 로그 추가
            yield stream_event("progress", {"percent": 100, "text": "오류가 발생했습니다. 로그를 확인해주세요.", "status": "failed"})
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/viewer/<session_id>')
def viewer(session_id):
    """생성된 3D 모델을 보여주는 뷰어 페이지를 렌더링합니다."""
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    mesh_url = None
    
    if os.path.exists(os.path.join(session_folder, 'model.obj')):
        mesh_url = f'/download/{session_id}/model.obj'
    elif os.path.exists(os.path.join(session_folder, 'model.ply')):
        mesh_url = f'/download/{session_id}/model.ply'
    else:
        return "<h3>생성된 3D 모델을 찾을 수 없습니다.</h3><p>처리 과정에서 오류가 발생했을 수 있습니다. 이전 페이지의 로그를 확인해주세요.</p>", 404
        
    return render_template('viewer.html', session_id=session_id, mesh_url=mesh_url)

@app.route('/download/<session_id>/<path:filename>')
def download_file(session_id, filename):
    """생성된 모델 파일을 다운로드할 수 있게 합니다."""
    directory = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
