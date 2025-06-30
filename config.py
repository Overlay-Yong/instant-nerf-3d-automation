import os

# --- ⚠️ 중요 설정 1: instant-ngp 설치 폴더 ---
# instant-ngp가 설치된 폴더의 '전체 경로'를 입력합니다.
# 예: 'C:/Users/YourName/Desktop/instant-ngp'
NERF_PATH = 'C:/Users/Yong/Desktop/instant-ngp/nerf_project/instant-ngp' # ⬅️ 본인의 경로 확인

# --- ⚠️ 중요 설정 2: COLMAP의 'bin' 폴더 경로 ---
# 다운로드한 COLMAP 폴더 안의 'bin' 폴더까지의 '전체 경로'를 입력합니다.
# 예: 'C:/colmap/bin'
COLMAP_BIN_PATH = 'C:/colmap/colmap-x64-windows-cuda/bin' # ⬅️ 본인의 'bin' 폴더 경로 확인

# -----------------------------------------------------------

# --- 아래 설정은 특별한 경우가 아니면 수정할 필요 없습니다 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000
TRAINING_STEPS = 3500
