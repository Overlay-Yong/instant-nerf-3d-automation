import os
import subprocess
import sys
import json
import shutil
import numpy as np
from PIL import Image
import trimesh

# 설정 파일에서 필요한 변수들을 가져옵니다.
from config import NERF_PATH, TRAINING_STEPS, COLMAP_BIN_PATH

def validate_and_optimize_images(images_folder):
    """
    업로드된 이미지들을 검증하고, 너무 큰 이미지는 학습에 적합하도록 최적화합니다.
    """
    print("🔍 이미지 검증 및 최적화 시작...")
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_files) < 20:
        return False, f"이미지가 너무 적습니다: {len(image_files)}장 (COLMAP 사용 시 최소 20장 권장)"
    if len(image_files) > 200:
        return False, f"이미지가 너무 많습니다: {len(image_files)}장 (최대 200장)"

    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        try:
            with Image.open(img_path) as img:
                if img.width > 1600 or img.height > 1600:
                    img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                    img.save(img_path, quality=90, optimize=True)
                    print(f"    - 이미지 최적화 완료: {img_file}")
        except Exception as e:
            print(f"⚠️ 이미지 처리 실패: {img_file} - {e}")
            return False, f"손상되었거나 잘못된 이미지 파일({img_file})이 있습니다."

    return True, f"✅ 이미지 {len(image_files)}장 검증 및 최적화 완료!"


def run_colmap_processing(session_folder):
    """
    획기적인 최종 방식: COLMAP의 모든 단계를 직접 제어하고,
    코드의 버그를 수정하여 안정적으로 transforms.json 파일을 생성합니다.
    """

    # --- 1. 경로 설정 및 환경 준비 ---
    images_folder = os.path.join(session_folder, 'images')
    database_path = os.path.join(session_folder, 'colmap.db')
    sparse_path = os.path.join(session_folder, 'colmap_sparse')
    text_path = os.path.join(session_folder, 'colmap_text')
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)

    colmap_executable = os.path.join(COLMAP_BIN_PATH, "colmap.exe")
    if not os.path.exists(colmap_executable):
        raise FileNotFoundError(f"COLMAP 실행 파일을 찾을 수 없습니다: {colmap_executable}")

    def run_command(args):
        full_command = [colmap_executable] + args
        yield f"🚀 실행: {' '.join(full_command)}"
        process = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in iter(process.stdout.readline, ''):
            yield line.strip()
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"COLMAP 명령어 실행 실패 (종료 코드: {process.returncode}): {' '.join(args)}")

    # --- 2. COLMAP 파이프라인 직접 실행 ---
    yield "--- 1/3: 이미지 특징점 추출 시작 ---"
    yield from run_command(["feature_extractor", "--database_path", database_path, "--image_path", images_folder, "--ImageReader.camera_model", "OPENCV"])

    yield "--- 2/3: 특징점 매칭 시작 ---"
    yield from run_command(["exhaustive_matcher", "--database_path", database_path])

    yield "--- 3/3: 3D 모델 재구성(매핑) 시작 ---"
    yield from run_command(["mapper", "--database_path", database_path, "--image_path", images_folder, "--output_path", sparse_path])

    if not os.listdir(sparse_path):
        raise RuntimeError("COLMAP 매핑에 실패하여 sparse 모델을 생성하지 못했습니다. 이미지 품질이나 촬영 각도를 확인해주세요.")
    
    yield "✅ COLMAP 3D 재구성 완료!"
    
    # --- 3. [핵심 버그 수정] transforms.json 직접 생성 ---
    yield "--- NeRF 형식(transforms.json) 직접 생성 시작 ---"
    
    # 3-1. 바이너리 모델을 텍스트 파일로 변환
    yield from run_command(["model_converter", "--input_path", os.path.join(sparse_path, "0"), "--output_path", text_path, "--output_type", "TXT"])

    # 3-2. 생성된 텍스트 파일 읽기 (안정적인 파싱 로직으로 수정)
    cameras = {}
    with open(os.path.join(text_path, "cameras.txt"), "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]
            fl_x, fl_y, cx, cy = params[0], params[1], params[2], params[3]
            cameras[cam_id] = { "w": W, "h": H, "fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy }

    all_frames = []
    with open(os.path.join(text_path, "images.txt"), "r") as f:
        lines = [line for line in f if not line.startswith("#")]
        for i in range(0, len(lines), 2):
            img_line = lines[i]
            parts = img_line.split()
            img_id = int(parts[0])
            q = np.array([float(p) for p in parts[1:5]]) # qw, qx, qy, qz
            t = np.array([float(p) for p in parts[5:8]]) # tx, ty, tz
            cam_id = int(parts[8])
            img_name = parts[9]
            
            R = np.eye(3)
            R[0, 0] = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
            R[0, 1] = 2.0 * (q[1] * q[2] - q[3] * q[0])
            R[0, 2] = 2.0 * (q[1] * q[3] + q[2] * q[0])
            R[1, 0] = 2.0 * (q[1] * q[2] + q[3] * q[0])
            R[1, 1] = 1.0 - 2.0 * (q[1] * q[1] + q[3] * q[3])
            R[1, 2] = 2.0 * (q[2] * q[3] - q[1] * q[0])
            R[2, 0] = 2.0 * (q[1] * q[3] - q[2] * q[0])
            R[2, 1] = 2.0 * (q[2] * q[3] + q[1] * q[0])
            R[2, 2] = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
            
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = t
            
            frame = {
                "file_path": os.path.join("images", img_name).replace("\\", "/"),
                "transform_matrix": transform_matrix.tolist(),
                **cameras[cam_id]
            }
            all_frames.append(frame)

    # 3-3. 최종 transforms.json 파일 작성
    first_cam_info = next(iter(cameras.values()))
    transforms_data = {
        "camera_angle_x": np.arctan(first_cam_info['w'] / (2 * first_cam_info['fl_x'])),
        "w": first_cam_info['w'],
        "h": first_cam_info['h'],
        "fl_x": first_cam_info['fl_x'],
        "fl_y": first_cam_info['fl_y'],
        "cx": first_cam_info['cx'],
        "cy": first_cam_info['cy'],
        "aabb_scale": 4,
        "frames": all_frames
    }

    with open(os.path.join(session_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms_data, f, indent=4)
    
    yield "✅ 모든 전처리 완료! NeRF 학습을 시작합니다."


def run_nerf_training(session_folder):
    """
    완벽하게 생성된 transforms.json을 사용하여 NeRF 모델을 학습합니다.
    """
    transforms_path = os.path.join(session_folder, 'transforms.json')
    if not os.path.exists(transforms_path):
        raise FileNotFoundError("transforms.json 파일을 찾을 수 없습니다.")

    output_path = os.path.join(session_folder, 'nerf_output')
    os.makedirs(output_path, exist_ok=True)
    snapshot_path = os.path.join(output_path, 'model.msgpack')

    yield "🧠 NeRF 모델 학습을 준비합니다..."

    command = [
        sys.executable,
        'scripts/run.py',
        f'--training_data={transforms_path}',
        '--n_steps', str(TRAINING_STEPS),
        '--save_snapshot', snapshot_path
    ]

    yield f"📊 {TRAINING_STEPS} 스텝으로 학습을 시작합니다..."

    process = subprocess.Popen(command, cwd=NERF_PATH, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in iter(process.stdout.readline, ''):
        yield line.strip()

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

    if os.path.exists(snapshot_path):
        yield "✅ NeRF 학습 완료!"
    else:
        raise FileNotFoundError("학습된 모델 파일을 찾을 수 없습니다.")


def export_to_mesh(session_folder):
    """
    학습된 NeRF 모델에서 3D 메쉬를 추출하고 변환합니다.
    """
    yield "✨ 3D 모델(메쉬) 추출을 시작합니다..."
    model_path = os.path.join(session_folder, 'nerf_output', 'model.msgpack')
    output_mesh_ply = os.path.join(session_folder, 'model.ply')
    output_mesh_obj = os.path.join(session_folder, 'model.obj')

    if not os.path.exists(model_path):
        raise FileNotFoundError("메쉬를 추출할 학습된 모델(.msgpack)을 찾을 수 없습니다.")

    command = [sys.executable, 'scripts/run.py', '--load_snapshot', model_path, '--save_mesh', output_mesh_ply, '--marching_cubes_res', '256']
    process = subprocess.run(command, cwd=NERF_PATH, capture_output=True, text=True, encoding='utf-8')

    if process.returncode == 0 and os.path.exists(output_mesh_ply):
        yield f"✅ PLY 형식 메쉬 추출 완료!"
    else:
        raise RuntimeError(f"메쉬 추출 과정에서 오류가 발생했습니다: {process.stdout} {process.stderr}")

    try:
        yield "📦 PLY를 OBJ 형식으로 변환합니다..."
        mesh = trimesh.load(output_mesh_ply)
        mesh.export(output_mesh_obj)
        yield "✅ OBJ 형식 변환 완료!"
    except ImportError:
        yield "⚠️ 'trimesh' 라이브러리가 없어 OBJ 변환을 건너뜁니다."
    except Exception as e:
        yield f"❌ OBJ 변환 오류: {str(e)}"
