﻿import os
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
    최종 방식: COLMAP의 모든 핵심 단계를 직접 제어하여 안정성을 극대화합니다.
    """
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

    yield "--- 1/4: 이미지 특징점 추출 시작 ---"
    yield from run_command(["feature_extractor", "--database_path", database_path, "--image_path", images_folder, "--ImageReader.camera_model", "OPENCV"])
    yield "--- 2/4: 특징점 매칭 시작 ---"
    yield from run_command(["exhaustive_matcher", "--database_path", database_path])
    yield "--- 3/4: 3D 모델 재구성(매핑) 시작 ---"
    yield from run_command(["mapper", "--database_path", database_path, "--image_path", images_folder, "--output_path", sparse_path])
    if not os.listdir(sparse_path):
        raise RuntimeError("COLMAP 매핑에 실패하여 sparse 모델을 생성하지 못했습니다. 이미지 품질이나 촬영 각도를 확인해주세요.")
    yield "--- 4/4: 생성된 모델을 텍스트 파일로 변환 시작 ---"
    yield from run_command(["model_converter", "--input_path", os.path.join(sparse_path, "0"), "--output_path", text_path, "--output_type", "TXT"])
    if not os.path.exists(os.path.join(text_path, "cameras.txt")):
        raise FileNotFoundError("COLMAP 모델을 텍스트로 변환하는 데 실패했습니다. cameras.txt 파일이 생성되지 않았습니다.")
    
    yield "--- NeRF 형식(transforms.json) 직접 생성 시작 ---"
    cameras = {}
    with open(os.path.join(text_path, "cameras.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split()
            cam_id, model, W, H = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]
            fl_x, fl_y, cx, cy = params[0], params[1], params[2], params[3]
            cameras[cam_id] = { "w": W, "h": H, "fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy }
    all_frames = []
    with open(os.path.join(text_path, "images.txt"), "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            q = np.array([float(p) for p in parts[1:5]]); t = np.array([float(p) for p in parts[5:8]])
            cam_id, img_name = int(parts[8]), parts[9]
            R = np.eye(3)
            R[0, 0] = 1 - 2 * (q[2]**2 + q[3]**2); R[0, 1] = 2 * (q[1]*q[2] - q[3]*q[0]); R[0, 2] = 2 * (q[1]*q[3] + q[2]*q[0])
            R[1, 0] = 2 * (q[1]*q[2] + q[3]*q[0]); R[1, 1] = 1 - 2 * (q[1]**2 + q[3]**2); R[1, 2] = 2 * (q[2]*q[3] - q[1]*q[0])
            R[2, 0] = 2 * (q[1]*q[3] - q[2]*q[0]); R[2, 1] = 2 * (q[2]*q[3] + q[1]*q[0]); R[2, 2] = 1 - 2 * (q[1]**2 + q[2]**2)
            transform_matrix = np.eye(4); transform_matrix[:3, :3] = R; transform_matrix[:3, 3] = t
            frame = {"file_path": os.path.join("images", img_name).replace("\\", "/"), "transform_matrix": transform_matrix.tolist(), **cameras[cam_id]}
            all_frames.append(frame)
    first_cam_info = next(iter(cameras.values()))
    transforms_data = {"camera_angle_x": np.arctan(first_cam_info['w'] / (2 * first_cam_info['fl_x'])), **first_cam_info, "aabb_scale": 4, "frames": all_frames}
    with open(os.path.join(session_folder, 'transforms.json'), 'w') as f: json.dump(transforms_data, f, indent=4)
    yield "✅ 모든 전처리 완료! NeRF 학습을 시작합니다."

def run_nerf_training(session_folder):
    transforms_path = os.path.join(session_folder, 'transforms.json')
    snapshot_path = os.path.join(session_folder, 'nerf_output', 'model.msgpack')
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    yield "🧠 NeRF 모델 학습을 준비합니다..."
    command = [sys.executable, os.path.join(NERF_PATH, 'scripts/run.py'), f'--training_data={transforms_path}', '--n_steps', str(TRAINING_STEPS), '--save_snapshot', snapshot_path]
    yield f"📊 {TRAINING_STEPS} 스텝으로 학습을 시작합니다..."
    process = subprocess.Popen(command, cwd=NERF_PATH, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in iter(process.stdout.readline, ''): yield line.strip()
    process.wait()
    if process.returncode != 0: raise subprocess.CalledProcessError(process.returncode, command)
    if not os.path.exists(snapshot_path): raise FileNotFoundError("학습된 모델 파일을 찾을 수 없습니다.")
    yield "✅ NeRF 학습 완료!"

def export_to_mesh(session_folder):
    yield "✨ 3D 모델(메쉬) 추출을 시작합니다..."
    model_path = os.path.join(session_folder, 'nerf_output', 'model.msgpack')
    output_mesh_obj = os.path.join(session_folder, 'model.obj')
    if not os.path.exists(model_path): raise FileNotFoundError("메쉬를 추출할 학습된 모델(.msgpack)을 찾을 수 없습니다.")
    command = [sys.executable, os.path.join(NERF_PATH, 'scripts/run.py'), '--load_snapshot', model_path, '--save_mesh', output_mesh_obj, '--marching_cubes_res', '256']
    result = subprocess.run(command, cwd=NERF_PATH, capture_output=True, text=True, encoding='utf-8')
    if result.returncode == 0 and os.path.exists(output_mesh_obj): yield f"✅ OBJ 형식 메쉬 추출 완료!"
    else: raise RuntimeError(f"메쉬 추출 과정에서 오류가 발생했습니다: {result.stdout} {result.stderr}")
