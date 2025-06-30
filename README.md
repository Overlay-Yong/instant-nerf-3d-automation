# Instant-NGP 3D Automation Web UI

이 프로젝트는 NVIDIA의 Instant-NGP 기술을 웹 인터페이스를 통해 쉽게 사용할 수 있도록 만든 자동화 애플리케이션입니다. 사용자는 여러 장의 이미지를 업로드하는 것만으로 NeRF 기반의 3D 모델을 생성하고 웹 뷰어로 즉시 확인할 수 있습니다.

## 주요 기능

-   **웹 기반 인터페이스:** 모든 과정을 웹 브라우저에서 처리합니다.
-   **이미지 업로드:** 다중 이미지 업로드를 지원합니다.
-   **자동화된 파이프라인:** COLMAP을 이용한 카메라 자세 추정부터 NeRF 학습, 3D 메쉬 추출까지 모든 과정을 자동으로 실행합니다.
-   **실시간 진행 상황:** Server-Sent Events(SSE)를 통해 모든 처리 과정을 실시간 로그로 확인할 수 있습니다.
-   **3D 모델 뷰어:** Three.js 기반의 웹 뷰어로 생성된 OBJ 모델을 즉시 확인하고 다운로드할 수 있습니다.

## 설치 및 실행 방법

### 1. 사전 요구사항

-   **Python 3.8 이상**
-   **NVIDIA 그래픽 카드** (CUDA 지원)
-   **Instant-NGP:** [공식 가이드](https://github.com/nvlabs/instant-ngp)에 따라 먼저 빌드를 완료해야 합니다.
-   **COLMAP:** [공식 릴리즈 페이지](https://github.com/colmap/colmap/releases)에서 다운로드하여 설치해야 합니다.
-   **Git**

### 2. 설치

1.  **리포지토리 클론:**
    ```bash
    git clone [https://github.com/Overlay-Yong/instant-nerf-3d-automation.git](https://github.com/Overlay-Yong/instant-nerf-3d-automation.git)
    cd instant-nerf-3d-automation
    ```

2.  **가상 환경 생성 및 활성화 (권장):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **필요한 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **`config.py` 설정:**
    -   프로젝트 루트에 있는 `config.py` 파일을 엽니다.
    -   `NERF_PATH` 변수에 로컬에 빌드된 **Instant-NGP 폴더의 전체 경로**를 입력합니다.
    -   `COLMAP_BIN_PATH` 변수에 로컬에 설치된 **COLMAP의 `bin` 폴더 전체 경로**를 입력합니다.

### 3. 애플리케이션 실행

```bash
python app.py
```

서버가 실행되면 웹 브라우저에서 `http://127.0.0.1:5000` 로 접속합니다.

## 사용 방법

1.  웹 페이지의 '파일 선택' 버튼을 클릭하여 3D로 만들고 싶은 객체를 촬영한 이미지(최소 20장 이상 권장)를 모두 선택합니다.
2.  '업로드 및 생성 시작' 버튼을 클릭합니다.
3.  처리 과정이 실시간 로그와 함께 표시됩니다. 모든 과정은 자동으로 진행되며, 수 분에서 수십 분이 소요될 수 있습니다.
4.  완료되면 자동으로 3D 뷰어 페이지로 이동하여 결과물을 확인할 수 있습니다.