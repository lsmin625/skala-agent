# AI Agent by LangGraph

## 파이썬 가상환경 구성

### 1. venv 가상환경 생성

```
python -m venv venv
```

※ 가상환경 구성 실패 (The virtual environment was not created successfully because ensurepip is not
available.) 경우, venv 패키지 설치 후 다시 명령 실행

```
sudo apt install python3.12-venv
```

### 2. venv 가상환경 활성화

```
.\venv\Scripts\activate     # windows
source ./venv/bin/activate  # macOS/Linux
```
### 3. 필수 모듈 설치
```
pip install -r requirements.txt
```