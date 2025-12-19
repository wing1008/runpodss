# RunPod Serverless - AnimateDiff Video Generator

AnimateDiff 기반 영상 생성 API (RunPod Serverless)

## 사양

- GPU: 48GB (A6000/A40 권장)
- Base: runpod/base:0.6.0-cuda12.1
- 출력: 2-4초 MP4 영상 (8fps)

## API 사용법

### 요청 형식

```json
{
  "input": {
    "prompt": "a cat walking in the garden, high quality, detailed",
    "negative_prompt": "bad quality, blurry",
    "steps": 20,
    "frames": 16,
    "seed": 12345
  }
}
```

### 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| prompt | string | O | - | 생성할 영상 설명 |
| negative_prompt | string | X | (기본 내장) | 제외할 요소 |
| steps | int | X | 20 | 추론 스텝 (10-30) |
| frames | int | X | 16 | 프레임 수 (8-24) |
| seed | int | X | random | 재현용 시드 |

> width/height는 VRAM 안정성을 위해 512x512로 고정

### 응답 형식

**성공:**
```json
{
  "status": "success",
  "output_path": "/workspace/output.mp4"
}
```

**실패:**
```json
{
  "status": "error",
  "message": "에러 설명"
}
```

## 배포

### 1. Docker 빌드 & 푸시

```bash
docker build -t your-dockerhub/animatediff-runpod:latest .
docker push your-dockerhub/animatediff-runpod:latest
```

### 2. RunPod 설정

1. RunPod Console → Serverless → Create Endpoint
2. Docker Image: `your-dockerhub/animatediff-runpod:latest`
3. GPU: 48GB (A6000/A40)
4. Container Disk: 30GB 이상
5. Idle Timeout: 필요에 따라 설정

## 로컬 테스트

```python
import runpod

# 테스트 이벤트
event = {
    "input": {
        "prompt": "a beautiful sunset over the ocean, cinematic",
        "steps": 20,
        "frames": 16
    }
}

# handler 직접 호출 (테스트용)
from handler import handler
result = handler(event)
print(result)
```

## 주의사항

- 첫 호출 시 모델 로딩에 시간이 걸림 (이후 캐싱됨)
- frames가 많을수록 VRAM 사용량 증가
- 24프레임 초과 시 자동으로 24로 제한
