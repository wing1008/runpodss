"""
RunPod Serverless Handler - AnimateDiff Video Generation
JI AI 영상 생성 파이프라인용
- 실제 AI 영상 생성 (더미 영상 제거)
- AnimateDiff + Stable Diffusion 기반
- T2V (Text-to-Video) + I2V (Image-to-Video) 지원

v2.0: I2V 모드 추가 (키프레임 → 비디오)
"""

import os
import gc
import uuid
import base64
import logging
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import runpod
import requests
from typing import Dict, Any, Optional, List, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 파이프라인 (lazy loading)
_pipeline_t2v = None
_pipeline_i2v = None


def get_device():
    """사용 가능한 디바이스 반환"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline_t2v():
    """AnimateDiff T2V 파이프라인 로드 (lazy loading)"""
    global _pipeline_t2v

    if _pipeline_t2v is not None:
        return _pipeline_t2v

    logger.info("Loading AnimateDiff T2V pipeline...")

    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

        device = get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"Device: {device}, dtype: {dtype}")

        # Motion Adapter 로드
        logger.info("Loading motion adapter...")
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=dtype
        )

        # 파이프라인 로드 (Realistic Vision 사용)
        logger.info("Loading base model...")
        pipe = AnimateDiffPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter=adapter,
            torch_dtype=dtype,
        )

        # 스케줄러 설정 (DDIM - 빠른 추론)
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
        )

        # 메모리 최적화
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing("auto")
        logger.info("Attention slicing enabled")

        # GPU로 이동
        pipe = pipe.to(device)

        _pipeline_t2v = pipe
        logger.info("T2V Pipeline loaded successfully")
        return _pipeline_t2v

    except Exception as e:
        logger.error(f"Failed to load T2V pipeline: {e}")
        raise


def load_pipeline_i2v():
    """AnimateDiff I2V 파이프라인 로드 (img2vid)"""
    global _pipeline_i2v

    if _pipeline_i2v is not None:
        return _pipeline_i2v

    logger.info("Loading AnimateDiff I2V pipeline...")

    try:
        from diffusers import (
            AnimateDiffVideoToVideoPipeline,
            MotionAdapter,
            DDIMScheduler
        )

        device = get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Motion Adapter 로드
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=dtype
        )

        # I2V 파이프라인 로드
        pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter=adapter,
            torch_dtype=dtype,
        )

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
        )

        # 메모리 최적화
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing("auto")

        pipe = pipe.to(device)

        _pipeline_i2v = pipe
        logger.info("I2V Pipeline loaded successfully")
        return _pipeline_i2v

    except Exception as e:
        logger.error(f"Failed to load I2V pipeline: {e}")
        raise


def load_image_from_input(image_input: Union[str, bytes]) -> Image.Image:
    """
    다양한 입력 형식에서 이미지 로드

    Args:
        image_input: base64 문자열, URL, 또는 바이트

    Returns:
        PIL.Image: 로드된 이미지
    """
    if isinstance(image_input, bytes):
        return Image.open(BytesIO(image_input)).convert("RGB")

    if isinstance(image_input, str):
        # base64 문자열인 경우
        if image_input.startswith("data:image"):
            # data URL 형식
            base64_data = image_input.split(",")[1]
            image_bytes = base64.b64decode(base64_data)
            return Image.open(BytesIO(image_bytes)).convert("RGB")

        elif len(image_input) > 200 and not image_input.startswith("http"):
            # 순수 base64 문자열
            try:
                image_bytes = base64.b64decode(image_input)
                return Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception:
                pass

        # URL인 경우
        if image_input.startswith("http"):
            response = requests.get(image_input, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        # 파일 경로인 경우
        if os.path.exists(image_input):
            return Image.open(image_input).convert("RGB")

    raise ValueError(f"Cannot load image from input: {type(image_input)}")


def image_to_video_frames(
    image: Image.Image,
    num_frames: int = 16,
    motion_strength: float = 0.1
) -> List[Image.Image]:
    """
    단일 이미지를 비디오 프레임 시퀀스로 변환
    (I2V 파이프라인 입력용)

    Args:
        image: 입력 이미지
        num_frames: 생성할 프레임 수
        motion_strength: 모션 강도 (미사용, 참고용)

    Returns:
        List[Image]: 프레임 리스트 (동일 이미지 반복)
    """
    # AnimateDiff I2V는 첫 프레임을 참조로 사용하여 모션 생성
    # 입력으로 동일 이미지 시퀀스 제공
    return [image.copy() for _ in range(num_frames)]


def generate_video_t2v(
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 20,
    num_frames: int = 16,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    fps: int = 8
) -> tuple:
    """
    AnimateDiff T2V (Text-to-Video)로 영상 생성

    Args:
        prompt: 영상 설명 프롬프트
        negative_prompt: 제외할 요소
        steps: 추론 스텝 (10-30)
        num_frames: 프레임 수 (8-24)
        width: 너비 (512 고정)
        height: 높이 (512 고정)
        guidance_scale: CFG 스케일
        seed: 재현용 시드
        fps: 출력 FPS

    Returns:
        tuple: (생성된 영상 경로, 사용된 시드)
    """
    from diffusers.utils import export_to_video

    # 파라미터 제한 (VRAM 보호)
    width = 512
    height = 512
    num_frames = min(max(num_frames, 8), 24)
    steps = min(max(steps, 10), 30)

    # 기본 네거티브 프롬프트
    if not negative_prompt:
        negative_prompt = (
            "bad quality, worst quality, low resolution, blurry, "
            "watermark, text, logo, deformed, ugly, duplicate, "
            "disfigured, poorly drawn, bad anatomy, wrong anatomy"
        )

    # 파이프라인 로드
    pipe = load_pipeline_t2v()
    device = get_device()

    # 시드 설정
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)

    logger.info(f"T2V: frames={num_frames}, steps={steps}, seed={seed}")
    logger.info(f"Prompt: {prompt[:100]}...")

    # 영상 생성
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    # 출력 디렉토리 생성
    os.makedirs("/tmp/output", exist_ok=True)
    output_path = f"/tmp/output/video_{uuid.uuid4().hex}.mp4"

    # 영상으로 내보내기
    export_to_video(output.frames[0], output_path, fps=fps)

    # 메모리 정리
    del output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"T2V video saved: {output_path}")
    return output_path, seed


def generate_video_i2v(
    image_input: Union[str, bytes],
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 20,
    num_frames: int = 16,
    guidance_scale: float = 7.5,
    strength: float = 0.8,
    seed: Optional[int] = None,
    fps: int = 8,
    motion_type: str = "default"
) -> tuple:
    """
    AnimateDiff I2V (Image-to-Video)로 키프레임에서 영상 생성

    Args:
        image_input: 입력 이미지 (base64, URL, 경로, 바이트)
        prompt: 영상 설명 프롬프트
        negative_prompt: 제외할 요소
        steps: 추론 스텝 (10-30)
        num_frames: 프레임 수 (8-24)
        guidance_scale: CFG 스케일
        strength: denoising 강도 (0.0-1.0, 낮을수록 원본 유지)
        seed: 재현용 시드
        fps: 출력 FPS
        motion_type: 모션 타입 ("default", "slow", "fast")

    Returns:
        tuple: (생성된 영상 경로, 사용된 시드)
    """
    from diffusers.utils import export_to_video

    # 이미지 로드
    logger.info("Loading input image for I2V...")
    input_image = load_image_from_input(image_input)

    # 이미지 크기 조정 (512x512로 고정)
    input_image = input_image.resize((512, 512), Image.LANCZOS)

    # 파라미터 제한
    num_frames = min(max(num_frames, 8), 24)
    steps = min(max(steps, 10), 30)
    strength = min(max(strength, 0.3), 1.0)  # 너무 낮으면 변화 없음

    # 모션 강도 조정
    motion_multipliers = {
        "slow": 0.5,
        "default": 1.0,
        "fast": 1.5
    }
    motion_mult = motion_multipliers.get(motion_type, 1.0)

    # 기본 네거티브 프롬프트
    if not negative_prompt:
        negative_prompt = (
            "bad quality, worst quality, low resolution, blurry, "
            "watermark, text, logo, deformed, ugly, duplicate, "
            "disfigured, poorly drawn, bad anatomy, wrong anatomy, "
            "static, no motion, frozen"
        )

    # I2V 파이프라인 로드
    pipe = load_pipeline_i2v()
    device = get_device()

    # 시드 설정
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)

    logger.info(f"I2V: frames={num_frames}, steps={steps}, strength={strength}, seed={seed}")
    logger.info(f"Prompt: {prompt[:100]}...")

    # 비디오 프레임 시퀀스 생성 (입력 이미지 반복)
    video_frames = image_to_video_frames(input_image, num_frames)

    # 영상 생성
    with torch.inference_mode():
        output = pipe(
            video=video_frames,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        )

    # 출력 디렉토리 생성
    os.makedirs("/tmp/output", exist_ok=True)
    output_path = f"/tmp/output/video_i2v_{uuid.uuid4().hex}.mp4"

    # 영상으로 내보내기
    export_to_video(output.frames[0], output_path, fps=fps)

    # 메모리 정리
    del output
    del video_frames
    del input_image
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"I2V video saved: {output_path}")
    return output_path, seed


# Backward compatibility alias
def generate_video(
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 20,
    num_frames: int = 16,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
) -> tuple:
    """기존 generate_video 호환성 유지 (T2V로 라우팅)"""
    return generate_video_t2v(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        num_frames=num_frames,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        seed=seed
    )


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless 핸들러 - T2V/I2V 지원

    Input (T2V - Text-to-Video):
    {
        "input": {
            "mode": "t2v",                      # optional, default "t2v"
            "prompt": "a woman walking in the city, cinematic",
            "negative_prompt": "bad quality",  # optional
            "steps": 20,                        # optional, 10-30
            "frames": 16,                       # optional, 8-24
            "guidance_scale": 7.5,              # optional
            "seed": 12345,                      # optional
            "fps": 8                            # optional
        }
    }

    Input (I2V - Image-to-Video):
    {
        "input": {
            "mode": "i2v",
            "image": "<base64 or URL>",         # required for i2v
            "prompt": "a woman walking, cinematic motion",
            "negative_prompt": "bad quality",   # optional
            "steps": 20,                        # optional
            "frames": 16,                       # optional
            "guidance_scale": 7.5,              # optional
            "strength": 0.8,                    # optional, denoising strength
            "motion_type": "default",           # optional: "slow", "default", "fast"
            "seed": 12345,                      # optional
            "fps": 8                            # optional
        }
    }

    Output (성공):
    {
        "status": "success",
        "mode": "t2v" or "i2v",
        "video_path": "/tmp/output/video_xxx.mp4",
        "seed": 12345,
        "frames": 16,
        "steps": 20
    }

    Output (실패):
    {
        "status": "error",
        "message": "에러 설명"
    }
    """
    try:
        job_input = event.get("input", {})

        # 모드 확인 (기본값: t2v)
        mode = job_input.get("mode", "t2v").lower()

        # 필수 파라미터
        prompt = job_input.get("prompt")
        if not prompt:
            return {
                "status": "error",
                "message": "Missing required field: prompt"
            }

        # 공통 파라미터
        negative_prompt = job_input.get("negative_prompt")
        steps = int(job_input.get("steps", 20))
        num_frames = int(job_input.get("frames", 16))
        guidance_scale = float(job_input.get("guidance_scale", 7.5))
        fps = int(job_input.get("fps", 8))
        seed = job_input.get("seed")
        if seed is not None:
            seed = int(seed)

        # I2V 모드
        if mode == "i2v":
            image_input = job_input.get("image")
            if not image_input:
                return {
                    "status": "error",
                    "message": "Missing required field for i2v: image"
                }

            strength = float(job_input.get("strength", 0.8))
            motion_type = job_input.get("motion_type", "default")

            output_path, used_seed = generate_video_i2v(
                image_input=image_input,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed,
                fps=fps,
                motion_type=motion_type
            )

        # T2V 모드 (기본)
        else:
            output_path, used_seed = generate_video_t2v(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                seed=seed,
                fps=fps
            )

        # 비디오 파일을 base64로 인코딩하여 반환
        video_base64 = None
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                video_base64 = base64.b64encode(f.read()).decode("utf-8")
            # 임시 파일 삭제
            os.remove(output_path)

        return {
            "status": "success",
            "mode": mode,
            "video_path": output_path,
            "video_base64": video_base64,
            "seed": used_seed,
            "frames": num_frames,
            "steps": steps,
            "fps": fps
        }

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        logger.error("CUDA OOM")
        return {
            "status": "error",
            "message": "CUDA out of memory. Try reducing frames or steps."
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "error",
            "message": f"Generation failed: {str(e)}"
        }


# ================== 헬스 체크 ==================

def health_check() -> Dict[str, Any]:
    """헬스 체크 응답"""
    return {
        "status": "healthy",
        "version": "2.0",
        "modes": ["t2v", "i2v"],
        "cuda_available": torch.cuda.is_available(),
        "device": get_device()
    }


# RunPod 서버리스 시작
runpod.serverless.start({"handler": handler})
