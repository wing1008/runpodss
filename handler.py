"""
RunPod Serverless Handler for AnimateDiff Video Generation
- AnimateDiff + Stable Diffusion 1.5 based
- Generates 2-4 second videos (16-24 frames)
- Output: /workspace/output.mp4
"""

import os
import torch
import runpod
from typing import Dict, Any, Optional

# Global pipeline (lazy loading)
_pipeline = None


def load_pipeline():
    """Load AnimateDiff pipeline with memory optimization."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from diffusers.utils import export_to_video

    # Load motion adapter
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2",
        torch_dtype=torch.float16
    )

    # Load pipeline with motion adapter
    # Using Realistic Vision for better quality
    pipe = AnimateDiffPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )

    # Use DDIM scheduler for faster inference
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
    )

    # Memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # Move to GPU
    pipe = pipe.to("cuda")

    # Enable memory efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass  # xformers not available, continue without it

    _pipeline = pipe
    return _pipeline


def generate_video(
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 20,
    frames: int = 16,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
) -> str:
    """Generate video using AnimateDiff."""
    from diffusers.utils import export_to_video

    # Enforce limits to prevent VRAM explosion
    width = 512
    height = 512
    frames = min(max(frames, 8), 24)
    steps = min(max(steps, 10), 30)

    # Default negative prompt
    if not negative_prompt:
        negative_prompt = (
            "bad quality, worst quality, low resolution, blurry, "
            "watermark, text, logo, deformed, ugly, duplicate"
        )

    # Load pipeline
    pipe = load_pipeline()

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Generate frames
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=frames,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=7.5,
            generator=generator,
        )

    # Export to video
    output_path = "/workspace/output.mp4"
    os.makedirs("/workspace", exist_ok=True)

    export_to_video(output.frames[0], output_path, fps=8)

    # Clear cache
    torch.cuda.empty_cache()

    return output_path


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler function.

    Input event format:
    {
        "input": {
            "prompt": "a cat walking in the garden",
            "negative_prompt": "bad quality",  # optional
            "steps": 20,                        # optional, default 20
            "frames": 16,                       # optional, default 16, max 24
            "width": 512,                       # fixed at 512
            "height": 512,                      # fixed at 512
            "seed": 12345                       # optional
        }
    }

    Output format:
    {
        "status": "success",
        "output_path": "/workspace/output.mp4"
    }
    or
    {
        "status": "error",
        "message": "error description"
    }
    """
    try:
        # Extract input
        job_input = event.get("input", {})

        # Validate required fields
        prompt = job_input.get("prompt")
        if not prompt:
            return {
                "status": "error",
                "message": "Missing required field: prompt"
            }

        # Extract optional parameters with defaults
        negative_prompt = job_input.get("negative_prompt", None)
        steps = int(job_input.get("steps", 20))
        frames = int(job_input.get("frames", 16))
        width = 512  # Fixed
        height = 512  # Fixed
        seed = job_input.get("seed", None)
        if seed is not None:
            seed = int(seed)

        # Generate video
        output_path = generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            frames=frames,
            width=width,
            height=height,
            seed=seed,
        )

        return {
            "status": "success",
            "output_path": output_path
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "status": "error",
            "message": "CUDA out of memory. Try reducing frames or steps."
        }

    except Exception as e:
        torch.cuda.empty_cache()
        return {
            "status": "error",
            "message": f"Generation failed: {str(e)}"
        }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
