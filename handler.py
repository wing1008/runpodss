import subprocess
import uuid
import runpod

def handler(event):
    try:
        prompt = event["input"].get("prompt", "a cinematic scene")
        frames = min(int(event["input"].get("frames", 16)), 24)

        out_path = f"/workspace/output_{uuid.uuid4().hex}.mp4"

        # 테스트용 더미 영상 생성 (ffmpeg)
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s=512x512:d=2",
            "-vf", "drawtext=text='RUNPOD OK':fontcolor=white:fontsize=48:x=50:y=200",
            out_path
        ], check=True)

        return {
            "status": "success",
            "video_path": out_path,
            "prompt": prompt,
            "frames": frames
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

runpod.serverless.start({"handler": handler})
