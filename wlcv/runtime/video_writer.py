"""Mac의 OpenCV 또는 Linux의 FFmpeg로 브라우저용 H.264를 저장한다."""
import shutil
import subprocess

import cv2


class FFmpegWriter:
    def __init__(self, executable, path, fps, size):
        width, height = size
        self.encoder = subprocess.Popen(
            [executable, "-nostdin", "-hide_banner", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}",
             "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
             "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-y", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, image):
        self.encoder.stdin.write(image.tobytes())

    def release(self):
        try:
            self.encoder.stdin.close()
        finally:
            code = self.encoder.wait()
        if code:
            raise OSError(f"FFmpeg H.264 인코딩이 종료 코드 {code}로 실패했습니다.")


def open_h264_writer(path, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if writer.isOpened():
        return writer
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise OSError("H.264 저장을 위해 avc1을 지원하는 OpenCV 또는 libx264가 포함된 FFmpeg가 필요합니다.")
    return FFmpegWriter(ffmpeg, path, fps, size)
