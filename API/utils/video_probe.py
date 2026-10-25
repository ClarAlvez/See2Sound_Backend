"""Small standalone decoder probe. No AI imports or model loading."""
import json
import math
import sys


def probe(path):
    import cv2
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError('Não foi possível abrir o vídeo. Verifique o arquivo e o codec.')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
        if not ok or frame is None or not frame.size:
            raise ValueError('O vídeo não contém um frame decodificável.')
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('FPS inválido ou não encontrado no vídeo.')
        duration = count / fps
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Não foi possível determinar a duração do vídeo.')
        return {'fps': fps, 'duration_seconds': duration,
                'width': int(frame.shape[1]), 'height': int(frame.shape[0])}
    finally:
        capture.release()


if __name__ == '__main__':
    try:
        print(json.dumps(probe(sys.argv[1]), ensure_ascii=True))
    except ValueError as error:
        print(json.dumps({'error': str(error)}, ensure_ascii=True))
        sys.exit(2)
