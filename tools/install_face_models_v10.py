"""Download OPCIONAL dos pesos oficiais OpenCV Zoo YuNet e SFace.

Nenhum peso de terceiros e distribuido no overlay; origem OpenCV Zoo.
"""
from __future__ import annotations
from pathlib import Path
from urllib.request import Request,urlopen
import argparse
import tempfile
import os

MODELS={
 'face_detection_yunet_2023mar.onnx':(
    'models/face_detection_yunet/face_detection_yunet_2023mar.onnx',100_000),
 'face_recognition_sface_2021dec.onnx':(
    'models/face_recognition_sface/face_recognition_sface_2021dec.onnx',10_000_000),
}


def download(directory, force=False):
    dest=Path(directory)
    dest.mkdir(parents=True,exist_ok=True)
    for name,(remote,min_bytes) in MODELS.items():
        final=dest/name
        if final.exists() and final.stat().st_size>=min_bytes and not force:
            print('Ja existe:',final)
            continue
        base=['https://media.githubusercontent.com/media/opencv/opencv_zoo/main/',
              'https://github.com/opencv/opencv_zoo/raw/refs/heads/main/']
        reason=[]
        for prefix in base:
            url=prefix+remote
            tmp=None
            try:
                req=Request(url,headers={'User-Agent':'See2Sound-CharacterGraph/1.0'})
                with urlopen(req,timeout=90) as response:
                    with tempfile.NamedTemporaryFile(dir=dest,delete=False,suffix='.download') as out:
                        tmp=Path(out.name)
                        while True:
                            chunk=response.read(1024*1024)
                            if not chunk:break
                            out.write(chunk)
                head=tmp.read_bytes()[:160]
                if tmp.stat().st_size<min_bytes or b'git-lfs' in head or head.lstrip().startswith(b'<'):
                    raise ValueError('Arquivo incompleto, ponteiro LFS ou HTML.')
                os.replace(tmp,final)
                print('OK:',final, '('+str(final.stat().st_size)+' bytes)')
                break
            except Exception as exc:
                reason.append(f'{type(exc).__name__}: {exc}')
                if tmp is not None:tmp.unlink(missing_ok=True)
        else:
            raise RuntimeError(f'Falha download de {name}: '+ '; '.join(reason))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',default='data/models/Face')
    p.add_argument('--force',action='store_true')
    a=p.parse_args()
    download(a.directory,a.force)
