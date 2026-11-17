"""Isolated Light-ASD inference for See2Sound. Never import the CUDA-only ASD.py.

Upstream: https://github.com/Junhua-Liao/Light-ASD
CPU/GPU supported. Requires torch, opencv-python, scipy, python_speech_features.
The standard OpenCV Haar detector is used for a portable baseline; users can
later swap it for S3FD/RetinaFace. ASD is not lip-reading or speaker verification.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path


def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1])
    x1,y1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0)
    area1=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area2=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(area1+area2-inter,1.)


def extract_frames(cap,start,end,cv2, fps=25):
    native = cap.get(cv2.CAP_PROP_FPS)
    if native <= 0:
        raise RuntimeError('O FPS do video nao pode ser identificado.')
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(start*native))
    frames = []
    target_index = 0
    target_count = max(1,int((end-start)*fps))
    last_image = None
    src_index = int(start*native)
    target_srcs = [round((start+(i+.5)/fps)*native) for i in range(target_count)]
    for idx in range(target_count):
        target = target_srcs[idx]
        while src_index <= target:
            ret,img = cap.read()
            if not ret:
                break
            last_image = img
            src_index += 1
        if last_image is None:
            break
        frames.append((start+(idx+.5)/fps,last_image.copy()))
    return frames


def extract_faces(frames, cv2, detector, *, max_faces=5, min_size=36):
    """Frame-dense face tracking by IoU, simple but bounded and deterministic."""
    tracks=[]
    for fi,(timestamp,image) in enumerate(frames):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        H,W=gray.shape
        boxes = detector.detectMultiScale(gray,scaleFactor=1.13,minNeighbors=4,minSize=(min_size,min_size))
        detections=[]
        for x,y,w,h in boxes:
            bbox=[float(x),float(y),float(x+w),float(y+h)]
            pad=int(.20*max(w,h))
            x0=max(0,x-pad); x1=min(W,x+w+pad)
            y0=max(0,y-pad); y1=min(H,y+h+pad)
            crop=gray[y0:y1,x0:x1]
            if crop.size == 0:
                continue
            crop=cv2.resize(crop,(112,112),interpolation=cv2.INTER_LINEAR)
            detections.append((bbox,crop))
        detections.sort(key=lambda x:(x[0][2]-x[0][0])*(x[0][3]-x[0][1]),reverse=True)
        used=set()
        for bbox,crop in detections[:max_faces]:
            candidates=[]
            for tid,track in enumerate(tracks):
                if tid in used or fi-track['last_frame']>4:
                    continue
                score=iou(bbox,track['bbox'])
                if score >= .20:
                    candidates.append((score,tid))
            if candidates:
                tid=max(candidates)[1]
                track=tracks[tid]
                track['observations'].append((fi,timestamp,bbox,crop))
                track['last_frame']=fi
                track['bbox']=bbox
                used.add(tid)
            else:
                tid=len(tracks)
                tracks.append({'bbox':bbox,'last_frame':fi,'observations':[(fi,timestamp,bbox,crop)]})
                used.add(tid)
    return tracks


def relate_face_to_person(track,frames,scene_intervals):
    """Map a face to an already reconciled person, never by scene presence alone."""
    from collections import Counter
    votes=Counter()
    valid_frames=0
    for idx,t,face_box,crop in track['observations']:
        matches = [(abs(float(f['timestamp'])-t),f) for f in frames if abs(float(f['timestamp'])-t) <= .6]
        if not matches:
            continue
        nearest_dt=min(delta for delta,_ in matches)
        nearest=[f for delta,f in matches if abs(delta-nearest_dt)<1e-6]
        frame=nearest[0]
        # Reject if a scene boundary falls between the evidence frame and target time.
        if any(min(float(frame['timestamp']),t)<float(s.get('end_time',-1))<max(float(frame['timestamp']),t)
               for s in scene_intervals):
            continue
        cx=(face_box[0]+face_box[2])/2
        cy=(face_box[1]+face_box[3])/2
        candidates=[]
        for person in frame.get('people',[]):
            bbox=person['bbox'];eid=person['entity_id']
            if not eid or not bbox or len(bbox)!=4:
                continue
            x0,y0,x1,y1=map(float,bbox)
            if x0<=cx<=x1 and y0<=cy<=y0+.72*(y1-y0):
                facew=face_box[2]-face_box[0]
                if facew > 1.25*(x1-x0):
                    continue
                personcx=(x0+x1)/2
                candidates.append((abs(cx-personcx)/max(x1-x0,1),eid))
        if candidates:
            candidates.sort()
            if len(candidates)>1 and candidates[1][0]-candidates[0][0]<.12:
                continue
            votes[candidates[0][1]]+=1
            valid_frames+=1
    if not votes:
        return None,0.0
    (eid,n),=votes.most_common(1)
    strength=n/max(len(track['observations']),1)
    if n<5 or strength<.55 or (len(votes)>1 and n/sum(votes.values())<.75):
        return None,strength
    return eid,strength


def load_audio(path,scipy_wav,signal,numpy):
    rate, data=scipy_wav.read(path)
    if data.ndim>1:
        data=data.astype(numpy.float64).mean(axis=1)
    if data.size==0:
        return numpy.zeros(0,dtype=numpy.float32)
    data=numpy.asarray(data,dtype=numpy.float32)
    # Resample into the MFCC reference sample rate.
    if rate != 16000:
        gcd=math.gcd(int(rate),16000)
        data=signal.resample_poly(data,16000//gcd,int(rate)//gcd)
    return data


def load_model(root,weights,device,torch):
    sys.path.insert(0,str(root))
    from model.Model import ASD_Model
    import torch.nn as nn
    network=ASD_Model().to(device)
    fc=nn.Linear(128,2).to(device)
    try:
        ckpt=torch.load(str(weights),map_location='cpu',weights_only=True)
    except TypeError:
        ckpt=torch.load(str(weights),map_location='cpu')
    if not isinstance(ckpt,dict):
        raise RuntimeError('Pesos do Light-ASD nao sao um state_dict.')
    # Compatible with official ASD().saveParameters(path).
    network_weights={k[len('model.'):]:v for k,v in ckpt.items() if k.startswith('model.')}
    if not network_weights:
        raise RuntimeError('Checkpoint nao possui chaves model.* esperadas pelo Light-ASD.')
    network.load_state_dict(network_weights,strict=True)
    fc.load_state_dict({'weight':ckpt['lossAV.FC.weight'], 'bias':ckpt['lossAV.FC.bias']},strict=True)
    network.eval();fc.eval()
    return network,fc


def raw_score_for_track(track,frames,audio,sample_rate,network,fc,device,cv2,numpy,torch,psf):
    """Return mean of upstream `lossAV.forward(...,labels=None)` class-1 raw logits."""
    observations=track['observations']
    if len(observations)<12:
        return None,0.0
    frame_start=observations[0][0]
    frame_end=observations[-1][0]
    count=frame_end-frame_start+1
    if count<12:
        return None,0.0
    coverage=len(observations)/count
    if coverage<.65:
        return None,coverage
    seen={fi:crop for fi,t,box,crop in observations}
    sequence=[]
    latest=None
    for fi in range(frame_start,frame_end+1):
        if fi in seen:
            latest=seen[fi]
        if latest is None:
            return None,coverage
        sequence.append(latest)
    times=[frames[frame_start][0],frames[frame_end][0]]
    a=max(0.0,times[0]-.5/25)
    b=times[1]+.5/25
    # `frames` are anchored to original video time; audio is not shifted.
    samples=audio[max(0,int(a*sample_rate)):max(0,int(b*sample_rate))]
    if len(samples)<2000:
        return None,coverage
    features=psf.mfcc(samples,sample_rate,numcep=13,winlen=.025,winstep=.010)
    number=min(len(sequence),len(features)//4)
    if number<12:
        return None,coverage
    sequence=numpy.stack(sequence[:number]).astype('float32')
    feature=features[:number*4].astype('float32')
    with torch.inference_mode():
        fa=torch.from_numpy(feature).unsqueeze(0).to(device)
        fv=torch.from_numpy(sequence).unsqueeze(0).to(device)
        emb_a=network.forward_audio_frontend(fa)
        emb_v=network.forward_visual_frontend(fv)
        logits=fc(network.forward_audio_visual_backend(emb_a,emb_v))
        score=float(logits[:,1].mean().detach().cpu())
    return score,coverage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    data=json.loads(Path(args.input).read_text(encoding='utf8'))
    import numpy as np
    import cv2
    import torch
    from scipy.io import wavfile
    from scipy import signal
    import python_speech_features as psf
    root=Path(data['light_asd_root'])
    device=('cuda' if torch.cuda.is_available() else 'cpu') if data['device']=='auto' else data['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA foi solicitado mas nao esta disponivel.')
    model,fc=load_model(root,Path(data['weights']),device,torch)
    face_file=Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'
    if not face_file.is_file():
        raise RuntimeError('OpenCV nao possui haarcascade_frontalface_default.xml.')
    detector=cv2.CascadeClassifier(str(face_file))
    audio=load_audio(data['audio_path'],wavfile,signal,np)
    cap=cv2.VideoCapture(data['video_path'])
    if not cap.isOpened():
        raise RuntimeError('OpenCV nao conseguiu abrir o video.')
    out=[];diagnostics=[]
    try:
        for task in data['turns']:
            start,end=float(task['start']),float(task['end'])
            timeline=extract_frames(cap,start,end,cv2)
            if len(timeline)<12:
                diagnostics.append({'turn_id':task['turn_id'],'reason':'not_enough_video_frames'})
                continue
            tracks=extract_faces(timeline,cv2,detector)
            scored=[]
            for tid,track in enumerate(tracks):
                eid,identity_coverage=relate_face_to_person(track,data['frames'],data.get('scene_intervals',[]))
                if eid is None:
                    continue
                raw,coverage=raw_score_for_track(track,timeline,audio,16000,model,fc,device,cv2,np,torch,psf)
                if raw is None:
                    continue
                scored.append({'turn_id':task['turn_id'], 'speaker_id':task['speaker_id'],
                               'entity_id':eid, 'timestamp':round((start+end)/2,4),
                               'raw_score':round(raw,4),
                               'face_coverage':round(min(coverage,len(track['observations'])/len(timeline)),4),
                               'track_coverage':round(coverage,4),
                               'face_frames':len(track['observations']),
                               'entity_spatial_coverage':round(identity_coverage,4),
                               'score_type':'uncalibrated_logit', 'face_track':tid})
            # Multiple short face tracks can belong to the same person.
            # Only DISTINCT entities compete for the current utterance.
            best_by_entity = {}
            for row in scored:
                existing = best_by_entity.get(row['entity_id'])
                if existing is None or existing['raw_score'] < row['raw_score']:
                    best_by_entity[row['entity_id']] = row
            scored = sorted(best_by_entity.values(),key=lambda x:x['raw_score'],reverse=True)
            for row in scored:
                runner=[x['raw_score'] for x in scored if x['entity_id']!=row['entity_id']]
                row['runner_up_raw_score']=max(runner) if runner else -999.0
                row.pop('face_track',None)
                out.append(row)
            if not scored:
                diagnostics.append({'turn_id':task['turn_id'],'reason':'no_scored_face_track'})
    finally:
        cap.release()
    result={'available':True,'source':'light_asd','model_weights':Path(data['weights']).name,
            'score_type':'uncalibrated_logit','turn_scores':out, 'diagnostics':diagnostics,
            'turns_requested':len(data['turns']),'turns_scored':len({x['turn_id'] for x in out}),
            'device':str(device),'face_detector':'opencv_haar'}
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')

if __name__=='__main__':
    main()
