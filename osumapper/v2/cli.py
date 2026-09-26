from __future__ import annotations

import argparse
import json
from dataclasses import asdict


def parser():
    p=argparse.ArgumentParser(description='Local osu! mapper V2 — isolated from V1; Quadro 6 GB defaults')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--source-db',required=True); q.add_argument('--files-root',required=True); q.add_argument('--data',required=True); q.add_argument('--version',choices=['v2'],default='v2'); q.add_argument('--limit',type=int)
    q.add_argument('--archive-index',action=argparse.BooleanOptionalAction,default=True,help='index selected maps inside _osz without extracting archives')
    q.add_argument('--cache-mel',action='store_true',help='persist full mel arrays; requires substantial additional disk space')
    q.add_argument('--max-metadata-gib',type=float,default=4.0,help='bound the rolling map/audio cache; full-library membership is independent of cache size')
    q.add_argument('--full-dataset',action=argparse.BooleanOptionalAction,default=True,help='index all usable selected downloads with a rotating cache; use a new dataset directory')
    q=sub.add_parser('features'); q.add_argument('checkpoint'); q.add_argument('--data',required=True); q.add_argument('--device',default='auto')
    q.add_argument('--max-gib',type=float,default=8.0); q.add_argument('--min-free-gib',type=float,default=8.0)
    q=sub.add_parser('train'); q.add_argument('--data',required=True); q.add_argument('--run',required=True); q.add_argument('--stage',choices=['audio','mapper'],required=True); q.add_argument('--model',choices=['v2-s','v2-l','tiny'],default='v2-s')
    q.add_argument('--resume'); q.add_argument('--features'); q.add_argument('--device',default='auto'); q.add_argument('--steps',type=int,default=100000); q.add_argument('--hours',type=float)
    q.add_argument('--batch-size',type=int,default=1); q.add_argument('--effective-batch',type=int,default=32); q.add_argument('--workers',type=int,choices=[0,2,4],default=0); q.add_argument('--precision',choices=['fp16','bf16'],default='fp16')
    q.add_argument('--checkpointing',action=argparse.BooleanOptionalAction,default=True); q.add_argument('--compile',action='store_true'); q.add_argument('--validate-every',type=int,default=250); q.add_argument('--sample-every',type=int,default=1000)
    q=sub.add_parser('benchmark'); q.add_argument('--output',required=True); q.add_argument('--compare',default='v2-s,v2-l'); q.add_argument('--device',default='auto'); q.add_argument('--steps',type=int,default=3); q.add_argument('--batches',default='1,2,4,8'); q.add_argument('--hours',type=float,default=2)
    q=sub.add_parser('generate'); q.add_argument('checkpoint'); q.add_argument('input'); q.add_argument('--output',required=True); q.add_argument('--stars',type=float,default=4); q.add_argument('--preset',choices=['Auto','Aim','Streams','Complex Rhythm','Custom'],default='Auto')
    for k in ('aim','streams','rhythm'): q.add_argument('--'+k,type=int,choices=[0,1,2])
    for k in ('ar','od','cs','hp','bpm','offset'): q.add_argument('--'+k,type=float)
    q.add_argument('--timing-map'); q.add_argument('--style-reference'); q.add_argument('--style-strength',type=float,default=.7); q.add_argument('--seed',type=int,default=42)
    q.add_argument('--variants',type=int,default=1); q.add_argument('--candidates',type=int,default=3); q.add_argument('--temperature',type=float,default=.8); q.add_argument('--top-p',type=float,default=.95); q.add_argument('--device',default='auto'); q.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True)
    q=sub.add_parser('evaluate'); q.add_argument('checkpoint'); q.add_argument('--suite',choices=['timing','mapping','styles','full'],default='full'); q.add_argument('--split',choices=['validation','test'],default='test'); q.add_argument('--count',type=int,default=32); q.add_argument('--output'); q.add_argument('--device',default='auto'); q.add_argument('--v1-run')
    q=sub.add_parser('experiment'); q.add_argument('--data',required=True); q.add_argument('--run',required=True); q.add_argument('--device',default='auto')
    q=sub.add_parser('label'); q.add_argument('map_id'); q.add_argument('--data',required=True)
    for k in ('aim','streams','rhythm'): q.add_argument('--'+k,type=int,choices=[0,1,2])
    q.add_argument('--exclude',action='store_true')
    q=sub.add_parser('ui'); q.add_argument('--home',default='local-data-v2'); q.add_argument('--port',type=int,default=7861); q.add_argument('--no-browser',action='store_true')
    q=sub.add_parser('worker',help=argparse.SUPPRESS); q.add_argument('job')
    return p


def main():
    from .jobs import dispatch
    a=parser().parse_args(); d=vars(a).copy(); command=d.pop('command')
    if command=='ui':
        from .ui import launch
        launch(a.home,a.port,not a.no_browser); return
    if command=='worker':
        from .jobs import worker
        worker(a.job); return
    if command=='label':
        from ..data import set_label
        set_label(a.data,a.map_id,{k:getattr(a,k) for k in ('aim','streams','rhythm')},a.exclude); return
    if command=='prepare':
        d.pop('version'); d['destination']=d.pop('data')
    elif command=='features': d['data_root']=d.pop('data'); d['max_gib']=d.pop('max_gib'); d['min_free_gib']=d.pop('min_free_gib')
    elif command=='train':
        from .training import TrainConfig
        cfg=TrainConfig(stage=a.stage,model=a.model,steps=a.steps,hours=a.hours or (24 if a.stage=='audio' else 120),batch_size=a.batch_size,effective_batch=a.effective_batch,workers=a.workers,device=a.device,precision=a.precision,checkpointing=a.checkpointing,compile_model=a.compile,validate_every=a.validate_every,sample_every=a.sample_every)
        d=dict(data_root=a.data,run_dir=a.run,config=asdict(cfg),resume=a.resume,features=a.features)
    elif command=='benchmark':
        d['output_dir']=d.pop('output'); d['compare']=tuple(d['compare'].split(',')); d['batches']=tuple(map(int,d['batches'].split(',')))
    elif command=='generate':
        d=dict(checkpoint_path=d.pop('checkpoint'),inputs=d.pop('input'),output_dir=d.pop('output'),config=d)
    elif command=='evaluate': d['checkpoint_path']=d.pop('checkpoint'); d['output_dir']=d.pop('output')
    elif command=='experiment': d['data_root']=d.pop('data'); d['run_dir']=d.pop('run')
    result=dispatch(command,d)
    if result is not None: print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
