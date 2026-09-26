"""Isolated V2 background workers; browser refresh never stops a job."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from ..data import atomic_json,load_json
from ..jobs import pid_alive,cancel_job,job_status


def start_job(home,action,arguments):
    root=Path(home).resolve()/'jobs'; root.mkdir(parents=True,exist_ok=True)
    for p in root.glob('*.status.json'):
        s=load_json(p,{})
        if s.get('status') in ('running','starting') and ((s.get('pid') and pid_alive(s['pid'])) or (not s.get('pid') and time.time()-p.stat().st_mtime<30)):
            raise ValueError('A V2 job is active; wait or cancel it first')
    name=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]; path=root/(name+'.json')
    atomic_json(path,{'action':action,'arguments':arguments}); atomic_json(path.with_suffix('.status.json'),{'status':'starting','action':action})
    with path.with_suffix('.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','osumapper.v2','worker',str(path)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),cwd=Path(__file__).resolve().parents[2],
                                 env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    atomic_json(path.with_suffix('.status.json'),{'status':'running','pid':process.pid,'action':action,'started':time.time()})
    return str(path)


def dispatch(action,args,progress=print,cancelled=lambda:False):
    args=dict(args)
    if action=='prepare':
        from .data import prepare
        return prepare(**args,progress=progress,cancelled=cancelled)
    if action=='features':
        from .features import build_cache
        return build_cache(**args,progress=progress,cancelled=cancelled)
    if action=='train':
        from .training import train,TrainConfig
        args['cfg']=TrainConfig(**args.pop('config'))
        return train(**args,progress=progress,cancelled=cancelled)
    if action=='generate':
        from .generation import generate,GenerationConfig
        args['cfg']=GenerationConfig(**args.pop('config'))
        return generate(**args,progress=progress,cancelled=cancelled)
    if action=='benchmark':
        from .benchmark import benchmark
        return benchmark(**args,progress=progress,cancelled=cancelled)
    if action=='evaluate':
        from .evaluation import evaluate
        return evaluate(**args,progress=progress,cancelled=cancelled)
    if action=='experiment':
        from .experiment import experiment
        return experiment(**args,progress=progress,cancelled=cancelled)
    raise ValueError(f'Unknown V2 job: {action}')


def worker(path):
    path=Path(path); spec=load_json(path); began=time.monotonic()
    def progress(value):
        print(value,flush=True)
        atomic_json(path.with_suffix('.status.json'),{'status':'running','pid':os.getpid(),'action':spec['action'],'message':str(value),'elapsed_seconds':time.monotonic()-began})
    try:
        result=dispatch(spec['action'],spec['arguments'],progress,lambda:path.with_suffix('.cancel').exists())
        status='cancelled' if path.with_suffix('.cancel').exists() else 'complete'
        atomic_json(path.with_suffix('.status.json'),{'status':status,'action':spec['action'],'result':result,'elapsed_seconds':time.monotonic()-began})
    except InterruptedError as exc:
        atomic_json(path.with_suffix('.status.json'),{'status':'cancelled','action':spec['action'],'message':str(exc)})
    except Exception as exc:
        traceback.print_exc(); atomic_json(path.with_suffix('.status.json'),{'status':'failed','action':spec['action'],'error':str(exc)})
        raise
