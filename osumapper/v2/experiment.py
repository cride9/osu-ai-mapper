"""Explicitly launched, resumable seven-day experiment. No automatic V1 takeover."""
from __future__ import annotations

import time
from pathlib import Path

from ..data import atomic_json,load_json
from .benchmark import benchmark,overfit,benchmark_io
from .features import build_cache
from .training import train,TrainConfig
from .evaluation import evaluate


def select_model(small,large,small_panel,large_panel):
    """Large must demonstrate both loss benefit and non-regressing valid maps."""
    if not large or large['best_validation']>.97*small['best_validation']: return 'v2-s'
    def measures(panel):
        if not panel: return None
        maps=[m for m in panel.get('maps',[]) if 'archive' in m]
        if not maps: return None
        boundary=[m['boundary_style_change'] for m in maps if m.get('boundary_style_change') is not None]
        if not boundary: return None
        return panel.get('valid_export_fraction',0),sum(boundary)/len(boundary)
    s,l=measures(small_panel),measures(large_panel)
    if s is None or l is None or l[0]<s[0] or l[1]>s[1]: return 'v2-s'
    return 'v2-l'


def experiment(data_root,run_dir,device='auto',progress=print,cancelled=lambda:False):
    root=Path(run_dir).resolve(); root.mkdir(parents=True,exist_ok=True)
    path=root/'experiment.json'; state=load_json(path,{'version':'v2','active_budget_hours':168,'phases':{},'quality_verified':False})
    def phase(name,function,counts=True):
        if state['phases'].get(name,{}).get('status')=='complete': return state['phases'][name]['result']
        if cancelled(): raise InterruptedError('Experiment cancelled; relaunch the same experiment directory')
        state.update(status='running',phase=name); atomic_json(path,state); progress(f'Experiment: {name}')
        tick=time.monotonic(); result=function()
        if cancelled() or (isinstance(result,dict) and result.get('status')=='cancelled'): raise InterruptedError('Experiment paused at a resumable phase')
        state['phases'][name]={'status':'complete','seconds':time.monotonic()-tick,'counts_toward_active_budget':counts,'result':result}
        atomic_json(path,state)
        return result
    bench=phase('preflight',lambda:benchmark(root/'benchmark',device=device,hours=1.5,progress=progress,cancelled=cancelled))
    if 'audio/v2-s' not in bench['recommended'] or 'mapper/v2-s' not in bench['recommended']:
        raise RuntimeError('Small V2 or audio stage failed the 6 GB memory gate; no long training launched')
    fit=phase('overfit',lambda:overfit(data_root,root/'overfit',device=device,progress=progress,cancelled=cancelled))
    if not fit['passed']: raise RuntimeError('Small-data overfit gate failed; inspect overfit/report.json')
    def run_training(stage,name,folder,hours):
        recommended=bench['recommended'][f'{stage}/{name}']
        cfg=TrainConfig(stage=stage,model=name,hours=hours,batch_size=recommended['batch_size'],checkpointing=recommended['checkpointing'],device=device)
        latest=folder/'last.pt'
        return train(data_root,folder,cfg,str(latest) if latest.exists() else None,progress=progress,cancelled=cancelled)
    audio_run=root/'audio'
    phase('audio',lambda:run_training('audio','v2-s',audio_run,24))
    phase('features',lambda:build_cache(data_root,audio_run/'best.pt',device,progress,cancelled),counts=False)
    io=phase('data-loader',lambda:benchmark_io(data_root,root/'benchmark',progress=progress,cancelled=cancelled))
    pilots={}; panels={}
    for name in ('v2-s','v2-l'):
        if f'mapper/{name}' not in bench['recommended']: continue
        folder=root/name
        # Reserve up to two of each eight pilot hours for actual full-song checks.
        def pilot(n=name,f=folder):
            r=bench['recommended'][f'mapper/{n}']
            cfg=TrainConfig(stage='mapper',model=n,hours=6,batch_size=r['batch_size'],checkpointing=r['checkpointing'],workers=io['recommended_workers'],device=device)
            return train(data_root,f,cfg,str(f/'last.pt') if (f/'last.pt').exists() else None,progress=progress,cancelled=cancelled)
        pilots[name]=phase(name+'-pilot',pilot)
        def panel(f=folder):
            began=time.monotonic()
            try:
                return evaluate(f/'best.pt',suite='full',split='validation',count=16,output_dir=f/'pilot-panel',device=device,progress=progress,cancelled=lambda:cancelled() or time.monotonic()-began>2*3600)
            except InterruptedError:
                if cancelled(): raise
                return {'status':'panel_budget_exhausted','maps':[],'quality_verified':False}
        panels[name]=phase(name+'-panel',panel)
    selected=select_model(pilots['v2-s'],pilots.get('v2-l'),panels.get('v2-s'),panels.get('v2-l'))
    state['selected_model']=selected; state['selection_reason']='Large requires ≥3% validation improvement and no full-song panel regression; otherwise small.'; atomic_json(path,state)
    result=phase('main',lambda:run_training('mapper',selected,root/selected,126))
    def final_panel():
        began=time.monotonic()
        try: return evaluate(root/selected/'best.pt',suite='full',count=32,device=device,progress=progress,cancelled=lambda:cancelled() or time.monotonic()-began>6*3600)
        except InterruptedError:
            if cancelled(): raise
            return {'status':'evaluation_budget_exhausted','quality_verified':False}
    phase('final-evaluation',final_panel)
    state.update(status='complete',checkpoint=result['checkpoint'],quality_verified=False,human_playtest_required=True)
    atomic_json(path,state); progress(state)
    return state
