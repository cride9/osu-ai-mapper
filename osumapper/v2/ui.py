"""Local-only V2 interface; separate port/home from the working V1."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from .. import audio
from ..data import load_json,set_label
from .data import readonly,read_cached_map
from .jobs import start_job,cancel_job,job_status
from .storage import decode_audio

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')


def build(home):
    import gradio as gr
    from ..preview import plot_map
    from .training import TrainConfig
    from .generation import GenerationConfig
    home=Path(home).resolve(); home.mkdir(parents=True,exist_ok=True)
    previous=sorted((home/'jobs').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True) if (home/'jobs').exists() else []
    previous=next((str(p) for p in previous if not p.name.endswith('.status.json')), '')
    def job(action,**kwargs):
        try: return start_job(home,action,kwargs)
        except (ValueError,RuntimeError) as exc: raise gr.Error(str(exc))
    def review(data,cursor):
        manifest=load_json(Path(data)/'manifest.json')
        if not manifest: raise gr.Error('Először készítsd elő a V2-adatokat.')
        db=readonly(manifest['index'])
        try:
            count=db.execute('SELECT COUNT(*) FROM records').fetchone()[0]; cursor=max(0,min(count-1,int(cursor)))
            raw=db.execute('SELECT payload FROM records ORDER BY id LIMIT 1 OFFSET ?',(cursor,)).fetchone()
        finally: db.close()
        row=json.loads(raw[0]); labels=load_json(Path(data)/'labels.json',{}).get(row['id'],{})
        styles=labels.get('styles',row['styles']); bm=read_cached_map(data,row)
        pcm=decode_audio(row['audio_path']); at=max(0,bm.objects[0].time/1000-.5); clip=pcm[round(at*audio.SR):round((at+30)*audio.SR)]
        return cursor,row['id'],f"{cursor+1}/{count} · {row['artist']} — {row['title']} · {row['stars']:.2f}★\n\nCímkeforrás: {'kézi' if labels else row['label_provenance']}", (audio.SR,clip),plot_map(bm,at),*[('Auto / unknown' if styles[k] is None else ['Low','Medium','High'][styles[k]]) for k in ('aim','streams','rhythm')]
    def save(data,mid,aim,streams,rhythm,exclude=False):
        if not mid: raise gr.Error('Tölts be egy mapot.')
        values={k:None if v=='Auto / unknown' else ['Low','Medium','High'].index(v) for k,v in zip(('aim','streams','rhythm'),(aim,streams,rhythm))}
        set_label(data,mid,values,exclude)
        return 'Mentve. Az új futások használják a módosítást; a futó dataset változatlan.'
    def train_job(data,run,stage,size,hours,batch,workers,resume):
        cfg=TrainConfig(stage=stage,model=size,hours=float(hours),batch_size=int(batch),workers=int(workers))
        return job('train',data_root=data,run_dir=run,config=asdict(cfg),resume=resume.strip() or None)
    def generate_job(checkpoint,folder,files,out,stars,preset,aim,streams,rhythm,seed,variants,candidates,bpm,offset,timing,reference,strength,ar,od,cs,hp):
        level=lambda v:None if v=='Auto / unknown' else ['Low','Medium','High'].index(v)
        cfg=GenerationConfig(stars=float(stars),preset=preset,aim=level(aim),streams=level(streams),rhythm=level(rhythm),seed=int(seed),variants=int(variants),candidates=int(candidates),bpm=bpm,offset=offset,timing_map=timing.strip() or None,style_reference=reference.strip() or None,style_strength=float(strength),ar=ar,od=od,cs=cs,hp=hp)
        inputs=files or folder.strip()
        if not inputs: raise gr.Error('Válassz zenét vagy egy mappát.')
        return job('generate',checkpoint_path=checkpoint,inputs=inputs,output_dir=out,config=asdict(cfg))
    def status(path,run):
        state,log=job_status(path)
        metrics=load_json(Path(run)/'status.json',{})
        elapsed=metrics.get('elapsed_seconds',0); limit=metrics.get('hours_target',120)*3600; step=metrics.get('step',0)
        pct=min(100,100*max(step/max(1,metrics.get('step_target',100000)),elapsed/max(1,limit)))
        note=f"{metrics.get('status','Nincs futás')} · {metrics.get('stage','')} · step {step:,} · {elapsed/3600:.2f} óra · {pct:.1f}%"
        if metrics.get('eta_seconds') is not None: note+=f" · hátralévő kb. {metrics['eta_seconds']/3600:.2f} óra"
        return state,log,note,metrics
    def preview_timing(song,bpm,offset,timing,out):
        from .timing import estimate,metronome
        from ..mapio import read
        if timing.strip(): points=[p for p in read(timing).timing if p.uninherited]
        elif bpm: points,_=estimate(__import__('numpy').zeros((1,2)),bpm=bpm,offset=offset)
        else: raise gr.Error('Adj meg BPM-et és offsetet, vagy timing referenciafájlt.')
        dest=Path(out)/'timing-preview.wav'; dest.parent.mkdir(parents=True,exist_ok=True)
        return metronome(song,points,dest),{'mode':'manual' if bpm else 'reference','points':len(points)}
    def preview_automatic(song,checkpoint,out):
        import numpy as np
        from .features import encode_recording,load_encoder
        from .runtime import device_for,gpu_session
        from .timing import estimate,metronome
        from ..data import digest
        if not song or not checkpoint: raise gr.Error('Adj meg zenét és saját V2 audiócheckpointot.')
        encoder,_=load_encoder(checkpoint)
        actual=device_for('auto'); path=Path(out)/'timing-preview'/digest(song)[:16]
        path.mkdir(parents=True,exist_ok=True)
        with gpu_session(actual,home):
            mel=audio.spectrogram(audio.decode(song))
            _,scores=encode_recording(encoder.to(actual),mel,path,actual,torch.float16)
        points,report=estimate(scores,strict=False)
        if not points: return None,report
        return metronome(song,points,path/'automatic-metronome.wav'),report
    import torch
    with gr.Blocks(title='osu! AI Mapper V2') as app:
        gr.Markdown('# osu! AI Mapper V2\nHelyi tanítás és generálás · Quadro 6 GB · a V1 külön marad')
        dataset=gr.Textbox(value=str(home/'dataset-full'),label='V2-adatkészlet mappája')
        current_job=gr.Textbox(label='Aktuális háttérfeladat',value=previous)
        with gr.Tab('Library'):
            source=gr.Textbox(label='A scriptek SQLite-adatbázisa',value='C:/Users/cride/Documents/Github/osu_dataset/osu_dataset.sqlite')
            files_root=gr.Textbox(label='Letöltött .osu, audio és _osz archívumok gyökérmappája',value='C:/Users/cride/Documents/Github/osu_dataset/osu_files')
            metadata_budget=gr.Number(value=4,label='Cserélhető map/audio-cache kerete (GiB) - nem maplimit')
            gr.Markdown('Teljes gyűjtemény: minden használható, kiválasztott letöltés bekerül az indexbe. A tömörített cache cserélődik; cache-hiánynál újraszámolás történik. Régi futáshoz hagyd meg a régi dataset útvonalát. Az új teljes datasethez új audio- és mapperfutás szükséges.')
            gr.Markdown('Csak a végleges dataset-tábla kiválasztott sorai kerülnek be. Az _osz archívumok közvetlenül olvashatók, kibontás és további audiómásolat nélkül.')
            prepare=gr.Button('V2-adatok ellenőrzése és előkészítése',variant='primary')
            report=gr.JSON(label='Lefedettség és kizárások'); refresh_report=gr.Button('Riport frissítése')
        with gr.Tab('Labeling'):
            cursor=gr.Number(value=0,precision=0,label='Map sorszáma (0-tól)'); mid=gr.Textbox(visible=False)
            heading=gr.Markdown(); music=gr.Audio(label='30 másodperces részlet'); pattern=gr.Plot(label='Mintanézet')
            options=['Auto / unknown','Low','Medium','High']
            with gr.Row():
                la=gr.Radio(options,value=options[0],label='Aim'); ls=gr.Radio(options,value=options[0],label='Streams'); lr=gr.Radio(options,value=options[0],label='Rhythm')
            with gr.Row():
                load=gr.Button('Betöltés'); accept=gr.Button('Accept'); skip=gr.Button('Skip'); exclude=gr.Button('Exclude')
            label_note=gr.Markdown()
        with gr.Tab('Training'):
            gr.Markdown('1. Memóriapróba → 2. Audiótanítás → 3. Feature-cache → 4. Mapper. Teljes kísérlet külön gombbal indítható.')
            run=gr.Textbox(value=str(home/'runs'/'audio-full'),label='Futás mappája')
            with gr.Row():
                stage=gr.Dropdown(['audio','mapper'],value='audio',label='Fázis'); size=gr.Dropdown(['v2-s','v2-l'],value='v2-s',label='Modell')
                hours=gr.Number(value=24,label='Teljes aktív időkeret (óra)'); batch=gr.Dropdown([1,2,4,8],value=1,label='Fizikai batch'); workers=gr.Dropdown([0,2,4],value=0,label='Adatbetöltő workers')
            resume=gr.Textbox(label='Folytatás: last.pt teljes útvonala',value='')
            bench=gr.Button('6 GB-os memóriapróba'); train_button=gr.Button('Tanítás / folytatás',variant='primary')
            encoder_checkpoint=gr.Textbox(value=str(home/'runs'/'audio-full'/'best.pt'),label='Saját audiócheckpoint a cache-hez')
            with gr.Row():
                feature_budget=gr.Number(value=8,label='Audiófeature-cache felső határa (GiB)')
                free_reserve=gr.Number(value=8,label='Mindig szabadon hagyott tárhely (GiB)')
            features_button=gr.Button('Audiófeature-cache beállítása / elkészítése')
            gr.Markdown('Teljes datasetnél minden felvétel használható marad a feature-cache méretétől függetlenül. A hiányzó feature-ök CPU-n készülnek; ez lassíthatja a mapper adatbetöltését.')
            experiment_root=gr.Textbox(value=str(home/'runs'/'experiment-full'),label='Hétnapos kísérlet mappája')
            experiment_button=gr.Button('168 órás kísérlet indítása / folytatása')
            progress_note=gr.Markdown(); training_metrics=gr.JSON(label='Loss, timing, sebesség, padding és VRAM')
        with gr.Tab('Generate'):
            checkpoint=gr.Textbox(value=str(home/'runs'/'mapper-full'/'best.pt'),label='V2 mapper checkpoint')
            songs=gr.File(file_count='multiple',file_types=['.mp3','.ogg','.wav','.flac'],type='filepath',label='Zenék')
            folder=gr.Textbox(label='Vagy egy zenemappa'); output=gr.Textbox(value=str(home/'generated'),label='Kimeneti mappa')
            with gr.Row():
                stars=gr.Slider(0,15,value=4,step=.1,label='Célcsillag'); preset=gr.Dropdown(['Auto','Aim','Streams','Complex Rhythm','Custom'],value='Auto',label='Stílus')
                ga=gr.Dropdown(options,value=options[0],label='Aim'); gs=gr.Dropdown(options,value=options[0],label='Streams'); grh=gr.Dropdown(options,value=options[0],label='Rhythm')
            with gr.Row():
                seed=gr.Number(value=42,precision=0,label='Seed'); variants=gr.Number(value=1,precision=0,label='Variánsok'); candidates=gr.Number(value=3,precision=0,label='Jelöltek variánsonként')
            reference=gr.Textbox(label='Mintavilág referencia .osu (opcionális)'); strength=gr.Slider(0,1,value=.7,label='Referencia hatása')
            with gr.Row():
                bpm=gr.Number(value=None,label='BPM (opcionális)'); offset=gr.Number(value=None,label='Offset ms (opcionális)')
            timing=gr.Textbox(label='Timing referencia .osu (külön a stílusreferenciától)')
            with gr.Row():
                ar=gr.Number(value=None,label='AR'); od=gr.Number(value=None,label='OD'); cs=gr.Number(value=None,label='CS'); hp=gr.Number(value=None,label='HP')
            generate_button=gr.Button('Teljes map generálása / folytatása',variant='primary')
            preview_song=gr.Textbox(label='Metronómos előnézethez egy zenefájl útvonala')
            timing_button=gr.Button('Megadott timing meghallgatása')
            automatic_button=gr.Button('Automatikus timing meghallgatása')
            timing_audio=gr.Audio(label='Metronóm + zene'); timing_quality=gr.JSON(label='Timing bizonyossága')
        with gr.Tab('Evaluate'):
            suite=gr.Dropdown(['timing','mapping','styles','full'],value='full',label='Mérés'); count=gr.Number(value=32,precision=0,label='Tesztzenék')
            v1_run=gr.Textbox(label='V1 futás mappája az átfedő tanítózenék kizárásához (opcionális)')
            evaluate_button=gr.Button('Ismeretlen zenék értékelése')
            gr.Markdown('Az automatikus és referencia-timingos mapok külön szerepelnek. A fun/flow értékeket valódi osu!stable playtest után kell kitölteni.')
        cancel=gr.Button('Aktuális feladat leállítása'); job_state=gr.Textbox(label='Feladat állapota',lines=4); log=gr.Textbox(label='Napló',lines=10)
        prepare.click(lambda s,f,d,b:job('prepare',source_db=s,files_root=f,destination=d,max_metadata_gib=float(b)),[source,files_root,dataset,metadata_budget],current_job)
        refresh_report.click(lambda d:load_json(Path(d)/'report.json',{}),dataset,report)
        review_outputs=[cursor,mid,heading,music,pattern,la,ls,lr]
        load.click(review,[dataset,cursor],review_outputs)
        skip.click(lambda d,c:review(d,c+1),[dataset,cursor],review_outputs)
        accept.click(save,[dataset,mid,la,ls,lr],label_note)
        exclude.click(lambda d,m,a,s,r:save(d,m,a,s,r,True),[dataset,mid,la,ls,lr],label_note)
        stage.change(lambda s:(str(home/'runs'/(s+'-full')),24 if s=='audio' else 120),stage,[run,hours])
        bench.click(lambda:job('benchmark',output_dir=str(home/'benchmark')),None,current_job)
        train_button.click(train_job,[dataset,run,stage,size,hours,batch,workers,resume],current_job)
        features_button.click(lambda d,c,b,f:job('features',data_root=d,checkpoint=c,device='auto',max_gib=float(b),min_free_gib=float(f)),[dataset,encoder_checkpoint,feature_budget,free_reserve],current_job)
        experiment_button.click(lambda d,r:job('experiment',data_root=d,run_dir=r),[dataset,experiment_root],current_job)
        generate_button.click(generate_job,[checkpoint,folder,songs,output,stars,preset,ga,gs,grh,seed,variants,candidates,bpm,offset,timing,reference,strength,ar,od,cs,hp],current_job)
        timing_button.click(preview_timing,[preview_song,bpm,offset,timing,output],[timing_audio,timing_quality])
        automatic_button.click(preview_automatic,[preview_song,encoder_checkpoint,output],[timing_audio,timing_quality])
        evaluate_button.click(lambda c,s,n,v:job('evaluate',checkpoint_path=c,suite=s,count=int(n),v1_run=v.strip() or None),[checkpoint,suite,count,v1_run],current_job)
        cancel.click(cancel_job,current_job,job_state)
        gr.Timer(2).tick(status,[current_job,run],[job_state,log,progress_note,training_metrics])
    return app


def launch(home,port=7861,open_browser=True):
    build(home).queue().launch(server_name='127.0.0.1',server_port=port,share=False,inbrowser=open_browser,allowed_paths=[str(Path(home).resolve())])
