"""Differentiable cost for putting a circle or slider head beyond the playfield."""
from __future__ import annotations

import torch

from .tokenizer import Tokenizer


def out_of_bounds_loss(model,hidden,batch,tokenizer=None):
    tok=tokenizer or Tokenizer()
    tokens=batch['tokens']; labels=batch['labels']
    xy_start,xy_count=tok.ranges['XY']
    heads=(tokens==tok.ids['CIRCLE']) | (tokens==tok.ids['SLIDER'])
    row,column=torch.where(heads & (torch.arange(tokens.shape[1],device=tokens.device)[None]+4<tokens.shape[1]))
    if not len(row): return hidden.sum()*0
    x_at=column+2; y_at=column+3
    good=(labels[row,x_at]>=xy_start)&(labels[row,x_at]<xy_start+xy_count)&(labels[row,y_at]>=xy_start)&(labels[row,y_at]<xy_start+xy_count)
    row,x_at,y_at=row[good],x_at[good],y_at[good]
    if not len(row): return hidden.sum()*0
    raw_cs=tokens[row,7]-tok.ranges['CS'][0]
    cs=torch.where((raw_cs>=0)&(raw_cs<=110),raw_cs.float()/10,5.)
    radius=54.4-4.48*cs
    coordinate=(torch.arange(xy_count,device=hidden.device,dtype=torch.float32)-1024)*2
    # The target may be a rare edge case in source maps; do not fight its CE label.
    tx=(labels[row,x_at]-xy_start-1024)*2
    ty=(labels[row,y_at]-xy_start-1024)*2
    good=(tx>=radius)&(tx<=512-radius)&(ty>=radius)&(ty<=384-radius)
    row,x_at,y_at,radius=row[good],x_at[good],y_at[good],radius[good]
    if not len(row): return hidden.sum()*0
    x_logits=model.project(hidden[row,x_at])[:,xy_start:xy_start+xy_count].float()
    y_logits=model.project(hidden[row,y_at])[:,xy_start:xy_start+xy_count].float()
    invalid_x=(coordinate[None]<radius[:,None])|(coordinate[None]>512-radius[:,None])
    invalid_y=(coordinate[None]<radius[:,None])|(coordinate[None]>384-radius[:,None])
    x_cost=(x_logits.softmax(-1)*invalid_x).sum(-1)
    y_cost=(y_logits.softmax(-1)*invalid_y).sum(-1)
    return (x_cost+y_cost).mean()/2
