"""Small numerical checks for the scientifically consequential ID/choice conversion."""
import itertools,json
import numpy as np
import torch
from run_mllm_responses import parse
from fit_gloss_spose import logits,upstream
for task,obj,status in [('T1',dict(glossiness=100,status='ok'),'valid'),('T1',dict(glossiness=101,status='ok'),'invalid_value'),('T1',dict(glossiness=True,status='ok'),'invalid_value'),('T1',dict(glossiness=None,status='unrateable'),'unrateable'),('T3',dict(odd='C',status='ok'),'valid'),('T2',dict(choice='tie',status='ok'),'valid')]:assert parse(json.dumps(obj),task)[1]==status
assert parse('```json\n'+json.dumps(dict(odd='B',status='ok'))+'\n```','T3')[1]=='valid'
w=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
for order in itertools.permutations(range(3)):
    choice=int(logits(w,torch.tensor([order],dtype=torch.long)).argmax(1));assert order[choice]==2
model=upstream.SPoSE(5,3);ids=torch.tensor([2,4,1]);assert torch.allclose(model(torch.eye(5)[ids]),model.fc.weight.T[ids])
print('PASS: all six odd-one-out permutations, upstream one-hot equivalence, strict score range/type parsing, explicit ties and unrateable outputs.')
