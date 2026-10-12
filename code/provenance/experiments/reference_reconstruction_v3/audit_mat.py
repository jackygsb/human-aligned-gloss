from pathlib import Path
import scipy.io as sio
import numpy as np
import csv,json
ROOT=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent/'audit';out.mkdir(exist_ok=True)
rows=[];errors=[];files=[]
def visit(v,path,source,depth=0):
    if depth>25:
        rows.append([source,path,'depth_limit','']);return
    if isinstance(v,dict):
        for k,x in v.items():
            if not k.startswith('__'):visit(x,path+'.'+k,source,depth+1)
    elif isinstance(v,(list,tuple)):
        for i,x in enumerate(v):visit(x,path+f'[{i}]',source,depth+1)
    elif isinstance(v,np.ndarray):
        rows.append([source,path,str(v.dtype),str(v.shape)])
        if v.dtype.names:
            for k in v.dtype.names:visit(v[k],path+'.'+k,source,depth+1)
        elif v.dtype==object:
            for i,x in enumerate(v.flat):visit(x,path+f'[{i}]',source,depth+1)
    else:rows.append([source,path,type(v).__name__,'scalar'])
for folder in ['ColorANDGlossConstancy-main','gloss_tinynetworks-main']:
    for f in (ROOT/folder).rglob('*.mat'):
        name=str(f.relative_to(ROOT));files.append(name)
        try:visit(sio.loadmat(f,simplify_cells=True),'',name)
        except Exception as e:errors.append({'file':name,'error':str(e)})
with (out/'mat_nested_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['file','field_path','dtype','shape']);w.writerows(rows)
summary={'files_attempted':len(files),'leaf_array_records':len(rows),'errors':errors,
 'reference_named_fields':[r for r in rows if any(k in r[1].lower() for k in ['reference','render','mesh','vertex','camera','hdr'])]}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'files_attempted':len(files),'records':len(rows),'errors':errors,'reference_named_fields_count':len(summary['reference_named_fields'])}))
