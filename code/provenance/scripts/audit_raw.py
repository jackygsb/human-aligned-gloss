from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'gloss_tinynetworks-main'

def read_raw():
    audit, accepted = [], []
    for f in sorted((SRC/'onlineData').glob('ID*.csv')):
        t = pd.read_csv(f)
        blocks = [t.iloc[8:93].sort_values('imgN'), t.iloc[93:178].sort_values('imgN')]
        valid = all(len(b)==85 and np.array_equal(b.imgN.to_numpy(), np.arange(85)) for b in blocks)
        catch = float(sum(b['slider.response'].iloc[0] for b in blocks)) if valid else np.nan
        rt = np.median(np.concatenate([b['key_resp.rt'].iloc[1:].to_numpy() for b in blocks])) if valid else np.nan
        passed = valid and catch <= 4 and rt > 3
        group = int(t.condition.iloc[7])
        audit.append(dict(source_file=f.name,group=group,valid_structure=valid,catch_sum=catch,median_rt=rt,accepted=passed))
        if passed:
            responses = np.stack([b['slider.response'].iloc[1:].fillna(b['start_specularN'].iloc[1:]).to_numpy() for b in blocks],axis=1)
            accepted.append((f.name,group,responses,blocks))
    return pd.DataFrame(audit), accepted

if __name__ == '__main__':
    a, d = read_raw()
    print(a.groupby(['valid_structure','accepted']).size().to_string())
    print(a.head().to_string(index=False))
    m=loadmat(str(SRC/'data/onlineData.mat'),squeeze_me=True,struct_as_record=False)['data']
    print('accepted',len(d),'author',len(m))
    if len(d)==len(m):
        print('max response difference',max(np.max(np.abs(x[2]-y.response_rawN)) for x,y in zip(d,m)))
