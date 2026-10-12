from pathlib import Path
import json,shutil,requests,concurrent.futures
B=Path(__file__).resolve().parents[1]
O=B/'mllm_material/nmi_manuscript_zh_v1'
S=O/'source';S.mkdir(exist_ok=True)
for sub in ['figures','tables','provenance','code']: (S/sub).mkdir(exist_ok=True)
T=O/'official_template/sn-article-template'
shutil.copy2(T/'sn-jnl.cls',S/'sn-jnl.cls')
shutil.copy2(T/'bst/sn-nature.bst',S/'sn-nature.bst')
dois={
'morimoto2026':'10.1038/s41562-026-02445-0','du2025':'10.1038/s42256-025-01049-z',
'morimoto2023':'10.1167/jov.23.7.8','fleming2014':'10.1016/j.visres.2013.11.004',
'fleming2017':'10.1146/annurev-vision-102016-061429','motoyoshi2007':'10.1038/nature05724',
'storrs2021':'10.1038/s41562-021-01097-6','pellacini2000':'10.1145/344779.344812',
'hebart2020':'10.1038/s41562-020-00951-3','rsa2008':'10.3389/neuro.06.004.2008',
'nili2014':'10.1371/journal.pcbi.1003553','diedrichsen2017':'10.1371/journal.pcbi.1005508',
'kriegeskorte2013':'10.1016/j.tics.2013.06.007','geirhos2020':'10.1038/s42256-020-00257-z',
'bh1995':'10.1111/j.2517-6161.1995.tb02031.x','efron1979':'10.1214/aos/1176344552',
'ward1992':'10.1145/133994.134078','fleming2003':'10.1167/3.5.3',
'marlow2012':'10.1016/j.cub.2012.08.009','anderson2009':'10.1167/9.11.10',
'olkkonen2010':'10.1167/10.9.5',
}
def fetch(kv):
 k,d=kv;p=S/'provenance'/f'{k}.json'
 if p.exists():return k,json.loads(p.read_text('utf-8'))
 r=requests.get('https://api.crossref.org/works/'+d,timeout=35);r.raise_for_status();m=r.json()['message'];p.write_text(json.dumps(m,ensure_ascii=False,indent=2),'utf-8');return k,m
out={}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
 for f in concurrent.futures.as_completed([ex.submit(fetch,x) for x in dois.items()]):
  try:k,m=f.result();out[k]=m;print(k,m.get('title'),flush=True)
  except Exception as e:print(type(e).__name__,str(e)[:150],flush=True)
(S/'provenance/bibliography_metadata.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8')
test=r'''\documentclass[sn-nature]{sn-jnl}
\usepackage{xeCJK}
\setCJKmainfont{SimSun}
\setCJKsansfont{Microsoft YaHei}
\usepackage{amsmath,booktabs,graphicx}
\begin{document}
\title{预训练多模态语言模型中的人类式光泽判断}
\author{作者信息待填写}
\abstract{中文排版测试，检验数学公式、参考文献与图片。}
\maketitle
\section{结果}
测试文本。$R^2=0.68$。模型的内部响应可以用于判断。
\end{document}
'''
(S/'test.tex').write_text(test,'utf-8')
