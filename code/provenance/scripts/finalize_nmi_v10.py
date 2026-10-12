from pathlib import Path
import re,shutil,json,hashlib,zipfile
from pypdf import PdfReader
B=Path(__file__).resolve().parents[1];O=B/'output/NMI_English_Manuscript_v10'
# A single textual manuscript, with captions and numerical reference list inlined.
s=(O/'main.tex').read_text(encoding='utf-8');leg=(O/'legends.tex').read_text(encoding='utf-8')
def balanced(text,start):
 depth=0
 for i in range(start,len(text)):
  if text[i]=='{' and (i==0 or text[i-1]!='\\'):depth+=1
  elif text[i]=='}' and (i==0 or text[i-1]!='\\'):
   depth-=1
   if depth==0:return text[start+1:i],i+1
 raise ValueError('Unbalanced group')
caps={}
for m in re.finditer(r'\\newcommand\{\\(legend\w+)\}\{',leg):caps[m.group(1)]=balanced(leg,m.end()-1)[0]
s=s.replace('\\input{legends.tex}\n','')
m=re.search(r'\\newcommand\{\\paperfigure\}\[3\]\{',s)
_,end=balanced(s,m.end()-1);s=s[:m.start()]+s[end:]
def fig(m):
 return '\\clearpage\n\\begin{center}\\includegraphics[width=\\textwidth,height='+m[3]+'\\textheight,keepaspectratio]{figures/'+m[1]+'.pdf}\\end{center}\n\\vspace{-2mm}\\noindent{\\small '+caps[m[2]]+'\\par}\\clearpage\n'
s=re.sub(r'\\paperfigure\{([^}]+)\}\{\\(legend\w+)\}\{([^}]+)\}',fig,s)
s=s.replace('\\bibliographystyle{sn-nature}\n','').replace('\\bibliography{references}',(O/'main.bbl').read_text(encoding='utf-8').replace('\\bibcommenthead',''))
s=s.replace('\\providecommand{\\bibcommenthead}{}\n','')
(O/'main_submission.tex').write_text(s,encoding='utf-8')
# Copy final display outputs, preserving editable build inputs.
for stem,label in [('main','Main'),('supplement','Supplement')]:shutil.copy2(O/f'{stem}.pdf',O/f'NMI_English_{label}_v10.pdf')
shutil.copy2(O/'figures/Fig1_overview.pdf',O/'Figure1_Overview_v10.pdf')
files=[]
for p in O.rglob('*'):
 if not p.is_file() or p.suffix in {'.aux','.log','.out','.blg'}:continue
 if p.name in {'main_user_20261007.tex'}:continue
 if 'audit' in p.parts and p.suffix=='.png':continue
 if p.name=='main_submission.pdf':continue
 files.append(p)
manifest={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='package_manifest.json'}
(O/'audit/package_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
z=O.parent/'NMI_English_LaTeX_v10.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in files:
  f.write(p,Path(O.name)/p.relative_to(O))
 if O/'audit/package_manifest.json' not in files:f.write(O/'audit/package_manifest.json',Path(O.name)/'audit/package_manifest.json')
print(json.dumps({'zip':str(z),'MB':round(z.stat().st_size/1024**2,1),'main_pages':len(PdfReader(O/'main.pdf').pages),'supplement_pages':len(PdfReader(O/'supplement.pdf').pages)}))
