from pathlib import Path
import re,json,subprocess
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
O=Path(__file__).resolve().parents[1]/'output/NMI_English_Manuscript_v9'
s=(O/'main.tex').read_text(encoding='utf-8')
def words(t):
 t=re.sub(r'\\(?:cite|ref|label)\{[^}]*\}','',t)
 t=re.sub(r'\\[a-zA-Z]+',' ',t)
 return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",t))
body=s.split('\\maketitle')[1].split('\\section{Methods}')[0]
abstract=re.search(r'\\abstract\{(.*?)\}\s*\\keywords',s,re.S).group(1)
report={'main_text_words_approx':words(body),'abstract_words':words(abstract)}
for stem in ['main','supplement']:
 pdf=PdfReader(O/f'{stem}.pdf');log=(O/f'{stem}.log').read_text(errors='replace')
 report[stem]={'pages':len(pdf.pages),'warnings':[l for l in log.splitlines() if any(x in l.lower() for x in ['undefined','overfull','missing character','rerun to get'])], 'bibliography_entries':(O/f'{stem}.bbl').read_text().count('\\bibitem')}
 folder=O/'audit'/stem;folder.mkdir(exist_ok=True)
 subprocess.run(['pdftoppm','-scale-to','650','-png',str(O/f'{stem}.pdf'),str(folder/'p')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 files=sorted(folder.glob('p-*.png'))
 for k in range(0,len(files),12):
  sheet=Image.new('RGB',(4*480,3*690),'#dddddd');d=ImageDraw.Draw(sheet)
  for j,p in enumerate(files[k:k+12]):
   im=Image.open(p);im.thumbnail((470,655));x=(j%4)*480;y=(j//4)*690;sheet.paste(im,(x,y+25));d.text((x+8,y+5),f'{stem} page {k+j+1}',fill='black')
  sheet.save(O/'audit'/f'{stem}_contact_{k//12+1}.png')
(O/'audit/quality_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True,indent=2))
