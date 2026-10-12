from pathlib import Path
import re,json,subprocess
from pypdf import PdfReader
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[1];O=B/'output/NMI_English_Manuscript_v10';A=O/'audit';A.mkdir(exist_ok=True)
def wc(t):
 t=re.sub(r'\\(?:cite|ref|label)\{[^}]*\}','',t);t=re.sub(r'\\[A-Za-z]+',' ',t)
 return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",t))
report=json.loads((A/'word_counts.json').read_text())
leg=(O/'legends.tex').read_text(encoding='utf-8')
report['legend_words']=[wc(p) for p in re.split(r'\\newcommand\{\\legend\w+\}\{',leg)[1:]]
for stem in ['main','supplement']:
 pdf=PdfReader(O/f'{stem}.pdf');log=(O/f'{stem}.log').read_text(encoding='utf-8',errors='replace')
 report[stem]={'pages':len(pdf.pages),'warnings':[l for l in log.splitlines() if any(x in l.lower() for x in ['undefined','overfull','missing character','rerun to get','labels may have changed'])], 'bibliography_entries':(O/f'{stem}.bbl').read_text().count('\\bibitem')}
 folder=A/stem;folder.mkdir(exist_ok=True)
 subprocess.run(['pdftoppm','-scale-to','800','-png',str(O/f'{stem}.pdf'),str(folder/'p')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 files=[folder/f'p-{i:02}.png' for i in range(1,len(pdf.pages)+1)]
 for k in range(0,len(files),9):
  sheet=Image.new('RGB',(3*600,3*850),'#dddddd');d=ImageDraw.Draw(sheet)
  for j,p in enumerate(files[k:k+9]):
   im=Image.open(p);im.thumbnail((590,815));x=(j%3)*600;y=(j//3)*850;sheet.paste(im,(x,y+25));d.text((x+8,y+5),f'{stem} page {k+j+1}',fill='black')
  sheet.save(A/f'{stem}_contact_{k//9+1}.png')
 text='\n\n'.join(p.extract_text() or '' for p in pdf.pages)
 (A/f'{stem}_pdf_text.txt').write_text(text,encoding='utf-8')
 report[stem]['replacement_glyph_count']=text.count('\ufffd')
source=(O/'main.tex').read_text(encoding='utf-8')
report['red_commands_remaining']=len(re.findall(r'\\(?:color|textcolor)\{red\}',source))
report['sections']=re.findall(r'\\section\*?\{([^}]+)\}',source)
(A/'quality_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=True,indent=2))
