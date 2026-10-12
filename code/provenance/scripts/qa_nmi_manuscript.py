from pathlib import Path
import subprocess,json,re
from pypdf import PdfReader
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[1];S=B/'mllm_material/nmi_manuscript_zh_v1/source';Q=B.parent/'tmp/pdfs/nmi_qa';Q.mkdir(parents=True,exist_ok=True)
report={}
for name in ['main','supplement']:
 p=S/(name+'.pdf');r=PdfReader(p);texts=[x.extract_text() or '' for x in r.pages]
 (Q/(name+'_text.txt')).write_text('\n\f\n'.join(texts),'utf-8')
 log=(S/(name+'.log')).read_text('utf-8',errors='replace')
 report[name]={'pages':len(r.pages),'chars_per_page':[len(t) for t in texts],'missing_glyphs':len(re.findall('Missing character:',log)),'undefined_citations':'undefined' in log.lower(),'overfull_boxes':re.findall(r'Overfull[^\n]*',log),'bibitems':len(re.findall(r'\\bibitem', (S/(name+'.bbl')).read_text('utf-8')))}
 subprocess.run(['pdftoppm','-scale-to','700','-png',str(p),str(Q/name)],check=True,capture_output=True)
 pages=sorted(Q.glob(name+'-*.png'))
 for batch in range(0,len(pages),8):
  tile=Image.new('RGB',(4*280,2*420),'#DDE3E8');d=ImageDraw.Draw(tile)
  for i,impath in enumerate(pages[batch:batch+8]):
   im=Image.open(impath);im.thumbnail((270,386));x=(i%4)*280+(280-im.width)//2;y=(i//4)*420+24;tile.paste(im,(x,y));d.text(((i%4)*280+10,(i//4)*420+6),impath.stem,fill='black')
  tile.save(Q/f'{name}_contact_{batch//8+1}.png')
(Q/'qa_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(report,ensure_ascii=False))
