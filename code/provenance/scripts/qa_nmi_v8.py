from pathlib import Path
import subprocess,json,re
from pypdf import PdfReader
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[1];S=B/'mllm_material/nmi_revision_v8/source';Q=B.parent/'tmp/pdfs/nmi_v8_qa';Q.mkdir(parents=True,exist_ok=True)
report={}
for name in ['main_NMI_submission_v8_red','supplement_NMI_submission_v8']:
 r=PdfReader(S/(name+'.pdf'));texts=[x.extract_text() or '' for x in r.pages];(Q/(name+'_text.txt')).write_text('\n\f\n'.join(texts),encoding='utf-8');log=(S/(name+'.log')).read_text('utf-8',errors='replace')
 report[name]={'pages':len(r.pages),'chars_per_page':[len(t) for t in texts],'missing_glyphs':len(re.findall('Missing character:',log)),'undefined':'undefined' in log.lower(),'overfull_boxes':re.findall(r'Overfull[^\n]*',log),'bibitems':len(re.findall(r'\\bibitem',(S/(name+'.bbl')).read_text('utf-8')))}
 subprocess.run(['pdftoppm','-scale-to','800','-png',str(S/(name+'.pdf')),str(Q/name)],check=True,capture_output=True)
 pages=sorted(Q.glob(name+'-*.png'))
 for batch in range(0,len(pages),8):
  tile=Image.new('RGB',(1200,880),'#DDE3E8');dr=ImageDraw.Draw(tile)
  for i,p in enumerate(pages[batch:batch+8]):
   im=Image.open(p);im.thumbnail((290,410));x=i%4*300+(300-im.width)//2;y=i//4*440+23;tile.paste(im,(x,y));dr.text((i%4*300+10,i//4*440+5),str(batch+i+1),fill='black')
  tile.save(Q/f'{name}_contact_{batch//8+1}.png')
(Q/'qa_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
