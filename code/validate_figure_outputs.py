"""Compare regenerated PDFs with a canonical manifest; optionally render contact sheets.

Requires pypdf and Pillow. PDF rasterization uses pdftoppm when available.
Whitespace-normalized extracted text and dimensions are compared, not PDF hashes
(which include timestamps). This supplements execution logs, not scientific review.
"""
from pathlib import Path
import argparse,json,re,shutil,subprocess,hashlib
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
MAIN={1:'Fig1_overview',2:'Figure2_HumanLike_Gloss_EN',3:'Figure3_Shared_Condition_Geometry_EN',
      4:'Figure4_Cross_Task_Transfer_EN',5:'Figure5_Frozen_Internal_Readout_EN',6:'Figure6_Internal_Direction_Intervention_EN'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'figures/regenerated');ap.add_argument('--render',action='store_true');args=ap.parse_args()
    out=args.output.resolve();qa=out/'qa';qa.mkdir(exist_ok=True)
    expected=json.loads((ROOT/'figures/expected_outputs.json').read_text())['figures']
    pairs=[(e['figure'],e,out/e['regenerated_path']) for e in expected]
    rows=[]
    for key,old,new in pairs:
        if not new.exists():rows.append({'figure':key,'error':'missing regenerated PDF'});continue
        b=PdfReader(new)
        norm=lambda s:re.sub(r'\s+',' ',s).strip()
        txb=norm(' '.join(p.extract_text() for p in b.pages))
        dims=lambda r:[[round(float(p.mediabox.width),3),round(float(p.mediabox.height),3)] for p in r.pages]
        rows.append({'figure':key,'pages':len(b.pages),'matches_page_count':old['pages']==len(b.pages),
                     'matches_dimensions':old['dimensions']==dims(b),'matches_extracted_text':old['text_sha256']==hashlib.sha256(txb.encode()).hexdigest(),
                     'published_text_characters':old['text_characters'],'regenerated_text_characters':len(txb)})
    if args.render:
        tool=shutil.which('pdftoppm')
        if not tool:raise RuntimeError('pdftoppm not found; text audit completed but rendering is required for --render')
        def render(item):
            key,old,new=item
            if new.exists():subprocess.run([tool,'-f','1','-singlefile','-scale-to','480','-png',str(new),str(qa/key)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(render,pairs))
        for batch in range((len(pairs)+11)//12):
            canvas=Image.new('RGB',(1920,1680),'#EDF0F4');draw=ImageDraw.Draw(canvas)
            for i,(key,_,_) in enumerate(pairs[batch*12:(batch+1)*12]):
                x=(i%4)*480;y=(i//4)*560;draw.text((x+14,y+8),key,fill='#24323B')
                p=qa/(key+'.png')
                if p.exists():
                    im=Image.open(p).convert('RGB');im.thumbnail((456,510));canvas.paste(im,(x+(480-im.width)//2,y+34))
            canvas.save(ROOT/f'docs/figure_reproduction_contactsheet_{batch+1}.png')
    target=ROOT/'docs/figure_reproduction_pdf_audit.json';target.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps({'figures':len(rows),'text_mismatches':[r['figure'] for r in rows if not r.get('matches_extracted_text')],
                      'dimension_mismatches':[r['figure'] for r in rows if not r.get('matches_dimensions')],'report':str(target)}))
if __name__=='__main__':main()
