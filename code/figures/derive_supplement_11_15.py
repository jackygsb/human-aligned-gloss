"""Reflow vector methods panels with local supplementary panel labels."""
from pathlib import Path
from io import BytesIO
from pypdf import PdfReader,PdfWriter
from reportlab.pdfgen import canvas
import os
OSFROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('OSF_REPRO_OUT',OSFROOT/'figures/regenerated'))
O=OUT/'supplement';O.mkdir(parents=True,exist_ok=True)
src=OUT/'workflows/Figure1_Methods_EN.pdf'
p=PdfReader(src).pages[0];W=float(p.mediabox.width);H=float(p.mediabox.height)
items=[('S11_methods',355,2140,[(11.905512,111.780443,'a'),(141.732283,111.780443,'b')]),
 ('S12_controller',2145,3200,[(11.905512,616.914301,'a')]),
 ('S12_records',3210,4608,[(11.905512,921.638711,'b')]),
 ('S13_geometry',4620,6810,[(11.905512,1321.607215,'')]),
 ('S14_readout',6820,8615,[(11.905512,1945.102766,'')]),
 ('S15_intervention',8620,10970,[(11.905512,2455.182018,'')])]
for name,top,bottom,labels in items:
 page=PdfReader(src).pages[0];buf=BytesIO();c=canvas.Canvas(buf,pagesize=(W,H))
 for x,y,t in labels:
  c.setFillColorRGB(1,1,1);c.rect(x-1,H-y-2,10,13,stroke=0,fill=1)
  c.setFillColorRGB(.125,.176,.220);c.setFont('Helvetica-Bold',9.637795);c.drawString(x,H-y,t)
 if name=='S11_methods':
  c.setFillColorRGB(.976,.980,.984);c.rect(22.2,H-557.314242-1.6,105,8.9,stroke=0,fill=1)
  c.setFillColorRGB(.396,.459,.514);c.setFont('Helvetica',6.519685);c.drawString(22.677165,H-557.314242,'comparisons (Fig. S12).')
 if name=='S13_geometry':
  c.setFillColorRGB(.976,.980,.984);c.rect(270,H-1902.905876-1.6,220,9,stroke=0,fill=1)
  c.setFillColorRGB(.396,.459,.514);c.setFont('Helvetica',6.23622);c.drawString(270.425197,H-1902.905876,'Question: do the tasks predict the same ordering?')
 c.save();buf.seek(0);page.merge_page(PdfReader(buf).pages[0])
 page.mediabox.lower_left=(0,H-bottom/1800*W);page.mediabox.upper_right=(W,H-top/1800*W);page.cropbox=page.mediabox
 writer=PdfWriter();writer.add_page(page);writer.write(O/f'{name}.pdf')
print('Prepared six vector panel assets.')
