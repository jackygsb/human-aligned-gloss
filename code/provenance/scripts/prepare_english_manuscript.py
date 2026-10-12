from pathlib import Path
import shutil,re,json,hashlib
from pypdf import PdfReader,PdfWriter
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';S=M/'nmi_revision_v8/source';O=B/'output/NMI_English_Manuscript_v9'
for sub in ['figures','figure_sources','code','audit','tables']:(O/sub).mkdir(parents=True,exist_ok=True)
for name in ['sn-jnl.cls','sn-nature.bst','references.bib']:shutil.copy2(S/name,O/name)
for name in ['main_NMI_submission_v8_red.tex','supplement_NMI_submission_v8.tex']:shutil.copy2(S/name,O/'audit'/name)
for p in (S/'tables').glob('*'):
 if p.is_file():shutil.copy2(p,O/'tables'/p.name)
names={2:'Figure2_HumanLike_Gloss_EN',3:'Figure3_Shared_Condition_Geometry_EN',4:'Figure4_Cross_Task_Transfer_EN',5:'Figure5_Frozen_Internal_Readout_EN',6:'Figure6_Internal_Direction_Intervention_EN'}
for n,name in names.items():
 root=M/f'figure{n}_english_v1';shutil.copy2(root/(name+'.pdf'),O/f'figures/Fig{n}.pdf');shutil.copy2(root/(name+'.svg'),O/'figure_sources'/(name+'.svg'))
 for cap in root.glob('*caption_EN*'):shutil.copy2(cap,O/'figure_sources'/cap.name)
root=M/'figure1_english_v6';shutil.copy2(root/'Figure1_Methods_EN.svg',O/'figure_sources/Figure1_Methods_EN.svg');shutil.copy2(root/'Figure1_caption_EN.txt',O/'figure_sources/Figure1_caption_EN.txt');shutil.copy2(root/'Figure1_Methods_EN.pdf',O/'figure_sources/Figure1_Methods_EN.pdf')
# Same Figure 1, split at its original panel boundaries for legible continued pages.
reader=PdfReader(root/'Figure1_Methods_EN.pdf');h=float(reader.pages[0].mediabox.height);w=float(reader.pages[0].mediabox.width)
for name,top,bottom in [('1a',0,355),('1bc',355,2140),('1d',2145,3200),('1e',3210,4608),('1f',4620,6810),('1g',6820,8615),('1h',8620,10970)]:
 p=PdfReader(root/'Figure1_Methods_EN.pdf').pages[0];p.mediabox.lower_left=(0,h-bottom/1800*w);p.mediabox.upper_right=(w,h-top/1800*w);p.cropbox=p.mediabox;writer=PdfWriter();writer.add_page(p);writer.write(O/f'figures/Fig{name}.pdf')
(O/'audit/source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'main_NMI_submission_v8_red.tex',S/'supplement_NMI_submission_v8.tex',S/'references.bib']},indent=2))
print(O)
