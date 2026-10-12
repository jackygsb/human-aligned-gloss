from pathlib import Path
import json,shutil,zipfile,hashlib,xml.etree.ElementTree as ET
import numpy as np
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure1_english_v1';A=O/'assets';A.mkdir(exist_ok=True)
manifest=json.loads((O/'source_manifest.json').read_text('utf-8'))
examples=json.loads((O/'example_records.json').read_text('utf-8'))
for item in manifest:
 rel=Path(item['source']);src=B/rel;dst=A/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 if rel.name=='full.jsonl':dst.write_text('\n'.join(json.dumps(examples[k]) for k in ['single_image','pairwise'])+'\n',encoding='utf-8')
 elif rel.name=='trials.jsonl':dst.write_text(json.dumps(examples['matching'])+'\n',encoding='utf-8')
 elif rel.suffix=='.npz':
  z=np.load(src);keys=['candidate','reference_means'] if rel.name=='directions.npz' else ['bridge_tokens'];np.savez_compressed(dst,**{k:z[k] for k in keys})
 else:shutil.copy2(src,dst)
# Copy the standalone plotting script and make its project root relocatable.
s=(B/'scripts/draw_figure1_english.py').read_text('utf-8')
s=s.replace("B=Path(__file__).resolve().parents[1];M=B/'mllm_material';V=M/'nmi_revision_v8/source';O=M/'figure1_english_v1';O.mkdir(exist_ok=True)","HERE=Path(__file__).resolve().parent;B=HERE/'assets';M=B/'mllm_material';V=M/'nmi_revision_v8/source';O=HERE;O.mkdir(exist_ok=True)")
s=s.replace("shutil.copy2(__file__,O/Path(__file__).name)","# This script is already stored in the output directory.")
(O/'draw_figure1_english.py').write_text(s,encoding='utf-8')
(O/'README_EN.txt').write_text('''FIGURE 1 | ENGLISH EDITABLE METHODS PLATE

Primary editable file: Figure1_Methods_EN.svg
All labels are live English text. Lines, axes, markers and the 36 x 36 dissimilarity matrices are vector objects. Image stimuli and token-projection overlays are embedded raster images. SVG image assets are self-contained; moving the SVG alone will not break the images.

Canvas: 180 x 227 mm. Font: Arial. Install Arial before editing to preserve the layout. Open the SVG with Inkscape or Adobe Illustrator; ungroup the objects when editing. Panel heading groups are named panel_a ... panel_h. PDF is vector output with embedded fonts. PNG is a 450 dpi preview. No claim of compliance with every journal-specific final production dimension is implied; the editable master can be resized during submission formatting.

Figure1_caption_EN.txt supplies the full English caption and the scope of each methodological exemplar. The displayed outputs and trajectory are archived InternVL records. Both models contribute to the actual RDMs, readout curves and intervention illustrations. Colors are consistent: teal = InternVL, orange = Qwen3; dashed gray in the intervention plot denotes brightness controls. Norm-matched random controls are described in the caption and fully shown in the existing results figure.

Rebuild: install numpy, pandas, matplotlib and pillow, then run python draw_figure1_english.py from this directory. The assets folder contains the exact images, selected reference arrays, token arrays and numeric tables needed for this figure. Raw response extracts contain only the displayed example records. No model calls are needed.

Source provenance: source_manifest.json identifies original project inputs and their SHA-256 hashes. Bundled arrays have been reduced to fields used by the figure. package_manifest.json hashes the actual bundled files. Human reference data and original stimulus assets remain attributable to Morimoto et al.; reconstruction assets and derived analyses are identified separately in their paths. Consult original dataset licenses for manuscript reuse.

Display details: RDM CSVs have no header and are read as complete 36 x 36 symmetric matrices. RDM values are divided by each matrix's own mean off-diagonal distance. Spatial response overlays show the same stimulus in both models, with independent model-specific color scales; they are not attention maps. Reference projection curves are rescaled separately to [0,1] for visual comparison. No new experimental responses were generated.
''',encoding='utf-8')
ns={'s':'http://www.w3.org/2000/svg'};root=ET.parse(O/'Figure1_Methods_EN.svg').getroot();texts=root.findall('.//s:text',ns);imgs=root.findall('.//s:image',ns)
assert len(texts)>100 and all('data:image/png;base64,' in im.attrib.get('{http://www.w3.org/1999/xlink}href','') for im in imgs)
assert not any('\u4e00'<=c<='\u9fff' for t in texts for c in ''.join(t.itertext()))
(O/'editability_check.json').write_text(json.dumps({'live_text_elements':len(texts),'embedded_images':len(imgs),'all_text_english':True,'self_contained_images':True},indent=2),encoding='utf-8')
paths=[p for p in O.rglob('*') if p.is_file() and p.name!='package_manifest.json']
(O/'package_manifest.json').write_text(json.dumps([{'path':p.relative_to(O).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],indent=2),encoding='utf-8');paths.append(O/'package_manifest.json')
zpath=M/'Figure1_English_Editable_Package.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for p in paths:z.write(p,Path('Figure1_English')/p.relative_to(O))
print('Live text:',len(texts),'Embedded images:',len(imgs),'ZIP MiB:',round(zpath.stat().st_size/1024**2,2))
