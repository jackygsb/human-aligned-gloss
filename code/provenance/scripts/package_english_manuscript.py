from pathlib import Path
import shutil,zipfile,hashlib,json
B=Path(__file__).resolve().parents[1];O=B/'output/NMI_English_Manuscript_v9'
for stem,label in [('main','Main'),('supplement','Supplement')]:
 shutil.copy2(O/f'{stem}.pdf',O/f'NMI_English_{label}_v9.pdf')
excluded={'.aux','.log','.out','.blg'}
paths=[p for p in O.rglob('*') if p.is_file() and p.suffix not in excluded and not ('audit' in p.parts and p.suffix=='.png')]
manifest={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(O/'audit/package_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
z=O.parent/'NMI_English_LaTeX_v9.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in paths+[O/'audit/package_manifest.json']:f.write(p,Path(O.name)/p.relative_to(O))
print(str(z));print(f'{z.stat().st_size/1024/1024:.1f} MB; {len(paths)} files')
