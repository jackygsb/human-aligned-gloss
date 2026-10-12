from pathlib import Path
O=Path(__file__).resolve().parents[1]/'output/NMI_English_Manuscript_v9'
p=O/'figure_plates.tex';s=p.read_text(encoding='utf-8');s=s.replace('′',"$^{\\prime}$");s=s[s.index('\\clearpage'):];p.write_text(s,encoding='utf-8')
p=O/'tables_en/S5.tex';s=p.read_text();s=s.replace('p{55mm}','p{53mm}');p.write_text(s)
p=O/'main.tex';s=p.read_text(encoding='utf-8');s=s.replace('Material appearance depends jointly on reflectance, shape and illumination','Material appearance depends on reflectance, shape and illumination').replace('Using 3,888 controlled rendered images','Using 3,888 rendered images').replace('independently collected pairwise choices. Supervised','pairwise choices. Supervised').replace('in general-purpose multimodal systems, providing a reproducible framework','in multimodal systems, providing a framework');p.write_text(s,encoding='utf-8')
