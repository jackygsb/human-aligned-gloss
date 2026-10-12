from pathlib import Path
import shutil
B=Path(__file__).resolve().parents[1];O=B/'mllm_material/figure1_english_v3'
P=O/'provenance_scripts';P.mkdir(exist_ok=True)
shutil.copy2(B/'scripts/audit_figure1_acquisition.py',P/'audit_figure1_acquisition.py')
(O/'ACQUISITION_COUNTS_EN.txt').write_text('''ACQUISITION COUNT DEFINITIONS

InternVL: 3,888 single-image responses + 4,000 pairwise responses + 46,375 reference-comparison responses = 54,263 primary responses.
Qwen3: 3,888 single-image responses + 4,000 pairwise responses + 46,581 reference-comparison responses = 54,469 primary responses.
Combined = 108,732 primary model responses.

Each model used 3,888 unique test images, 2,000 image pairs in two display orders, and 7,776 complete reference searches (two starts per test image). One search contains multiple comparison responses. These counts are different units and should not be added together. The response total excludes pilots, practice, catch trials, other tasks, and internal interventions. Model responses are not independent images or human participants.

acquisition_audit.json and acquisition_counts.csv give verified counts and input hashes. acquisition_examples.json contains all displayed example records, selected by metadata or manifest order. The repeated reference images depict actual levels presented, not additional unique test stimuli.

Figure1_e_Acquisition_EN.svg is the independently editable panel e. The full Figure1_Methods_EN.svg incorporates this panel; panels b-d retain their previous wording and examples.

Rebuilding the figure requires no models or raw collection: the bundled images, numeric inputs and acquisition JSON files suffice. The separate provenance_scripts/audit_figure1_acquisition.py documents the original count audit; to rerun the audit, place it under scripts/ in the complete original benchmark project, whose full raw logs are required. The editable package contains selected example logs, not the entire experimental archive.
''',encoding='utf-8')
s=(B/'scripts/package_figure1_english_v2.py').read_text('utf-8')
s=s.replace('figure1_english_v2','figure1_english_v3').replace('Detailed_v2','Detailed_v3').replace('180 x 434','180 x 548')
s=s.replace("exec(compile(s,str(__file__),'exec'))", "s=s.replace('Task-only editable extract: Figure1_bcd_Task_Detail_EN.svg', 'Task-only editable extract: Figure1_bcd_Task_Detail_EN.svg\\nAcquisition panel: Figure1_e_Acquisition_EN.svg\\nCount definitions: ACQUISITION_COUNTS_EN.txt')\nexec(compile(s,str(__file__),'exec'))")
exec(compile(s,str(__file__),'exec'))
