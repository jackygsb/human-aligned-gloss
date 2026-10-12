from pathlib import Path
import json,hashlib,subprocess,sys,platform
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material'
paths=list((BASE.parent/'SPoSE-master').glob('*.py'))+list((BASE.parent/'SPoSE-master/models').glob('*.py'))+[BASE.parent/'gloss_tinynetworks-main/fig3_model_comparison.m']+list((BASE/'scripts').glob('*mllm*.py'))+list((BASE/'scripts').glob('*spose*.py'))+[BASE/'scripts/internvl_backend.py',BASE/'scripts/analyze_human_reference.py']
paths+=list(map(lambda n:BASE/'scripts'/n,['run_gloss_pipeline.py','prepare_input_control.py','verify_protocol_assets.py','analyze_choice_robustness.py','analyze_residual_controls.py','check_experiment_semantics.py','check_structured_bootstrap.py','package_gloss_results.py','audit_author_baselines.py']))
record=dict(python=platform.python_version(),local_source_git_commit='not inferable from supplied ZIP copies; record content hashes instead',files={str(p.relative_to(BASE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},model_pins={p.parent.name:json.loads(p.read_text())['revision'] for p in (BASE/'models').glob('*/manifest.json') if 'revision' in json.loads(p.read_text())})
(OUT/'protocol/source_provenance.json').write_text(json.dumps(record,indent=2))
packages=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
(OUT/'protocol/runtime_requirements_freeze.txt').write_text(packages)
print('Sources, pinned model revisions and runtime versions recorded.')
