"""Create a local, allowlisted reproducibility snapshot; never publish automatically."""
from pathlib import Path
import json,hashlib,zipfile,datetime
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'mllm_material/matching_qwen3vl4b_pilot12'
GPT=BASE/'mllm_material/matching_subscription_cli_v1/matching_pilot12'
def main():
    cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'));files={}
    def add(p):
        if p.is_file():files[p.relative_to(BASE.parent).as_posix()]=p
    for name in ['run_qwen3_matching12.py','compare_pilot12_baselines.py','download_mllm_models.py','package_matching12.py','run_subscription_matching_pilot.py','analyze_subscription_pilot12.py','train_tiny_control.py']:
        add(BASE/'scripts'/name)
    for p in ROOT.rglob('*'):
        if p.suffix.lower() in ['.json','.jsonl','.csv','.png','.svg','.md','.txt','.html','.py'] and p.name not in ['processes.json','artifact_manifest.json']:add(p)
    for name in ['protocol.json','summary.json','analysis.json','state.json','matches.csv','image_summary.csv','results.png','结果解读.md']:add(GPT/name)
    for folder in GPT.glob('batch_*'):
        for name in ['mapping.json','prompt.txt','schema.json','answer.json','result.json']:add(folder/name)
    for relative in cfg['hashes']:add(BASE/relative)
    for name in ['README.md','LICENSE','download_plan.json','manifest.json','config.json','generation_config.json','preprocessor_config.json','processor_config.json','chat_template.json']:add(BASE/'models/qwen3vl4b'/name)
    for name in ['README.md','LICENSE','fig3_model_comparison.m']:add(BASE.parent/'gloss_tinynetworks-main'/name)
    add(BASE.parent/'gloss_tinynetworks-main/data/pred_validation/prediction_3888imgs.csv')
    for arch in ['onelayer_models','threelayer_models']:
        for p in (BASE.parent/'gloss_tinynetworks-main/data/networks'/arch).glob('*/corrs_*.csv'):add(p)
    for relative in ['annotations/gloss_annotations.csv','protocols/gloss_group_folds.csv','protocols/robustness_v2_config.json','mllm_material/matching_full_v1/protocol.json','mllm_material/matching_full_v1/schedule.json','mllm_material/matching_full_v1/image_summary.csv','robustness_v2/tiny_paper_lr/all_predictions.csv']:add(BASE/relative)
    for p in (BASE/'robustness_v2/tiny_paper_lr').glob('*_audit.json'):add(p)
    # Include the independently trained tiny checkpoints needed to audit predictions.
    for p in (BASE/'robustness_v2/tiny_paper_lr').glob('*.pt'):add(p)
    records={name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in files.items()}
    state=json.loads((ROOT/'status.json').read_text(encoding='utf-8'))
    manifest=dict(created=datetime.datetime.now().isoformat(),collection_state=state['state'],files=records,privacy='Allowlisted experimental records only. Credentials, browser profiles and CLI session event logs excluded. Full original local records retained.',redistribution='Author assets retain author MIT notice; model weights not bundled, use pinned official repository and license. This archive is local, not uploaded.')
    dest=ROOT/'reproducibility_snapshot.zip';temp=dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name,p in files.items():z.write(p,name)
        z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    temp.replace(dest)
    (ROOT/'release_manifest.json').write_text(json.dumps(dict(snapshot=str(dest.name),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),n_files=len(records),bytes=dest.stat().st_size,state=state['state']),indent=2),encoding='utf-8')
    print('Saved local reproducibility snapshot:',len(records),'files;',state['state'],flush=True)
if __name__=='__main__':main()
