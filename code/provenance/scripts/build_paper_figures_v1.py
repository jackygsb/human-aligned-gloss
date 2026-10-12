"""InternVL paper figures and fixed cross-task analyses from existing responses only."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,csv,hashlib,html,shutil,time,zipfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';F=M/'matching_full_v1';O=M/'paper_figures_v1'
for sub in ['figures','tables','source_data','methods','supplement','code']:(O/sub).mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rho(x,y):return float(spearmanr(x,y).statistic)
def status(s):print(s,flush=True);save(O/'status.json',dict(state=s,time=time.time()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
C=['#24788d','#d48136','#705c99','#72818a']
def figure(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(O/'figures'/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
def box(ax,text):ax.axis('off');ax.text(0,1,text,va='top',transform=ax.transAxes,linespacing=1.55)
def main():
    ann=pd.read_csv(B/'annotations/gloss_annotations.csv');full=pd.read_csv(F/'image_summary.csv');folds=pd.read_csv(B/'protocols/gloss_group_folds.csv')
    rawpath=M/'responses/internvl35_4b/full.jsonl';raw=[json.loads(l) for l in rawpath.read_text(encoding='utf-8').splitlines() if l.strip()]
    ids=[r['request_id'] for r in raw];assert len(ids)==len(set(ids))
    one=[r for r in raw if r['task']=='T1' and r['condition']=='main' and r['parse_status']=='valid'];two=[r for r in raw if r['task']=='T2' and r['parse_status']=='valid']
    single=pd.DataFrame([dict(image_id=r['image_ids'][0],single_score=r['parsed']['glossiness']) for r in one]);assert len(single)==3888 and single.image_id.nunique()==3888
    d=full.merge(single,on='image_id',validate='one_to_one').merge(ann[['image_id','source_image_index','image_512_path','hue','chroma']],on='image_id',validate='one_to_one').merge(folds[['image_id','shape_fold','illumination_fold']],on='image_id',validate='one_to_one')
    h=d[d.cohort=='heldout_matching'].copy().reset_index(drop=True);assert len(h)==3849
    plan=dict(created=time.time(),scope='Existing data only; no new collection or positional checks. InternVL primary, other models and tiny analysis supplementary.',main_figures=['Task and acquisition design','Human agreement and physical disagreement','Shared illumination and shape effects','Cross-task prediction'],primary_cohort='3849 matching-heldout images, all 3888 tabulated separately',continuous_cross_task='Two directions isotonic regression fitted solely to model outputs; existing 12 shape and 12 illumination group folds, development images excluded. No human labels in calibration.',pair_cross_task='T1 and matching differences predict collected T2 choices; 5 pair-group-heldout stratified folds; both display orders grouped; same images may appear in training and test pairs. Logistic regression C=1, no intercept, training std scaling, no human labels in fit.',pair_scoring='Canonical a selection frequency from two display orders; fractional ties. Raw directional prediction gets half credit for predicted ties. Brier error targets canonical pair mean, not individual human choices.',pair_sampling='1000 human/physical-discordant plus 1000 random pairs from existing selection; not a random sample of all pairs. Human pair preferences are derived from mean human matching scores, not newly measured binary responses.',brain_alignment='Not included',triplet='Existing paused exploratory data retained outside the main three-task analysis; no deletion or claim of positive triplet evidence.',journal_scope='Candidate main-figure package, not assertion of Nature-family acceptance readiness')
    if not (O/'methods/analysis_plan.json').exists():save(O/'methods/analysis_plan.json',plan)
    status('Analyzing continuous cross-task agreement')
    d.to_csv(O/'source_data/image_level_all3888.csv',index=False)
    out=[];cont=[];splits=[]
    for factor in ['shape','illumination']:
        for source,target in [('single_score','model_c'),('model_c','single_score')]:
            pred=np.zeros(len(h));baseline=np.zeros(len(h))
            for fold in range(12):
                test=h[factor+'_fold'].to_numpy()==fold;train=~test
                f=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(h.loc[train,source],h.loc[train,target]);pred[test]=f.predict(h.loc[test,source]);baseline[test]=h.loc[train,target].mean()
                splits.append(dict(split=factor,fold=fold,source=source,target=target,train_image_ids=h.image_id[train].tolist(),test_image_ids=h.image_id[test].tolist()))
            y=h[target].to_numpy();mse=np.mean((y-pred)**2)
            cont.append(dict(split=factor,source=source,target=target,n=len(h),r2=float(1-mse/np.var(y)),mse=float(mse),mae=float(np.mean(abs(y-pred))),spearman=rho(y,pred),skill_vs_training_mean=float(1-mse/np.mean((y-baseline)**2))))
            out.extend(dict(image_id=h.image_id.iloc[j],split=factor,source=source,target=target,observed=float(y[j]),predicted=float(pred[j])) for j in range(len(h)))
    pd.DataFrame(cont).to_csv(O/'tables/Table3_continuous_cross_task.csv',index=False);pd.DataFrame(out).to_csv(O/'source_data/continuous_cross_task_oof.csv',index=False);save(O/'methods/continuous_split_audit.json',splits)
    # Reconstruct canonical pair choices directly from the original model responses.
    status('Analyzing pair predictions from independent task outputs')
    pairs=pd.read_csv(M/'manifests/pairs2000_analysis_only.csv');index=d.set_index('source_image_index');groups={}
    for r in two:groups.setdefault(int(r['group'].removeprefix('pair')),[]).append(r)
    rec=[]
    for pid,p in pairs.iterrows():
        rows=groups.get(pid,[]);assert len(rows)==2
        a=int(p.a);b=int(p.b);assert {tuple(r['indices']) for r in rows}=={(a,b),(b,a)}
        yy=[]
        for r in rows:
            choice=r['parsed']['choice'];picked=None if choice=='tie' else r['indices'][ord(choice)-65]
            yy.append(.5 if picked is None else float(picked==a))
        ai=index.loc[a+1];bi=index.loc[b+1]
        rec.append(dict(pair_id=int(pid),stratum=p.stratum,a_id=ai.image_id,b_id=bi.image_id,a_index=a,b_index=b,t2_probability_a=np.mean(yy),order1_a=yy[0],order2_a=yy[1],swap_consistent=yy[0]==yy[1],delta_single=ai.single_score-bi.single_score,delta_matching=ai.model_c-bi.model_c,delta_human=ai.human_c-bi.human_c,delta_physical=ai.physical_c-bi.physical_c,matching_heldout_both=ai.cohort=='heldout_matching' and bi.cohort=='heldout_matching'))
    p=pd.DataFrame(rec);preds={};pairfold=np.empty(len(p),int)
    cv=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20260928).split(p,p.stratum))
    for name,cols in [('single',['delta_single']),('matching',['delta_matching']),('joint',['delta_single','delta_matching'])]:
        x=p[cols].to_numpy();pred=np.zeros(len(p));y=p.t2_probability_a.to_numpy()
        for fold,(train,test) in enumerate(cv):
            std=x[train].std(axis=0);std[std<1e-12]=1
            xx=x[train]/std;classifier=LogisticRegression(C=1,fit_intercept=False,solver='lbfgs',max_iter=1000)
            classifier.fit(np.vstack([xx,xx]),np.r_[np.ones(len(train)),np.zeros(len(train))],sample_weight=np.r_[2*y[train],2*(1-y[train])])
            pred[test]=classifier.predict_proba(x[test]/std)[:,1];pairfold[test]=fold
        p[name+'_cv_probability']=pred;preds[name]=pred
    p['pair_cv_fold']=pairfold
    pairmetrics=[]
    for stratum,sel in [('all',np.ones(len(p),bool)),('random',p.stratum.eq('random').to_numpy()),('discordant',p.stratum.eq('discordant').to_numpy()),('both_matching_heldout',p.matching_heldout_both.to_numpy())]:
        z=p[sel];y=z.t2_probability_a.to_numpy()
        for name,col in [('single','delta_single'),('matching','delta_matching'),('human_reference','delta_human'),('physical_reference','delta_physical')]:
            x=z[col].to_numpy();direction=np.sign(x);direction[np.abs(x)<1e-10]=0
            score=.5+direction*(y-.5)
            item=dict(stratum=stratum,predictor=name,n_pairs=len(z),agreement_with_observed_choices=float(score.mean()),predictor_tie_fraction=float((direction==0).mean()),brier_cv=None,brier_skill_vs_half=None)
            if name in preds:
                err=np.mean((y-z[name+'_cv_probability'].to_numpy())**2);item.update(brier_cv=float(err),brier_skill_vs_half=float(1-err/np.mean((y-.5)**2)))
            pairmetrics.append(item)
        err=np.mean((y-z.joint_cv_probability.to_numpy())**2);pairmetrics.append(dict(stratum=stratum,predictor='joint',n_pairs=len(z),agreement_with_observed_choices=None,predictor_tie_fraction=None,brier_cv=float(err),brier_skill_vs_half=float(1-err/np.mean((y-.5)**2))))
    p.to_csv(O/'source_data/pair_cross_task_predictions.csv',index=False);pd.DataFrame(pairmetrics).to_csv(O/'tables/Table4_pair_cross_task.csv',index=False)
    # Main agreement statistics, keeping sample denominators explicit.
    scores=[]
    for cohort,z in [('matching_heldout',h),('all',d)]:
        for col in ['single_score','model_c','physical_c']:scores.append(dict(cohort=cohort,model_output=col,n=len(z),human_spearman=rho(z.human_c,z[col]),human_pearson=float(pearsonr(z.human_c,z[col]).statistic)))
    pd.DataFrame(scores).to_csv(O/'tables/Table2_human_agreement.csv',index=False)
    shared=read(F/'shared_condition_effects/summary.json');matching=read(F/'summary.json');pairhuman=pd.read_csv(M/'statistics/pair_summary.csv');pairhuman=pairhuman[pairhuman.model=='internvl35_4b'];pairhuman.to_csv(O/'tables/Table5_human_pair_reference.csv',index=False)
    summary=dict(n_single_images=len(single),n_pairs=len(p),n_pair_responses=len(two),n_matching_images=len(full),n_matching_trials=matching['completed_main_trials'],primary_n=len(h),single_matching_spearman=rho(h.single_score,h.model_c),single_matching_pearson=float(pearsonr(h.single_score,h.model_c).statistic),human_agreement=scores,continuous_cross_task=cont,pair_cross_task=pairmetrics,matching_heldout_stats=matching['statistics']['heldout_matching'],shared_condition_effects=shared['conditions'],pair_cv_scope='held-out pairs, not held-out identities',limitations=plan)
    mt=[json.loads(l) for l in (F/'trials.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    summary['n_matching_decisions']=sum(t['decisions'] for t in mt)
    save(O/'summary.json',summary)
    # Full prompts and model/version provenance.
    prompts=read(M/'protocol/prompts.json');mp=read(F/'protocol.json')
    protocolrows=[dict(task='Single-image rating',stimuli='one original scene image',n_units=3888,n_responses=3888,prompt=prompts['T1'],system=prompts['system'],response='integer 0..100'),dict(task='Direct pair choice',stimuli='two test images, both presentation orders',n_units=2000,n_responses=4000,prompt=prompts['T2'],system=prompts['system'],response='A/B/tie'),dict(task='Reference matching',stimuli='test image A + one reconstructed reference B; adaptive levels',n_units=3888,n_responses=7776,prompt=mp['prompt'],system=mp['system'],response='A/B/SAME, converted to search threshold c')]
    protocolrows[2]['completed_matches']=7776
    protocolrows[2]['n_responses']=summary['n_matching_decisions']
    pd.DataFrame([dict(task=r['task'],condition=r['condition'],parse_status=r['parse_status']) for r in raw]).value_counts().rename('n').reset_index().to_csv(O/'supplement/S0_all_existing_task_inventory.csv',index=False)
    pd.DataFrame(protocolrows).to_csv(O/'tables/Table1_tasks_and_prompts.csv',index=False)
    save(O/'methods/prompts_verbatim.json',protocolrows)
    for src,dst in [(M/'responses/internvl35_4b/full_environment.json','internvl_single_pair_environment.json'),(F/'protocol.json','matching_protocol.json'),(M/'models/internvl35_4b/manifest.json','model_manifest.json')]:
        if src.exists():shutil.copyfile(src,O/'methods'/dst)
    if (B/'models/internvl35_4b/manifest.json').exists():shutil.copyfile(B/'models/internvl35_4b/manifest.json',O/'methods/model_manifest.json')
    status('Rendering main figures')
    # Figure 1: actual, indexed examples and concise acquisition design.
    fig=plt.figure(figsize=(12,6.6));gs=fig.add_gridspec(2,3,height_ratios=[.85,1],hspace=.14,wspace=.18)
    def img(path):return np.asarray(Image.open(path).convert('RGB'))
    def montage(paths):
        ims=[np.asarray(Image.open(x).convert('RGB').resize((256,256))) for x in paths]
        if len(ims)==1:ims=[np.full((256,128,3),255,dtype=np.uint8),ims[0],np.full((256,128,3),255,dtype=np.uint8)]
        return np.concatenate(ims,axis=1)
    first=one[0];pair=groups[0][0];trial=json.loads((F/'trials.jsonl').read_text().splitlines()[0]);k=trial['start']
    examples=[([B/first['image_paths'][0]],'A  Single-image rating'),([B/x for x in pair['image_paths']],'B  Direct pair choice'),([B/trial['image_path'],B/f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png'],'C  Reference matching')]
    for j,(paths,title) in enumerate(examples):
        ax=fig.add_subplot(gs[0,j]);ax.imshow(montage(paths));ax.set_title(title,loc='left',fontweight='bold');ax.axis('off')
    box(fig.add_subplot(gs[1,0]),'"Rate the apparent glossiness..."\n\nOne scene image → integer 0–100\nActual example: '+str(first['parsed']['glossiness'])+'\n\n3,888 images; one main rating each\nFresh context; greedy decoding\nNo human ratings in model prompts')
    box(fig.add_subplot(gs[1,1]),'"Which object’s surface looks glossier?"\n\nTwo images → A / B / tie\nActual displayed-order choice: '+pair['parsed']['choice']+'\n\n2,000 pairs × two display orders\n1,000 random + 1,000 discordant pairs\nHuman reference derived from ratings')
    trace=' → '.join(map(str,trial['trace_indices']))
    box(fig.add_subplot(gs[1,2]),'"Which object looks glossier?"\n\nTest + reference → A / B / SAME\n52 reconstructed reference levels\n\nExample search: '+trace+'\nFinal level '+str(trial['final_presented_index'])+f"; c = {trial['matching_c']:.4f}"+'\n3,888 images × two starts; ≤7 steps\nHuman labels used only for analysis')
    figure(fig,'Fig1_tasks_and_acquisition')
    save(O/'source_data/Fig1_examples.json',dict(single=first,pair=pair,matching=trial,reference_index=k,selection='First recorded example in each task, not selected by performance'))
    # Figure 2: human agreement and disagreement with physical ordering.
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for a,col,title,yl in [(ax[0,0],'single_score','A  Single-image judgment','InternVL rating (0–100)'),(ax[0,1],'model_c','B  Reference matching','InternVL matched c')]:
        hb=a.hexbin(h.human_c,h[col],gridsize=27,mincnt=1,cmap='Blues');fig.colorbar(hb,ax=a,label='Image count',shrink=.8);a.set(xlabel='Human matched c',ylabel=yl,title=title+f'\nrho = {rho(h.human_c,h[col]):.3f}; n = {len(h):,}')
    a=ax[1,0];stats=matching['statistics']['heldout_matching'];vals=[stats['human_physical_spearman'],stats['human_model_spearman'],stats['partial_pearson_linear_physical_c']]
    a.bar(range(3),vals,color=[C[3],C[0],C[2]]);a.set(xticks=range(3),xticklabels=['Human vs physical\nSpearman','Human vs model\nSpearman','Physical adjusted\npartial Pearson'],ylim=(0,1),ylabel='Association',title='C  Beyond a linear physical-c relationship');a.tick_params(axis='x',labelsize=8)
    for j,v in enumerate(vals):a.text(j,v+.025,f'{v:.3f}',ha='center')
    a=ax[1,1];order=pairhuman.set_index('stratum').loc[['random','discordant']]
    v=order.follow_human.to_numpy();a.bar(range(2),v,color=[C[0],C[1]]);a.errorbar(range(2),v,yerr=[v-order.ci_low.to_numpy(),order.ci_high.to_numpy()-v],fmt='none',color='black',capsize=4);a.axhline(.5,color='gray',ls='--');a.set(xticks=[0,1],xticklabels=['Random pairs\nn = 1,000','Human–physical discordant\nn = 1,000'],ylim=(0,1),ylabel='Choice agreement with human rating order',title='D  Direct choices track human appearance')
    for j,vv in enumerate(v):a.text(j,.9,f'{vv:.1%}',ha='center')
    figure(fig,'Fig2_human_alignment')
    # Figure 3: published adjusted-condition estimates from existing full analysis.
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for a,dim,title in zip(axes,['illumination_id','shape_id'],['A  Shared illumination effects','B  Shared shape effects']):
        e=pd.read_csv(F/f'shared_condition_effects/{dim}_adjusted_effects.csv');e.to_csv(O/'source_data'/f'Fig3_{dim}.csv',index=False)
        xx=e.human_adjusted_sd.to_numpy();yy=e.model_adjusted_sd.to_numpy()
        a.errorbar(xx,yy,xerr=[np.maximum(0,xx-e.human_ci_low),np.maximum(0,e.human_ci_high-xx)],yerr=[np.maximum(0,yy-e.model_ci_low),np.maximum(0,e.model_ci_high-yy)],fmt='o',ms=4,color=C[0],ecolor='#c5d7dd',elinewidth=.8)
        # Condition IDs and every confidence interval remain in the source tables.
        a.axhline(0,color='gray',lw=.7);a.axvline(0,color='gray',lw=.7);a.set(xlabel='Human adjusted effect (SD)',ylabel='InternVL adjusted effect (SD)',title=title+f"\nrho = {shared['conditions'][dim]['effect_spearman']:.3f}; 36 conditions")
    figure(fig,'Fig3_shared_condition_effects')
    # Figure 4: cross-task prediction does not use human labels.
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    a=ax[0,0];hb=a.hexbin(h.single_score,h.model_c,gridsize=25,mincnt=1,cmap='Blues');fig.colorbar(hb,ax=a,label='Image count',shrink=.8);a.set(xlabel='Single-image rating',ylabel='Reference matched c',title=f"A  Cross-task association\nrho = {summary['single_matching_spearman']:.3f}; n = {len(h):,}")
    a=ax[0,1];ct=pd.DataFrame(cont)
    for j,source in enumerate(['single_score','model_c']):
        vv=[ct[(ct.split==f)&(ct.source==source)].r2.iloc[0] for f in ['shape','illumination']];a.bar(np.arange(2)+(j-.5)*.32,vv,width=.32,color=C[j],label='Rating → matching' if j==0 else 'Matching → rating')
    a.set(xticks=[0,1],xticklabels=['Held-out shapes','Held-out illuminations'],ylim=(0,.88),ylabel='Held-out model-output R²',title='B  Predicting one task from the other');a.axhline(0,color='gray',lw=.5);a.legend(fontsize=8)
    a=ax[1,0];pm=pd.DataFrame(pairmetrics)
    for j,source in enumerate(['single','matching']):
        vv=[pm[(pm.stratum==s)&(pm.predictor==source)].agreement_with_observed_choices.iloc[0] for s in ['random','discordant']];a.bar(np.arange(2)+(j-.5)*.32,vv,width=.32,color=C[j],label='Rating order' if j==0 else 'Matching order')
    a.axhline(.5,color='gray',ls='--');a.set(xticks=[0,1],xticklabels=['Random pairs','Discordant pairs'],ylim=(0,1),ylabel='Agreement with actual InternVL pair choices',title='C  Cross-task directional prediction');a.legend(fontsize=8)
    a=ax[1,1]
    for j,name in enumerate(['single','matching','joint']):
        pp=p[name+'_cv_probability'];bins=pd.cut(pp,np.linspace(0,1,9),include_lowest=True);z=pd.DataFrame(dict(pred=pp,y=p.t2_probability_a,bin=bins)).groupby('bin',observed=True).agg(pred=('pred','mean'),observed=('y','mean'),n=('y','size'));z.to_csv(O/'source_data'/f'Fig4_calibration_{name}.csv',index=False);a.plot(z.pred,z.observed,'o-',color=C[j],label=name)
    a.plot([0,1],[0,1],ls='--',color='gray');a.set(xlabel='Held-out predicted P(choose canonical a)',ylabel='Observed frequency across display orders',title='D  Pair-held-out choice prediction',xlim=(0,1),ylim=(0,1));a.legend(fontsize=8)
    figure(fig,'Fig4_cross_task_consistency')
    # Supplementary existing comparisons, kept out of the central evidence chain.
    for src,name in [(M/'matching_qwen3vl4b_pilot12/comparison/same12_metrics.csv','S1_other_models_and_tiny_same12.csv'),(F/'incremental_tiny/overall_metrics.csv','S2_incremental_tiny_metrics.csv'),(F/'incremental_tiny/summary.json','S2_incremental_tiny_summary.json')]:shutil.copyfile(src,O/'supplement'/name)
    captions='''# Main figure captions / 主图说明

**Figure 1. Three behavioral readouts from a frozen multimodal model.** Actual first-recorded stimuli and responses illustrate single-image rating, direct pair choice, and adaptive reference matching. Full prompts, system messages, response schemas and sampling counts are provided in Table 1. Reference images are reconstructed, not the original author reference set. Two deterministic starts are not independent simulated human observers. / 三种问答范式及真实输入输出实例，强调提示词、返回格式、搜索控制和数据记录的可复现性。

**Figure 2. Alignment with human gloss judgments.** Panels A–C use 3,849 matching-heldout images; the matching protocol's 39 development images are excluded. Panel C distinguishes rank correlations from partial Pearson correlation controlling linear physical c. Panel D uses two presentation orders of 1,000 random and 1,000 human–physical-discordant pairs; the human reference is derived from published mean matching judgments. Error bars reproduce the existing crossed-bootstrap uncertainty and do not treat pair observations as independent human trials. / 全量一致性、超出物理参数的关联及冲突对上的判断方向。

**Figure 3. Shared illumination- and shape-dependent judgment patterns.** Adjusted effects use the existing exploratory analysis of all 3,888 images, controlling categorical physical levels, viewpoint, chroma, hue harmonics and the other condition factor. Points are 36 condition effects, not individual images. Bars are pointwise cluster-bootstrap intervals; inferential simultaneous bands are retained in source data. Reported associations do not imply shared causal mechanisms. / 共同的照明与形状效应，使用全部数据，与 Fig.2 留出子集的分母不同。

**Figure 4. Cross-task consistency without fitting human ratings.** Panel A compares independently collected scalar outputs. Panel B fits monotonic mappings using model outputs on training conditions and predicts held-out shapes or illuminations. Panel C predicts observed pair choices directly from rating or matching order, with half credit for predictor ties. Panel D uses five-fold pair-held-out logistic prediction; both presentation orders stay together, but image identities can recur across folds. These results test a shared gloss-related judgment component, not an identified internal representation or novel-image pair generalization. / 跨任务预测校准完全不使用人类评分；区分原始方向预测、留出条件映射和留出图对预测。
'''
    (O/'methods/figure_captions.md').write_text(captions,encoding='utf-8')
    methods='''# 方法与可复现范围

人类行为数据来自 Morimoto 等人的既有光泽匹配实验；本工作没有新增人类实验。InternVL3.5-4B 冻结，不使用本数据集人类标签微调。模型实际看见的是有背景的图像；作者小网络对照输入不同，应在补充材料明确。

三种问答协议的完整原文保存在 prompts_verbatim.json 与 Table1_tasks_and_prompts.csv。单图任务输出 0–100 整数；直接比较输出 A/B/tie；参考匹配输出 A/B/SAME，控制器使用固定随机起始档位与最多七步二分搜索，将最后呈现的 k 转换为 (k−1)×0.1487/51。终止可来自 SAME、边界、区间耗尽或步数限制，不能把所有匹配值都称为模型明确判断相等。

单图与双图源文件包含已暂停的三图探索响应；本包主分析仅使用预先确定的三种任务，三图探索数据未删除，不能声称所有曾运行任务都得到正向证据。既有协议并非对外预注册，本次跨任务分析属于后续探索性分析。matching-heldout 指相对于匹配协议开发的留出，不等于所有项目决策从未接触这些图。

跨任务连续映射采用递增 isotonic regression，独立按形状和照明各12折留出；只在训练组拟合模型输出之间的映射，人类分数不参与。两个方向均完整报告。直接比较采用原始两种呈现顺序映射到统一图像ID后的选择频率，单图与参考匹配各自的差值用于预测它；逻辑回归只用训练图对的模型选择，测试图对不重复，但图像身份可重复。数据划分、逐图和逐对预测全部保存。

主图以 InternVL 为核心。GPT/Qwen 的 12 图先导结果和小网络增量检验置于补充材料，不作为所有模型已完成全量条件验证的证据。“自然涌现”在这里若使用，应操作化为无需本任务行为标签专门训练即可完成判断；未检验训练轨迹或参数规模阈值。当前只支持 gloss，不概括为全部材料属性。

参考文献 Human-like object concept representations emerge naturally in multimodal large language models 的行为部分还包含稳定性、留出选择预测和可解释结构，不只是一次模型—人类相关。本包借鉴其行为证据组织方式，不纳入脑对齐，也不因题目相似就宣称达到同一期刊发表标准。

复现：从项目根目录执行 mllm_runtime/Scripts/python.exe scripts/build_paper_figures_v1.py。不调用模型、不下载数据、不改变旧响应文件。原始素材许可和重建参考来源遵从原项目记录。最终投稿前仍需审查方法新颖性、数据范围与机制解释的强度。
'''
    (O/'methods/Methods_and_scope.md').write_text(methods,encoding='utf-8')
    shutil.copyfile(Path(__file__),O/'code'/Path(__file__).name)
    sources=[rawpath,F/'image_summary.csv',F/'trials.jsonl',M/'protocol/prompts.json',M/'manifests/pairs2000_analysis_only.csv',B/'annotations/gloss_annotations.csv',B/'protocols/gloss_group_folds.csv',F/'shared_condition_effects/summary.json',M/'statistics/pair_summary.csv']
    save(O/'methods/input_manifest.json',{str(p.relative_to(B)):sha(p) for p in sources})
    write_report(summary,captions)
    save(O/'artifact_manifest.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.suffix not in ['.zip'] and p.name not in ['artifact_manifest.json','status.json']})
    status('complete')
    print(json.dumps(dict(single_matching_rho=summary['single_matching_spearman'],continuous=cont,pair=pairmetrics),indent=2),flush=True)
def write_report(s,captions):
    note=['# InternVL 主论文图表包 v1','',f"已完成四张主图、五张主表和对应源数据。单图评分 {s['n_single_images']} 张，双图 {s['n_pairs']} 对、{s['n_pair_responses']} 次回答，参考匹配 {s['n_matching_images']} 张、{s['n_matching_trials']} 次。没有新增采集。",'',f"单图评分与参考匹配跨任务相关 rho={s['single_matching_spearman']:.3f}。下面分别列出留出预测，不将相关性替代预测验证。",'', '## 留出形状/照明的任务间预测','']
    for r in s['continuous_cross_task']:note.append(f"- {r['split']}：{r['source']} → {r['target']}，R²={r['r2']:.3f}，rho={r['spearman']:.3f}。")
    note+=['','## 实际双图选择能否由其他任务预测','']
    for r in s['pair_cross_task']:
        if r['stratum']=='all':note.append(f"- {r['predictor']}：方向一致率={r['agreement_with_observed_choices'] if r['agreement_with_observed_choices'] is not None else '不适用'}；留出图对 Brier={r['brier_cv']}。")
    note+=['','## 论文贡献与范围','', '可以围绕“任务化提示词与参考匹配采集方法，使冻结的多模态模型表现出人类式光泽判断；这种一致性延伸到形状/照明规律和不同问答任务”组织正文。提示词设计和采集流程属于可复现方法贡献；若没有与其他方法的对照，不能单独宣称该提示词最优。','', '是否构成强有力的跨任务证据应同时看图对预测和留出映射，不只挑选较高相关。主结论仍是行为层面的光泽判断，不是已经解析内部材料表征，也不是证明了所有材料属性能力。','', '本包已能形成一份有清晰论证结构的论文结果初稿；能否达到目标期刊要求还取决于新颖性、证据范围与审稿，不能从正相关直接保证。下一步优先审阅这四张主图和主张边界，而不是自动扩大采集。']
    (O/'论文结果初稿与审阅说明.md').write_text('\n'.join(note),encoding='utf-8')
    body='<meta charset="utf-8"><title>InternVL 主论文图表</title><style>body{font:18px system-ui;margin:35px;max-width:1250px;line-height:1.65;color:#17333e}img{max-width:100%;border:1px solid #eee}pre{white-space:pre-wrap}a{color:#176b88}</style><h1>Human-like gloss judgments in a pretrained multimodal model</h1><p>InternVL 主图包 v1 · 既有数据分析 · 未新增采集 · 小网络与其他模型置于补充材料</p><p>单图评分与参考匹配相关：<b>'+f"{s['single_matching_spearman']:.3f}"+'</b>。跨任务校准仅使用模型输出，人类评分只用于人类一致性分析。</p>'
    for name,title in [('Fig1_tasks_and_acquisition','Fig. 1 · 问答范式与数据采集'),('Fig2_human_alignment','Fig. 2 · 人类一致性及偏离物理值的判断'),('Fig3_shared_condition_effects','Fig. 3 · 共同照明与形状效应'),('Fig4_cross_task_consistency','Fig. 4 · 跨任务判断与留出预测')]:body+='<h2>'+title+'</h2><img src="figures/'+name+'.png"><p><a href="figures/'+name+'.pdf">PDF</a> · <a href="figures/'+name+'.svg">SVG</a></p>'
    body+='<h2>结果与投稿定位</h2><pre>'+html.escape('\n'.join(note))+'</pre><details><summary>完整图注及方法边界</summary><pre>'+html.escape(captions)+'</pre></details><h2>表格与复现</h2>'
    for p in sorted((O/'tables').glob('*.csv')):body+='<p><a href="tables/'+p.name+'">'+p.name+'</a></p>'
    body+='<p><a href="methods/Methods_and_scope.md">方法说明</a> · <a href="methods/prompts_verbatim.json">完整提示词</a> · <a href="paper_figures_reproducibility.zip">复现包</a></p>'
    (O/'report.html').write_text(body,encoding='utf-8')
if __name__=='__main__':main()
