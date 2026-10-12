"""Post-run integrity audit and scientifically bounded Chinese interpretation."""
from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];O=B/'mllm_material/mechanisms_v1'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.loads((O/'artifact_manifest.json').read_text());bad=[r['path'] for r in manifest if digest(O/r['path'])!=r['sha256']];assert not bad,bad
 print('Verified existing',len(manifest),'hashes',flush=True)
 d=pd.read_csv(O/'stimuli.csv');plan=json.loads((O/'plan.json').read_text());selection=d[d.image_id.isin(plan['intervention_images'])];audit={'previous_artifact_hashes_verified':len(manifest),'intervention_lighting_ids':sorted(map(int,selection.illumination_id.unique())),'intervention_shape_count':int(selection.shape_id.nunique()),'intervention_views':sorted(map(int,selection.viewpoint_id.unique())),'models':{}}
 for n in ['InternVL','Qwen3']:
  p=O/n;counts={'reference_features':len(list((p/'features').glob('reference_*.npz'))),'original_features':len(list((p/'features').glob('morimoto_????.npz'))),'interventions':len(list((p/'interventions').glob('*.json'))),'matches':len(list((p/'matching').glob('*.json')))};assert list(counts.values())==[52,3888,3024,60];audit['models'][n]=counts
 iv=pd.read_csv(O/'tables/interventions.csv').merge(d[['image_id','shape_id']],on='image_id');rng=np.random.default_rng(20260929);rows=[]
 for n,g in iv.groupby('model'):
  p=g.groupby(['shape_id','direction','dose']).delta_p.mean().unstack(['direction','dose'])
  for control in ['brightness','random0','random1','random2']:
   v=((p[('candidate',1.)]-p[('candidate',-1.)])-(p[(control,1.)]-p[(control,-1.)])).to_numpy();boot=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(4000)]);rows.append(dict(model=n,control=control,n_shape_clusters=len(v),effect=v.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
 pd.DataFrame(rows).to_csv(O/'tables/shape_cluster_intervention_contrasts.csv',index=False)
 audit['all_unrestricted_top_tokens_valid']=bool(iv.unrestricted_top_is_candidate.all());audit['raw_npz_bytes']=sum(p.stat().st_size for p in O.rglob('*.npz'));audit['lightweight_zip_integrity']=zipfile.ZipFile(O/'mechanisms_reproducibility.zip').testzip() is None
 (O/'validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');shutil.copy2(__file__,O/'code'/Path(__file__).name)
 text='''# 内部响应机制实验：结果解读\n\n## 已完成\n\n每模型 3,888 张原始刺激 + 52 张参考图的指定层响应；72 张图双顺序共 3,024 条干预记录；12 张图五种设置共 60 次完整匹配。两模型合计 7,880 份主特征、6,048 条干预、120 次匹配。模型权重未更新。\n\n## 主要发现一：回答前的内部响应可预测人类判断\n\n固定最后语言层的读出结果（不是从测试集挑选最佳层）：InternVL 在形状/照明分组留出时，人类评分预测 Spearman=0.752/0.773，R²=0.589/0.611；Qwen3 分别为 0.813/0.819，R²=0.668/0.674。读出器是以训练部分人类标签拟合的分析工具，不应与原有零任务微调的直接回答性能混写。\n\n训练部分拟合物理/颜色基线后，加入最后语言层对残差的读出，人类总体评分 R²：InternVL 0.261→0.587（形状）、0.272→0.605（照明）；Qwen3 0.261→0.640、0.272→0.642。这提示可读出的信息超出所选物理/颜色基线，但不能据此排除所有图像特征混杂。\n\n## 主要发现二：不使用人类标签定义的候选方向迁移到测试图\n\n只根据固定参考球的 52 档光泽序列定义视觉映射层方向。该方向在参考序列上与档位相关约 0.998（这是方向来源数据，不是独立泛化证据）；在 3,888 张原始测试图中，与人类评分相关为 InternVL 0.320、Qwen3 0.477，与模型匹配值相关为 0.255、0.506。与整图平均亮度的相关分别为 −0.111、0.091。\n\n空间解释并不相同：InternVL 候选响应与局部对比度平均空间相关约 0.464、梯度幅度 0.422；Qwen3 为 −0.153、−0.120。与作者公开小网络核的空间响应关联弱（−0.049、0.082）。因此还不能说两个大模型复用了同一个高光检测核；可以说存在行为作用相近、具体视觉编码可能不同的候选方向。原核为全量人类监督版本，仅作描述性滤波对照。\n\n## 主要发现三：改变内部响应，会定向改变光泽选择\n\n图像与提示词保持不变，只给测试物体的视觉映射 token 加上或减去候选方向。两种图像位置平均后：\n\n| 模型 | +1 剂量的选择概率变化 | −1 剂量的选择概率变化 |\n|---|---:|---:|\n| InternVL | +9.92 个百分点 | −9.37 个百分点 |\n| Qwen3 | +10.25 个百分点 | −9.87 个百分点 |\n\n半剂量约产生一半效应。候选方向的双向差效应，强于等范数亮度方向及三个随机方向；图像重采样及补充的 36 个形状聚类重采样区间见相应 CSV。随机方向作为本次抽取的三个对照，不能代表所有可能方向的完整零分布。\n\n12 张图的完整匹配也按预期移动：InternVL 平均 c 从 0.11153 变为减方向时 0.10399、加方向时 0.12295；Qwen3 从 0.11711 变为 0.10399、0.12878。原图像和模型权重保持不变。\n\n## 必须和主要结果一起说明的范围\n\n- 此次干预样本覆盖 36 个形状、视角 1/2，但全部来自照明编号 1。按每形状最前两个合格图像的预先规则选择造成这一范围；结果不能直接推广为所有照明下的干预效应。全量层级读出则覆盖全部照明。\n- 概率是 A/B/SAME 条件归一化（SAME 半权）。所有干预的非受限最高概率 token 都是有效候选；三候选概率总质量约 InternVL 0.96、Qwen3 接近 1。\n- 完整匹配使用三候选受限选择，属于机制实验；不是重新采集了原自由生成任务。\n- 内部加性位移可以离开自然激活分布。定向作用支持候选成分参与此任务判断，不足以把它命名为纯光泽神经元或唯一机制，也不证明人脑采用相同机制。\n- 两模型的视觉处理路径不同，但不能把它们视为完全独立语言模型家族。\n\n## 论文可支持的表述\n\n冻结多模态模型的内部响应包含可泛化读出的人类光泽信息；从独立参考序列定义的视觉响应方向，在原始测试图上与光泽判断关联，并在所测固定照明条件下对模型选择及参考匹配产生可控、方向一致且强于本次亮度/随机对照的影响。\n\n这补充了行为一致性之外的功能干预证据。目前最强的是“相关信息存在且候选内部成分参与判断”，而不是已完整揭示光泽算法。\n\n## 复现文件\n\nreport.html 为图文报告，figures 为三张核心机制图及匹配补充图（PNG/SVG/PDF）。tables 保留逐图读出、逐次干预、匹配结果及空间相关。每模型 features 为指定层原始响应、桥接 token、全词表 logits 和提示词元数据；约 10.7 GB 数组保留在本地，未重复装入轻量 zip。artifact_manifest.json 为全部文件的校验清单。mechanisms_reproducibility.zip 包含代码、分析、干预和匹配记录、方法文档及清单。\n'''
 (O/'结果解读.md').write_text(text,encoding='utf-8')
 p=O/'report.html';h=p.read_text(encoding='utf-8');intro='<section style="background:#edf5fa;padding:20px"><h2>完成并核验：核心结果</h2><p>回答前最后语言层预测人类评分的留出相关：InternVL 0.752–0.773；Qwen3 0.813–0.819。小型读出器有监督拟合，基础模型冻结。</p><p>仅由参考序列定义的候选内部方向，增强时使选择测试物体更光泽的概率增加约 10 个百分点，减弱时下降约 9–10 个百分点；作用强于所测等强度亮度与随机对照。完整匹配也同向移动。</p><p><b>干预范围：</b>72 张图覆盖 36 个形状，但全部是照明 1、视角 1/2。不能声称跨全部照明的机制干预已验证。空间特征对照未显示两模型共享同一个小网络式高光核。</p><p><a href="结果解读.md">中文详细结果与范围</a> · <a href="validation.json">数据核验</a> · <a href="mechanisms_reproducibility.zip">轻量复现包</a></p></section>'
 h=h.replace('<h2>1 信息定位</h2>',intro+'<h2>1 信息定位</h2>');p.write_text(h,encoding='utf-8')
 (O/'Methods.md').write_text((O/'Methods.md').read_text(encoding='utf-8')+'\n\nPost-run coverage audit: the predetermined first-two-IDs-per-shape intervention selection contains 36 shapes, illumination ID 1 only, and viewpoints 1/2. Intervention generalization across illumination has not been established. Supplementary shape-cluster paired-contrast intervals (36 clusters, 4,000 bootstraps) are in tables/shape_cluster_intervention_contrasts.csv. These do not replace the original retained image-bootstrap estimates.\n',encoding='utf-8')
 # Reuse verified unchanged hashes; update altered/new artifacts, avoiding re-reading all raw arrays.
 old={r['path']:r for r in manifest};updated=[]
 changed={'report.html','Methods.md'}
 for p in O.rglob('*'):
  if not p.is_file() or p.name in ['artifact_manifest.json','mechanisms_reproducibility.zip','pipeline_status.json'] or p.name.endswith('.log'):continue
  key=str(p.relative_to(O));updated.append(old[key] if key in old and key not in changed else dict(path=key,bytes=p.stat().st_size,sha256=digest(p)))
 (O/'artifact_manifest.json').write_text(json.dumps(updated,indent=2),encoding='utf-8')
 with zipfile.ZipFile(O/'mechanisms_reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.suffix!='.zip' and 'features' not in p.parts and not p.name.endswith('.log'):z.write(p,p.relative_to(O))
 assert zipfile.ZipFile(O/'mechanisms_reproducibility.zip').testzip() is None
 print(pd.DataFrame(rows).to_string(index=False));print('Final verification complete')
if __name__=='__main__':main()
