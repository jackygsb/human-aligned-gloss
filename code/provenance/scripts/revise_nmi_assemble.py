from pathlib import Path
import re,shutil,json,tokenize,io
B=Path(__file__).resolve().parents[1];O=B/'mllm_material/nmi_revision_v8/source';V=B.parent/'tmp/pdfs/nmi_clean_compile/NMI_Chinese_Manuscript'
extras=[
('align2021','Jia, Chao and Yang, Yinfei and Xia, Ye and others','Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision','2021','Proceedings of Machine Learning Research','139','4904--4916','https://proceedings.mlr.press/v139/jia21b.html'),
('blip2023','Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven','BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models','2023','Proceedings of Machine Learning Research','202','19730--19742','https://proceedings.mlr.press/v202/li23q.html'),
('flamingo2022','Alayrac, Jean-Baptiste and Donahue, Jeff and Luc, Pauline and others','Flamingo: a Visual Language Model for Few-Shot Learning','2022','arXiv','','','https://arxiv.org/abs/2204.14198'),
('siglip2023','Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas','Sigmoid Loss for Language Image Pre-Training','2023','arXiv','','','https://arxiv.org/abs/2303.15343'),
('geirhos2021','Geirhos, Robert and Narayanappa, Kantharaju and Mitzkus, Benjamin and others','Partial success in closing the gap between human and machine vision','2021','Advances in Neural Information Processing Systems','34','23885--23899','https://arxiv.org/abs/2106.07411'),
('texture2019','Geirhos, Robert and Rubisch, Patricia and Michaelis, Claudio and Bethge, Matthias and Wichmann, Felix A. and Brendel, Wieland','ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness','2019','International Conference on Learning Representations','','','https://arxiv.org/abs/1811.12231')]
bib=(V/'references.bib').read_text('utf-8')+'\n'+(O/'new_references.bib').read_text('utf-8')+'\n'
for key,author,title,year,journal,volume,pages,url in extras:
 fields=dict(author=author,title='{'+title+'}',year=year,journal=journal,volume=volume,pages=pages,url=url)
 bib+='@article{'+key+',\n'+',\n'.join(' '+k+' = {'+v+'}' for k,v in fields.items() if v)+'\n}\n\n'
(O/'references.bib').write_text(bib,encoding='utf-8')
# Correct Python string escapes in the builder before re-running it.
p=B/'scripts/revise_nmi_text.py';s=p.read_text('utf-8');ts=[]
for t in tokenize.generate_tokens(io.StringIO(s).readline):
 if t.type==tokenize.STRING and '\\rho' in t.string and not t.string.lower().startswith(('r','f')):t=t._replace(string='r'+t.string)
 ts.append(t)
p.write_text(tokenize.untokenize(ts),encoding='utf-8')
exec(compile(p.read_text('utf-8'),str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})
p=O/'main_NMI_submission_v8_red.tex';s=p.read_text('utf-8')
s=s.replace('e，受约束参考匹配任务中的完整行为变化。方向干预能够推动模型匹配值向预测方向移动，而随机方向影响较小。',r'\revision{完整参考匹配干预的逐图配对轨迹见补充图 S6。}')
s=s.replace('}{fig:readout}',r'\revision{阴影为固定折外预测的 2,000 次形状与照明交叉重采样 95\% 逐项区间，随机种子为 20261002；区间条件于已拟合的读出器。}'+ '}{fig:readout}')
s=s.replace('相关差的配对区间见补充图 S4。}',r'相关差的配对区间见补充图 S4；重采样方法参见文献\cite{efron1979,owen2012}。}')
s=s.replace('八项候选相对对照的双侧符号随机化检验采用',r'八项候选相对对照的双侧符号随机化检验采用').replace('并共同 BH 校正，',r'并共同 BH 校正\cite{bh1995}，')
s=s.replace('\n\n\\revision','\n\\revision')
s=s.replace('c、d 的浅色点为',r'a、b 的候选方向阴影为 36 个形状重采样的 95\% 区间。c、d 的浅色点为')
p.write_text(s,encoding='utf-8')
p=O/'supplement_NMI_submission_v8.tex';s=p.read_text('utf-8')
prompt=r'''
匹配 system 提示为：
\begin{quote}\small You are a visual observer performing a surface-gloss matching task. Follow the response format exactly.\end{quote}
每一步的核心问题为：
\begin{quote}\small Compare the apparent surface gloss of the two objects. Which object looks glossier? Judge gloss, not overall brightness. Answer exactly A if Image A looks glossier, B if Image B looks glossier, or SAME if their surface gloss looks indistinguishable.\end{quote}
步骤之间不累积对话，输入身份与控制器状态在外部记录。
'''
s=s.replace(r'\stitle{S2}',prompt+r'\stitle{S2}',1)
s=s.replace('该序列复现原方法的名义范围与场景构造；参考物几何为重建版本，与原始人类实验参考并非逐像素相同。',r'''参考序列采用 RGB 渲染、balanced Ward 镜面项（粗糙度 $\alpha=0.05$）和中性漫反射系数 0.24；512$\times$512 像素，每个渲染基底每像素 1,024 次采样。环境使用 \texttt{doge2.hdr} 庭院，视点方位 315 度、视距 3.3、视场 50 度，球面由固定种子 20260927 的 140 个平滑突起生成。整个序列采用统一曝光及 sRGB 编码。镜面系数按 $\rho_{s,k}=(c_k+(1/16)^{1/3})^3-1/16$ 转换；该基准项沿用数据处理公式，而渲染漫反射另设为 0.24，故本文 $c_k$ 为参考序列的名义标签。庭院选择依据论文示例外观，原作方法文本所记环境为 Uffizi；几何与环境的重建取值均按实际使用版本记录。''')
s=s.replace('基础模型权重保持冻结，行为采集的输入', 'InternVL 将每张 512 像素图编码为单个 448 像素图块；Qwen 使用原生图像处理与 BF16 推理。Qwen 全量协议使用确定性解码（do\_sample=false，最多 64 个输出令牌）。基础模型权重保持冻结，行为采集的输入')
s=s.replace('模型行为均来自已保存的冻结模型响应，本次修订仅增加统计分析与可视化。','模型行为均来自固定的冻结模型响应，分析与图表使用同一版本的数据标识。')
s=s.replace('新增自助抽样次数为 2,000','读出区间另使用种子 20261002；新增自助抽样次数为 2,000')
# Fix SI figure order to keep the numbering in main stable; S4 is intentionally discussed before RSA.
s=s.replace(r'\shead{补充参考文献}',r'\clearpage')
s=s.replace('点表示合并折外预测的 $R^2$，虚线仅连接同一种分析在三个系统中的结果，不表示连续自变量。','每个面板为一个系统，点表示合并折外预测的 $R^2$，连线表示由基线到加性、低秩模型的依次扩展。')
s=s.replace('两点连线表示同一评估集合、同一划分中的模型扩展；其统计评价采用配对预测差值，而非比较两根独立柱。','a，同一评估集合和划分中的模型扩展；b，加入 InternVL 的配对增量及形状与照明交叉聚类 95\\% 区间。')
chunks=re.split(r'(?=\\shead\{S\d+｜)',s);prefix=chunks[0];byid={int(re.search(r'\\shead\{S(\d+)｜',c).group(1)):c for c in chunks[1:]}
order=[2,3,4,5,7,8,9,10,11,6,12,13,14,15,16,17,18,19]
s=prefix+''.join(re.sub(r'(\\shead\{S)\d+',lambda m:m.group(1)+str(new),byid[old],count=1) for new,old in enumerate(order,2))
counter=iter(range(1,20));s=re.sub(r'\\stitle\{S\d+\}',lambda m:r'\stitle{S'+str(next(counter))+'}',s)
p.write_text(s,encoding='utf-8')
keys=set(re.findall(r'@\w+\{([^,]+)',bib));mainkeys=set(k for c in re.findall(r'\\cite\w*\{([^}]+)\}',(O/'main_NMI_submission_v8_red.tex').read_text('utf-8')) for k in c.split(','))
print('Bibliography',len(keys),'main cited',len(mainkeys),'missing',sorted(mainkeys-keys))
(O/'provenance/reference_audit.json').write_text(json.dumps({'library_count':len(keys),'main_cited_count':len(mainkeys),'missing_keys':sorted(mainkeys-keys),'uncited_in_main':sorted(keys-mainkeys)},indent=2),encoding='utf-8')
