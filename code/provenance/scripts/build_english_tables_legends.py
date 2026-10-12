from pathlib import Path
import re,shutil
import pandas as pd
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=B/'output/NMI_English_Manuscript_v9';T=O/'tables';E=O/'tables_en';E.mkdir(exist_ok=True)
def esc(s):
 s=str(s);s=''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}'}.get(c,c) for c in s)
 for a,b in [('ρ',r'$\rho$'),('α',r'$\alpha$'),('β',r'$\beta$'),('Δ',r'$\Delta$'),('×',r'$\times$'),('→',r'$\rightarrow$'),('≤',r'$\leq$'),('≥',r'$\geq$'),('∈',r'$\in$'),('−','--'),('²',r'$^2$'),('⁻⁵',r'$^{-5}$'),('‖',r'$\Vert$'),('½',r'$1/2$')]:s=s.replace(a,b)
 return s
def table(n,title,headers,rows,widths):
 spec=''.join(f'p{{{w}mm}}' for w in widths)
 text=r'\par\noindent\textbf{Supplementary Table '+n+' | '+esc(title)+r'}\par\smallskip'+'\n'+r'\begin{center}\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{'+spec+r'}\toprule'+'\n'
 text+=' & '.join(esc(x) for x in headers)+r'\\\midrule'+'\n'
 for row in rows:text+=' & '.join(esc(x) for x in row)+r'\\'+'\n'
 text+=r'\bottomrule\end{tabular}\end{center}'+'\n';(E/f'{n}.tex').write_text(text,encoding='utf-8')
ci=lambda r:f'[{r.ci_low:.3f}, {r.ci_high:.3f}]'
table('S1','Behavioral and internal-analysis sample sizes',['Analysis unit','InternVL','Qwen3'],[['Single-image ratings (images)','3,888','3,888'],['Pairwise choices (pairs; two orders)','2,000','2,000'],['Complete matches (two per image)','7,776','7,776'],['Internal extraction (tests + references)','3,940','3,940'],['Intervention response records','3,024','3,024'],['Complete-match intervention (12 images, five settings)','60','60']],[100,20,20])
table('S2','Matching controller and output definitions',['Item','Definition'],[['Input','Test and current reference image'],['Allowed choices','Test, reference, SAME'],['Maximum comparisons','7'],['Stopping output','Last presented reference c'],['Image estimate','Mean of two starting-point matches'],['Complete-match intervention','Start at level 27; constrained three-answer choice']],[42,102])
d=pd.read_csv(T/'v8_pair_ci.csv');d=d[d.predictor.eq('human')&~d.stratum.eq('all')];table('S3','Image-node intervals for pairwise human-order agreement',['Model','Stratum','Agreement','95% interval'],[[r.model,{'random':'Random','discordant':'Conflict'}[r.stratum],f'{r.estimate:.3f}',ci(r)] for r in d.itertuples()],[30,28,28,50])
d=pd.read_csv(T/'v8_cross_task_ci.csv');table('S4','Continuous cross-task out-of-fold prediction',['Model','Direction','Holdout','R² [95% interval]'],[[r.model,'Rating → match' if r.source=='single_score' else 'Match → rating',r.split,f'{r.estimate:.3f} {ci(r)}'] for r in d.itertuples()],[23,36,27,54])
d=pd.read_csv(T/'v8_condition_tests.csv');table('S5','Fixed condition-effect correspondence',['Model','Factor','ρ [95% interval]','p','q'],[[r.model,r.factor.replace('_id',''),f'{r.estimate:.3f} {ci(r)}',f'{r.p:.3g}',f'{r.q_bh4:.3g}'] for r in d.itertuples()],[22,23,55,21,23])
d=pd.read_csv(T/'rsa.csv');table('S6','Complete RSA family; 36 nodes per comparison',['Space/factor','Pair','ρ','95% interval','q'],[[{'raw':'Raw','physical_color_adjusted':'Adjusted','interaction_only':'Interaction'}[r.space]+'/'+('shape' if r.factor=='shape' else 'light'),r.agent1+'–'+r.agent2,f'{r.rsa:.3f}',ci(r),f'{r.q_bh_18:.3g}'] for r in d.itertuples()],[31,40,13,39,20])
d=pd.read_csv(T/'v8_agreement_ci.csv');labels={'InternVL_single':'InternVL rating','InternVL_matching':'InternVL matching','Qwen3_single':'Qwen3 rating','Qwen3_matching':'Qwen3 matching','physical':'Physical parameter'};table('S7','Human-agreement effect sizes',['Output','Spearman ρ','95% interval'],[[labels[r.measure],f'{r.estimate:.3f}',ci(r)] for r in d.itertuples() if r.measure in labels],[62,32,45])
d=pd.read_csv(T/'mechanism_readout.csv');rows=[]
for model in ['InternVL','Qwen3']:
 for split in ['shape','illumination']:
  z=d[(d.model==model)&(d.split==split)&d.layer.eq('language_36')].set_index('target');rows.append([model,split,f'{z.loc["human","spearman"]:.4f}',f'{z.loc["human","r2"]:.4f}',f'{z.loc["human_beyond_physical_color","r2"]:.4f}'])
table('S8','Final sampled language-block readout',['Model','Holdout','Human ρ','Human R²','Combined R²'],rows,[23,30,26,26,32])
d=pd.read_csv(T/'v8_intervention_tests.csv');table('S9','Paired shape-level candidate-minus-control contrasts',['Model','Control','Difference','95% interval','q'],[[r.model,{'brightness':'Brightness','random0':'Random 1','random1':'Random 2','random2':'Random 3'}[r.control],f'{r.estimate:.3f}',ci(r),f'{r.q_bh8:.1g}'] for r in d.itertuples()],[23,26,25,44,22])
# Keep complete new figure legends and all original new panel references.
text=r'\section*{Main figures}'+'\n'+r'\noindent Figure 1 is reproduced on continued pages to preserve the detailed prompts and methods at readable scale. Its editable full-length master is included in the source package.'+'\n'
text+=r'\clearpage\noindent\textbf{Figure 1a--c}\par\begin{center}\includegraphics[width=\textwidth]{figures/Fig1a.pdf}\par\medskip\includegraphics[width=\textwidth,height=.73\textheight,keepaspectratio]{figures/Fig1bc.pdf}\end{center}'+'\n'
text+=r'\clearpage\noindent\textbf{Figure 1d,e (continued)}\par\begin{center}\includegraphics[width=\textwidth,height=.40\textheight,keepaspectratio]{figures/Fig1d.pdf}\par\medskip\includegraphics[width=\textwidth,height=.51\textheight,keepaspectratio]{figures/Fig1e.pdf}\end{center}'+'\n'
for n in ['1f','1g','1h']:text+=r'\figpage{1 (continued)}{Fig'+n+'}\n'
cap=(M/'figure1_english_v6/Figure1_caption_EN.txt').read_text(encoding='utf-8');text+=r'\figlegend{1}{'+esc(cap)+'}\n'
for n in range(2,7):
 root=M/f'figure{n}_english_v1';p=next(root.glob('*caption_EN*'));cap=p.read_text(encoding='utf-8');cap=cap.replace('**','');text+=r'\figpage{'+str(n)+'}{Fig'+str(n)+'}\n'+r'\figlegend{'+str(n)+'}{'+esc(cap)+'}\n'
(O/'figure_plates.tex').write_text(text,encoding='utf-8')
print('English tables and complete figure legends written.')
