from pathlib import Path
import re,json,shutil
B=Path(__file__).resolve().parents[1];O=B/'output/NMI_English_Manuscript_v10';V=B/'output/NMI_English_Manuscript_v9'
body=(O/'body.tex').read_text(encoding='utf-8')
replacements={
'Reflectance models and perceptual parameterizations specify surface properties independently of the judgments elicited by rendered images':'Rendered stimuli provide known reflectance parameters alongside independently measured appearance judgments',
'Across both models, the three tasks yielded 108,732 responses, including 15,552 complete matches derived from multiple comparisons':'Across both models, acquisition yielded 108,732 individual responses; reference comparisons produced 15,552 complete searches',
'We therefore tested agreement among images with the same nominal physical gloss.':'To isolate correspondence beyond the generating parameter, we compared images with the same nominal physical gloss.',
'The evidence for alignment is consequently strongest for relative appearance and within-level covariation, both of which are also expressed by direct choices.':'The correspondence is therefore strongest at the level of relative appearance, supported by within-level covariation and independently measured pairwise choices.',
'The shared structure is therefore a systematic scene dependence of appearance, particularly for illumination, rather than evidence for an unrestricted common latent space.':'The common geometry was therefore dominated by additive shape and illumination dependencies, particularly pronounced for illumination.',
'Direction construction used neither human test-image labels nor the supervised readout coefficients.':'A unit dose corresponded to the fitted activation change across the full reference range. Direction construction used neither human test-image labels nor the supervised readout coefficients.',
'Thus, a direction defined independently from a reference sequence exerted a signed, dose-dependent influence on gloss decisions.':'A direction derived from reference responses without human labels thus exerted a signed, dose-dependent influence on gloss decisions.',
'a binary comparison':'a pairwise choice',
'Applied to gloss, this framework reveals human-aligned appearance judgments and a functional visual contribution within general-purpose multimodal models.':'These results establish human-aligned, scene-dependent gloss judgments as an accessible property of frozen multimodal models, and identify an internal visual component through which their judgments can be modulated.'}
for a,b in replacements.items():body=body.replace(a,b)
body=re.sub(r'(?:A unit dose corresponded to the fitted activation change across the full reference range\.\s*)+', 'A unit dose corresponded to the fitted activation change across the full reference range. ', body)
body=body.replace('we evaluate correspondence in ordering and response structure rather than treating human and model matching values as an identical psychophysical scale (Methods).', 'we evaluate correspondence in ordering and response structure rather than treating human and model matching values as an identical psychophysical scale. The behavioral and internal analyses are summarized in Fig.~1c,d (Methods).')
(O/'body.tex').write_text(body,encoding='utf-8')
clean=(O/'audit/user_clean_nonred.tex').read_text(encoding='utf-8')
methods='\\section{Methods}'+clean.split('\\section{Methods}',1)[1].split('\\section*{Data availability}',1)[0]
methods=methods.replace('Supplementary Sections','Supplementary Notes').replace('Supplementary Section','Supplementary Note')
methods=methods.replace('categorical physical gloss, chroma','categorical physical-gloss and chroma terms').replace('recorded physical/color terms','categorical physical-gloss and chroma terms and the first two sine/cosine hue harmonics')
methods=methods.replace('Supplementary Notes~S2 and S3 and Fig.~1c','Supplementary Notes~S2 and S3 and Supplementary Fig.~S11')
methods=methods.replace('(Fig.~1e)','(Supplementary Fig.~S12)')
methods=methods.replace('Controls comprised a whole-image-luminance covariance direction and three seeded random directions projected onto the subspace orthogonal to $\\mathbf d$.','Controls comprised a whole-image-luminance covariance direction, estimated from original test images outside the 72-image intervention set, and three seeded random directions projected onto the subspace orthogonal to $\\mathbf d$.')
methods=methods.replace('Its five settings were the shared baseline','The test image was always presented first and the reference second; this auxiliary assay was not order-counterbalanced. Its five settings were the shared baseline')
methods=methods.replace('Supplementary Notes~S1--S19, Figs.~S1--S10 and Tables~S1--S9','Supplementary Notes~S1--S20, Supplementary Figs.~S1--S15 and Supplementary Tables~S1--S9')
methods=methods.replace('12 shape-group or illumination-group folds.','12 shape-group or illumination-group folds. Representational prediction and interpretation follow the distinction between readout performance and representational mechanism\\cite{diedrichsen2017,kriegeskorte2016}.')
(O/'methods.tex').write_text(methods,encoding='utf-8')
abstract='''Surface gloss depends on reflectance, shape and illumination, allowing human judgments to diverge from physical parameters. We ask whether frozen vision--language models preserve these perceptual dependencies. Using 3,888 rendered images and published human matches, we measure ratings, pairwise choices and reference matches in InternVL3.5-4B and Qwen3-VL-4B-Instruct. Without adaptation to the target behavioral labels, both models track human variation within physical-gloss levels, preferentially follow human ordering in conflict pairs, and share shape- and illumination-dependent response organization. Ratings and matches predict each other in held-out scene conditions and anticipate pairwise choices. Supervised readouts of frozen internal states predict human judgments beyond physical and color covariates. Separately, a visual direction derived from reference responses without human labels bidirectionally shifts gloss choices more than equal-norm controls. These results establish structured human alignment in continuous surface appearance and provide predictive and functional evidence about the internal information accompanying these judgments.'''
preamble=r'''\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,booktabs,graphicx,array,longtable}
\usepackage[numbers,sort&compress]{natbib}
\providecommand{\bibcommenthead}{}
\usepackage[margin=24mm]{geometry}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{1em}
\setlength{\parskip}{3pt}
\emergencystretch=3em
\raggedbottom
\input{legends.tex}
\newcommand{\paperfigure}[3]{\clearpage\begin{center}\includegraphics[width=\textwidth,height=#3\textheight,keepaspectratio]{figures/#1.pdf}\end{center}\vspace{-2mm}\noindent{\small #2\par}\clearpage}
\begin{document}
\title{Human-aligned gloss judgments in frozen vision--language models}
\author{Author names and affiliations to be supplied}
\date{}
\maketitle
\begin{abstract}
'''+abstract+r'''
\end{abstract}
'''
# Insert the six figures after the corresponding Results sections in the reading copy.
intro,rest=body.split('\\section{Results}',1)
results,discussion=rest.split('\\section{Discussion}',1)
parts=re.split(r'(?=\\subsection\{)',results)
assembled=intro+'\\section{Results}\n'
labels=['one','two','three','four','five','six'];heights=['.69','.69','.69','.69','.69','.69']
for i,part in enumerate([p for p in parts if p.strip()]):
 assembled+=part
 name='Fig1_overview' if i==0 else f'Fig{i+1}'
 assembled+=f'\\paperfigure{{{name}}}{{\\legend{labels[i]}}}{{{heights[i]}}}\n'
assembled+='\\section{Discussion}\n'+discussion
end=r'''
\section*{Data availability}
Published human judgments and original test stimuli are available from the source study\cite{morimoto2026}. Derived source tables, analysis manifests, reconstructed references and model responses are retained with this study. A public repository identifier and asset redistribution terms are to be inserted before submission.
\section*{Code availability}
Analysis, acquisition and figure-generation code are retained in the study archive. A permanent public repository identifier is to be inserted before submission. Third-party models and original stimulus assets remain subject to their respective licenses.
\section*{Acknowledgements}
To be completed by the authors.
\section*{Funding}
To be completed by the authors.
\section*{Author contributions}
To be completed by the authors.
\section*{Competing interests}
Declaration to be supplied by the authors.
\section*{Additional information}
Supplementary Information accompanies this manuscript. Correspondence details are to be supplied by the authors.
\bibliographystyle{sn-nature}
\bibliography{references}
\end{document}
'''
(O/'main.tex').write_text(preamble+assembled+'\n\\clearpage\n'+methods+end,encoding='utf-8')
# Supplement remains complete; amend precise methods and remove drafting-history language.
s=(V/'supplement.tex').read_text(encoding='utf-8')
s=s.replace('Human-aligned material perception and its internal representations in frozen vision--language models','Human-aligned gloss judgments in frozen vision--language models')
s=s.replace('Supplementary Sections','Supplementary Notes').replace('Supplementary Section','Supplementary Note').replace('Section S4','Supplementary Note S4')
s=s.replace('categorical physical gloss, chroma','categorical physical-gloss and chroma terms')
s=s.replace('The revised main Fig. 5h additionally quantifies','Main Fig. 5h quantifies')
s=s.replace('The English redraw reads the headerless 36-by-36 matrix CSVs explicitly without a header, retaining all conditions.','All 36 conditions are retained in the matrices and inferential analyses.')
s=s.replace('Candidate, luminance-associated and three equal-norm random orthogonal directions are tested','The luminance-associated covariance direction is estimated from original test images outside the 72-image intervention set. Candidate, luminance-associated and three equal-norm random orthogonal directions are tested')
s=s.replace('Each match starts at level 27, allows at most seven steps and uses constrained choice among three answer tokens.','Each match starts at level 27, allows at most seven steps and uses constrained choice among three answer tokens. The test image is always presented first and the reference second; unlike the primary fixed-reference assay, this auxiliary assay is not order-counterbalanced.')
s=s.replace('Residual analysis first fits the physical/color nuisance baseline on training data','Residual analysis first fits a nuisance baseline using categorical physical-gloss and chroma terms and first- and second-order sine/cosine hue terms on training data')
s=s.replace('Supplementary Figs. S1--S10 and Tables S1--S9','Supplementary Figs. S1--S15 and Tables S1--S9')
s=s.replace('Sections S5--S12','Notes S5--S12').replace('Sections S13--S19','Notes S13--S19')
s=s.replace('The base models remain frozen throughout;','Supplementary Note S20 documents figure-selection rules; Supplementary Figs. S11--S15 preserve the complete illustrated acquisition and analysis protocols. The base models remain frozen throughout;')
extra=(O/'supplement_extensions.tex').read_text(encoding='utf-8')
s=s.replace('\\clearpage\\bibliography{references}',extra+'\n\\clearpage\\begingroup\\setlength{\\bibsep}{4pt}\\bibliography{references}\\endgroup')
(O/'supplement.tex').write_text(s,encoding='utf-8')
def wc(t):
 t=re.sub(r'\\(?:cite|ref|label)\{[^}]*\}','',t);t=re.sub(r'\\[A-Za-z]+',' ',t)
 return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",t))
report={'main_text_approx_words':wc(body),'abstract_approx_words':wc(abstract),'methods_approx_words':wc(methods)}
(O/'audit/word_counts.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
