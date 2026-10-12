from pathlib import Path
import shutil,json
import numpy as np
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure1_english_v4'
P=O/'provenance_scripts';P.mkdir(exist_ok=True)
for name in ['analyze_behavioral_spaces_v1.py','build_paper_figures_v1.py','analyze_qwen3_full.py']:
    shutil.copy2(B/'scripts'/name,P/name)
checks=[]
for observer in ['Human','InternVL','Qwen3']:
    p=np.load(M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_shape_{observer}_profiles_by_view.npy').mean(0)
    d=((p[:,None,:]-p[None,:,:])**2).mean(2)
    saved=np.loadtxt(M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_shape_{observer}_rdm.csv',delimiter=',')
    assert np.allclose(d,saved)
    checks.append({'observer':observer,'profile_shape':list(p.shape),'n_unique_distances':len(d[np.triu_indices(36,1)]),'max_distance_reconstruction_error':float(np.abs(d-saved).max())})
(O/'RSA_figure_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(O/'PANEL_F_METHODS_EN.txt').write_text('''PANEL F: BEHAVIORAL RSA AND CROSS-TASK PREDICTION

The main flow illustrates physical/color-adjusted behavioral profiles. Each of the three datasets (aggregated human judgments, InternVL matches, Qwen3 matches) supplies values for the same 3,888 images. The examples in step 1 are the saved human means and model two-start mean matches, not the single-image ratings shown in panel e.

Nuisance adjustment is fitted separately to each dataset: intercept, categorical physical c, categorical chroma, and sine/cosine hue harmonics 1 and 2. Fit on two viewpoints and obtain residuals on the held-out viewpoint; rotate through all three. Average residuals across views into the 36 x 36 shape-by-illumination table. The example table and two shape curves are actual human data.

Shape profiles contain 36 lighting responses. Illumination profiles contain 36 shape responses (transpose the table). Profile distance is the mean squared coordinate difference. The three shape RDMs are independently estimated, with matching row/column condition order. Their display is normalized by mean off-diagonal distance and capped at 3, as in the prior version. The distance strips show all 630 upper-triangle entries transformed to ranks for display. Spearman RSA compares corresponding entries, not figure pixel colors.

Inference: 4,999 two-sided permutations of condition identities (simultaneous row and column permutation in one RDM), 2,000 condition-node bootstraps (excluding self-pairs induced by repeated sampled nodes), BH correction over 18 full-profile comparisons: 3 response spaces x 2 condition factors x 3 dataset pairs. Response spaces are raw, physical/color-adjusted, and interaction-only. Interaction-only additionally subtracts row and column means and adds the grand mean (double centering). This workflow is independent-profile RSA; it does not fit a model coordinate system to human coordinates. Low-rank SVD analyses elsewhere in the project are not depicted in this panel.

Cross-task consistency is a separate analysis within each model. Continuous rating-to-match and match-to-rating mappings are isotonic regressions, fitted and tested in 12 grouped folds based on shape or illumination. Each fold groups three of the 36 condition identities; 11 folds train, one tests. Rotate the held-out fold and compare out-of-fold predictions with recorded targets using R2 and Spearman correlation. This is distinct from the held-out-view nuisance adjustment used for RSA. The original analysis scripts define the retained cohort.

For pairwise judgments, the sign of a single-score or reference-match difference predicts which stimulus should look glossier; predictions are checked against actual responses in both display orders. Human-physical conflict is a predefined pair subset in which the human and physical orderings disagree. It is not an extra dimension in the RDM and is not an arrow between rating scales. Joint logistic prediction analyses in the full analysis are not depicted in this basic methods panel.

Reproduction: the standalone plotting script and bundled assets recreate the figure without model inference. The original analysis scripts in provenance_scripts document the source procedures; rerunning complete analyses requires the original project inputs and environment. RSA_figure_validation.json confirms that the plotted profiles reproduce each saved shape RDM numerically.
''',encoding='utf-8')
s=(B/'scripts/package_figure1_english_v3.py').read_text('utf-8')
s=s.replace('figure1_english_v3','figure1_english_v4').replace('Detailed_v3','Detailed_v4').replace('180 x 548','180 x 768')
exec(compile(s,str(__file__),'exec'))
