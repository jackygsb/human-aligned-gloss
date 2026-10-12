%--------------------------------------------------------------------------
% This script generates Figure 3 for the manuscript.
% It compares computational model performance in gloss perception by:
% - Loading human and model response data,
% - Computing and comparing correlation coefficients for one-layer, three-layer,
%   and ResNet18-based networks (with/without additional training data),
% - Including image statistics and specular metrics models,
% - Visualizing the relationship between correlation-to-ground-truth and
%   correlation-to-human for all approaches in a single comprehensive plot.
% The script saves the resulting figure in the 'figs' directory.
%
% Author: TM, 2025
%--------------------------------------------------------------------------

% Source code to generate figures
clearvars; close all; % cleaning

disp('Generating figure 3...')

%% Load Data
load(fullfile('data','onlineData'))
load(fullfile('data','imageStats_corrCoeff'))
load(fullfile('data','imgStats_multiRegression_corrCoeff'))
load(fullfile('data','fig_parameters'))

%% One-layer and three-la models: load correlation coefficient
kernelN_list.onelayer = [1 2 4 9];
kernelN_list.threelayer = [9 16 32 64];

filedir.onelayer = fullfile('data','networks','onelayer_models');
filedir.threelayer = fullfile('data','networks','threelayer_models');

cvtypeList = {'shape','lighting'};
traininglabelList = {'human','groundtruth'};

for model = {'onelayer','threelayer'}
    for kernelN_idx = 1:length(kernelN_list.(model{1}))
        kernelN = kernelN_list.(model{1})(kernelN_idx);
        for traininglabel = traininglabelList
            for cvtype = cvtypeList % validation type (shape-based or lighting-based cross validation)
                for ii = 1:12

                    % Determine label ordering for correlation
                    if strcmp(traininglabel{1}, 'human')
                        % correlation to human response (human), or correlation to
                        % physical ground-truth (gt)
                        corrlabelList = {'human', 'gt'};
                    else
                        corrlabelList = {'gt', 'human'};
                    end

                    for corr_label = corrlabelList
                        fname = fullfile(filedir.(model{1}), [traininglabel{1},'_kernelN',num2str(kernelN), ...
                            '_',cvtype{1}, num2str(ii),'/corrs_',corr_label{1},'.csv']);

                        temp = readmatrix(fname);

                        % Pick the max correlation row according to label type
                        % and use the same id to get the correlation for the
                        % other objective
                        if (strcmp(traininglabel{1},'human') && strcmp(corr_label{1},'human')) || ...
                           (strcmp(traininglabel{1},'groundtruth') && strcmp(corr_label{1},'gt'))
                            [~,maxid] = max(temp(:,1));
                        end

                        corr_all.(model{1}).(traininglabel{1}).(corr_label{1})(ii,strcmp(cvtype{1},cvtypeList),kernelN_idx) = temp(maxid,1);
                    end
                end
            end
        end
    end
end

% Reshape and average correlation results for plotting
for traininglabel = {'human','groundtruth'}
    for corr_label = {'human','gt'}
        oneLayer_corr.(traininglabel{1}).(corr_label{1}) = mean(reshape(corr_all.onelayer.(traininglabel{1}).(corr_label{1}),24,length(kernelN_list.onelayer)));
        oneLayer_corr_SD.(traininglabel{1}).(corr_label{1}) = std(reshape(corr_all.onelayer.(traininglabel{1}).(corr_label{1}),24,length(kernelN_list.onelayer)));

        threeLayers_corr.(traininglabel{1}).(corr_label{1}) = mean(reshape(corr_all.threelayer.(traininglabel{1}).(corr_label{1}),24,length(kernelN_list.threelayer)));
        threeLayers_corr_SD.(traininglabel{1}).(corr_label{1}) = std(reshape(corr_all.threelayer.(traininglabel{1}).(corr_label{1}),24,length(kernelN_list.threelayer)));
    end
end

%% Visualization & Figure generation
cnt = 0;lw = 0.2;symbolsize = 50;

% set color codes
c_human = [90 152 152]/255;
c_human_mean = [160 212 212]/255;
c_oneLayer_human = [78 85 246]/255;
c_oneLayer_gt = [184 0 127]/255;
c_specularStr = [151 217 92]/255;
c_ResNet18 = c_oneLayer_gt;
c_additionalImage = [248 92 1]/255;

basedir_additional = fullfile('data','networks','additional_trainingimgs');

% load ResNet18 trained on ground-truth (additional training)

% additional 500,000 imgs
temp = readmatrix([basedir_additional,'/groundtruth_ResNet18_kernelN64_addtrainingN_500000/corrs_gt.csv']);
[ResNet18_additional500000.gt,maxid] = max(temp(:,1));
temp = readmatrix(fullfile(basedir_additional,'groundtruth_ResNet18_kernelN64_addtrainingN_500000/corrs_human.csv'));
ResNet18_additional500000.human = temp(maxid,1);

% additional 100,000 imgs
temp = readmatrix([basedir_additional,'/groundtruth_ResNet18_kernelN64_addtrainingN_100000/corrs_gt.csv']);
[ResNet18_additional100000.gt,maxid] = max(temp(:,1));
temp = readmatrix([basedir_additional,'/groundtruth_ResNet18_kernelN64_addtrainingN_100000/corrs_human.csv']);
ResNet18_additional100000.human = temp(maxid,1);

% additional 10,000 imgs
temp = readmatrix([basedir_additional,'/groundtruth_ResNet18_kernelN64_addtrainingN_10000/corrs_human.csv']);
[ResNet18_additional10000.human,maxid] = max(temp(:,1)); 
temp = readmatrix([basedir_additional,'/groundtruth_ResNet18_kernelN64_addtrainingN_10000/corrs_gt.csv']);
ResNet18_additional10000.gt = temp(maxid,1);

% load three-layer models trained on ground-truth (additional training)

% additional 500,000 imgs
temp = readmatrix([basedir_additional,'/groundtruth_threelayer_kernelN64_addtrainingN_500000/corrs_gt.csv']);
[twoArea_additional500000.gt,maxid] = max(temp(:,1)); 
temp = readmatrix([basedir_additional,'/groundtruth_threelayer_kernelN64_addtrainingN_500000/corrs_human.csv']);
twoArea_additional500000.human =temp(maxid,1); 

% additional 100,000 imgs
temp = readmatrix([basedir_additional,'/groundtruth_threelayer_kernelN64_addtrainingN_100000/corrs_gt.csv']);
[twoArea_additional100000.gt,maxid] = max(temp(:,1)); 
temp = readmatrix([basedir_additional,'/groundtruth_threelayer_kernelN64_addtrainingN_100000/corrs_human.csv']);
twoArea_additional100000.human = temp(maxid,1); 

% additional 10,000 imgs
temp = readmatrix([basedir_additional,'/groundtruth_threelayer_kernelN64_addtrainingN_10000/corrs_gt.csv']);
[twoArea_additional10000.gt,maxid] = max(temp(:,1)); 
temp = readmatrix([basedir_additional,'/groundtruth_threelayer_kernelN64_addtrainingN_10000/corrs_human.csv']);
twoArea_additional10000.human = temp(maxid,1); 

% load ResNet18 trained on ground-truth (no additional training)
basedir_ResNet18 = fullfile('data','networks');
temp = readmatrix([basedir_ResNet18,'/groundtruth_ResNet18_kernelN64/corrs_gt.csv']);
[ResNet18.gt,maxid] = max(temp(:,1)); 
temp = readmatrix([basedir_ResNet18,'/groundtruth_ResNet18_kernelN64/corrs_human.csv']);
ResNet18.human = temp(maxid,1); 

%% compute correlation to groundtruth and correlation to other participants for each participant
groupN_summary = zeros(54,1);
for N = 1:length(data)
    onlineData(:,:,N) = data(N).response_Pellacini_c;
    groupN_summary(N) = data(N).groupN;
end

for groupN = 1:54
    idx = find(groupN_summary == groupN);
    for N = 1:length(idx)
        response_group.(['group',num2str(groupN)])(:,N) = mean(data(idx(N)).response_Pellacini_c,2);
        if N == 1
            gt_group.(['group',num2str(groupN)])(:,1) = gt(idx(N)).Pellacini_c(1:84);
        end
    end
end

for groupN = 1:54
    for subjectN = 1:size(response_group.(['group',num2str(groupN)]),2)
        cnt = cnt + 1;
        obs1 = response_group.(['group',num2str(groupN)])(:,subjectN);
        obs_rest = mean(response_group.(['group',num2str(groupN)])(:,[1:subjectN-1,subjectN+1:end]),2);
        gt_temp = gt_group.(['group',num2str(groupN)]);

        human_human_corr(cnt) = corr(obs1,obs_rest);
        human_gt_corr(cnt) = corr(obs1,gt_temp);
    end
end

%% Generate figure
fig = figure;
ax = gca;

% fill the figure panel with different colors and draw a diagonal line
fill([0,0,1],[0,1,1],c_oneLayer_human,'FaceAlpha',0.05,'EdgeColor','none');hold on;
fill([0,1,1],[0,0,1],c_oneLayer_gt,'FaceAlpha',0.05,'EdgeColor','none');hold on;
line([0,100],[0,100],'Color','k','LineWidth',0.5)

%%%%%% plot human participant %%%%%%
scatter(human_gt_corr,human_human_corr,20,c_human,'o','filled','MarkerEdgeColor','none','LineWidth',lw,'MarkerFaceAlpha',0.4);hold on;
scatter(mean(human_gt_corr),mean(human_human_corr),50,c_human_mean,'x','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',1,'LineWidth',1.5);hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% plot one-Layer rgb (human label) %%%%%%
for kernelN = 1:size(oneLayer_corr.human.human,2)
    scatter(mean(oneLayer_corr.human.gt(:,kernelN)),mean(oneLayer_corr.human.human(:,kernelN)),40,c_oneLayer_human,'^','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% plot one-Layer rgb (gt label) %%%%%%
for kernelN = 1:size(oneLayer_corr.groundtruth.human,2)
    scatter(mean(oneLayer_corr.groundtruth.gt(:,kernelN)),mean(oneLayer_corr.groundtruth.human(:,kernelN)),40,c_oneLayer_gt,'^','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% plot two-Layers rgb (human label) %%%%%%
for kernelN = 1:size(threeLayers_corr.human.human,2)
    scatter(mean(threeLayers_corr.human.gt(:,kernelN)),mean(threeLayers_corr.human.human(:,kernelN)),50,c_oneLayer_human,'s','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
    scatter(mean(threeLayers_corr.groundtruth.gt(:,kernelN)),mean(threeLayers_corr.groundtruth.human(:,kernelN)),50,c_oneLayer_gt,'s','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% plot ResNet18 (no additional) %%%%%%
scatter(ResNet18.gt,ResNet18.human,40,c_ResNet18,'>','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% plot networks trained on additional images %%%%%%
scatter(ResNet18_additional100000.gt,ResNet18_additional100000.human,40,c_additionalImage,'>','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
scatter(ResNet18_additional500000.gt,ResNet18_additional500000.human,40,c_additionalImage,'>','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;

hold on;

%scatter(twoArea_additional10000.gt,twoArea_additional10000.human,50,c_oneLayer_human,'s','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
scatter(twoArea_additional100000.gt,twoArea_additional100000.human,50,c_additionalImage,'s','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
scatter(twoArea_additional500000.gt,twoArea_additional500000.human,50,c_additionalImage,'s','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% plot image statistics models %%%%%%
for N = 1:length(corrCoeff_imgStats.label)
    x = corrCoeff_imgStats.gtvsmodel(N,:);
    y = corrCoeff_imgStats.humanvsmodel(N,:);
    scatter(mean(x),mean(y),40,[.8 .8 .8],'d','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',1,'LineWidth',lw);hold on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% plot image statistics Multiple regression models %%%%%%
x = corrCoeff_imgStats_multiRegression.gtvsmodel;
y = corrCoeff_imgStats_multiRegression.humanvsmodel;
scatter(mean(x),mean(y),60,[.8 .8 .8],'p','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',1,'LineWidth',lw);hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% plot image statistics Multiple regression models %%%%%%
load(fullfile('data','specularMetrics_multiRegression_corrCoeff.mat'))
x = corrCoeff_specularMetrics_multiRegression.gtvsmodel;
y = corrCoeff_specularMetrics_multiRegression.humanvsmodel;
scatter(mean(x),mean(y),60,c_specularStr,'p','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% plot specular reflection models %%%%%%
for modellabel = {'contrast','coverage','sharpness'}
    x = abs(corrCoeff_specularMetrics.gt.(modellabel{1}));
    y = abs(corrCoeff_specularMetrics.human.(modellabel{1}));

    scatter(mean(x),mean(y),40,c_specularStr,'v','filled','MarkerEdgeColor',[0 0 0],'MarkerFaceAlpha',0.5,'LineWidth',lw);hold on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xlim([0 1]);ylim([0 1]);axis square
xlabel('Correlation to ground-truth','FontWeight', 'Bold');ylabel('Correlation to human','FontWeight', 'Bold');

fig.Units           = 'centimeters';
fig.Position = [10,10,figp.twocolumn/2,figp.twocolumn/2];
fig.Color           = 'w';
fig.InvertHardcopy  = 'off';

xticks(0:0.25:1)
yticks(0:0.25:1)

ax.XTickLabel = {'0.00','0.25','0.50','0.75','1.00'};
ax.YTickLabel = {'0.00','0.25','0.50','0.75','1.00'};

ax.FontName = 'Arial';
ax.Color = ones(1,3);
ax.FontSize = figp.fontsize;
ax.XColor = 'k';ax.YColor = 'k';

ax.LineWidth = 0.5;
ax.Units = 'centimeters';
ax.Position = [0.95 0.85 7.6 7.6];
ticklengthcm(ax,0.0)
grid minor
box off
exportgraphics(fig,fullfile('figs','fig3(model_comparison).pdf'),'ContentType','vector')

disp('Done.')
close all



%% ============================================================
%  Stats tests (24 paired correlations; correlation to human responses)
%  (1) Multi-reg (all luminance stats) vs mean luminance only
%  (2) Multi-reg (contrast+coverage+sharpness) vs sub-band contrast only
%  (3) One-layer (1 kernel) vs Three-layer (64 kernels)
%% ============================================================

alpha = 0.05; % 95% CI

% --- Helper: find mean-luminance model row in corrCoeff_imgStats.label ---
labels = corrCoeff_imgStats.label;
if iscell(labels)
    labels_lower = lower(string(labels));
else
    labels_lower = lower(string(labels(:)));
end

% Prefer label containing both "mean" and "lum"
idx_meanlum = find(contains(labels_lower, "mean"), 1, 'first');
if isempty(idx_meanlum)
    idx_meanlum = find(contains(labels_lower, "mean luminance"), 1, 'first');
end
if isempty(idx_meanlum)
    idx_meanlum = find(contains(labels_lower, "luminance"), 1, 'first');
end
if isempty(idx_meanlum)
    error('Could not identify the mean-luminance model in corrCoeff_imgStats.label. Please check label strings.');
end

% ============================================================
% (1) Luminance: multi-reg vs mean luminance-only
% ============================================================
% Positive diff => multi-reg > mean-luminance
y_lum_multi = corrCoeff_imgStats_multiRegression.humanvsmodel(:);
y_meanlum   = corrCoeff_imgStats.humanvsmodel(idx_meanlum, :).';

if numel(y_lum_multi) ~= numel(y_meanlum)
    error('Size mismatch: multi-reg luminance (%d) vs mean-luminance (%d).', numel(y_lum_multi), numel(y_meanlum));
end

y_lum_multi = abs(y_lum_multi);
y_meanlum   = abs(y_meanlum);

diff_lum = y_lum_multi - y_meanlum;
[~, p_lum, ~, stats_lum] = ttest(diff_lum, 0, 'Alpha', alpha, 'Tail', 'both');

n_lum  = numel(diff_lum);
df_lum = stats_lum.df;
t_lum  = stats_lum.tstat;

dz_lum = mean(diff_lum) / std(diff_lum);
[deltaL_lum, deltaU_lum] = local_nct_delta_ci(t_lum, df_lum, alpha);
dzL_lum = deltaL_lum / sqrt(n_lum);
dzU_lum = deltaU_lum / sqrt(n_lum);

% ============================================================
% (2) Specular: multi-reg vs contrast-only
% ============================================================
% Positive diff => multi-reg > contrast-only
y_spec_multi = corrCoeff_specularMetrics_multiRegression.humanvsmodel(:);
y_contrast   = corrCoeff_specularMetrics.human.contrast(:);

if numel(y_spec_multi) ~= numel(y_contrast)
    error('Size mismatch: multi-reg specular (%d) vs contrast-only (%d).', numel(y_spec_multi), numel(y_contrast));
end

y_spec_multi = abs(y_spec_multi);
y_contrast   = abs(y_contrast);

diff_spec = y_spec_multi - y_contrast;
[~, p_spec, ~, stats_spec] = ttest(diff_spec, 0, 'Alpha', alpha, 'Tail', 'both');

n_spec  = numel(diff_spec);
df_spec = stats_spec.df;
t_spec  = stats_spec.tstat;

dz_spec = mean(diff_spec) / std(diff_spec);
[deltaL_spec, deltaU_spec] = local_nct_delta_ci(t_spec, df_spec, alpha);
dzL_spec = deltaL_spec / sqrt(n_spec);
dzU_spec = deltaU_spec / sqrt(n_spec);

% ============================================================
% (3) One-layer (1 kernel) vs Three-layer (64 kernels)
% ============================================================
% Positive diff => three-layer(64) > one-layer(1)

% Find kernel indices
idx_1kernel  = find(kernelN_list.onelayer  == 1,  1, 'first');   % 1
idx_64kernel = find(kernelN_list.threelayer == 64, 1, 'first');  % 4

if isempty(idx_1kernel) || isempty(idx_64kernel)
    error('Could not find kernel indices: one-layer(1) or three-layer(64).');
end

% --- IMPORTANT: use RAW 24 values (not means) ---
% corr_all.onelayer.human.human has size: (12 x 2 x 4)
% reshape -> (24 x 4) where 24 = 12(shape) + 12(lighting)
r_one_all   = reshape(corr_all.onelayer.human.human,   24, length(kernelN_list.onelayer));
r_three_all = reshape(corr_all.threelayer.human.human, 24, length(kernelN_list.threelayer));

r_one1    = r_one_all(:, idx_1kernel);       % 24x1
r_three64 = r_three_all(:, idx_64kernel);    % 24x1

diff_net = r_three64 - r_one1;

% Paired t-test + Cohen's dz + CI(dz)
[~, p_net, ~, stats_net] = ttest(diff_net, 0, 'Alpha', alpha, 'Tail', 'both');

n_net  = numel(diff_net);
df_net = stats_net.df;
t_net  = stats_net.tstat;

dz_net = mean(diff_net) / std(diff_net);
[deltaL_net, deltaU_net] = local_nct_delta_ci(t_net, df_net, alpha);
dzL_net = deltaL_net / sqrt(n_net);
dzU_net = deltaU_net / sqrt(n_net);

% ============================================================
% Print full stats
% ============================================================
fprintf('\n=== Requested paired comparisons (n = %d patterns) ===\n', n_lum);

fprintf('\n(1) All luminance stats (multi-reg) vs mean luminance-only:\n');
if p_lum < 1e-3
    fprintf('t(%d) = %.2f, p < 0.001, Cohen''s dz = %.3f, 95%% CI [%.3f, %.3f]\n', ...
        df_lum, t_lum, dz_lum, dzL_lum, dzU_lum);
else
    fprintf('t(%d) = %.2f, p = %.3f, Cohen''s dz = %.3f, 95%% CI [%.3f, %.3f]\n', ...
        df_lum, t_lum, p_lum, dz_lum, dzL_lum, dzU_lum);
end

fprintf('\n(2) All specular stats (multi-reg: contrast+coverage+sharpness) vs contrast-only:\n');
if p_spec < 1e-3
    fprintf('t(%d) = %.2f, p < 0.001, Cohen''s dz = %.3f, 95%% CI [%.3f, %.3f]\n', ...
        df_spec, t_spec, dz_spec, dzL_spec, dzU_spec);
else
    fprintf('t(%d) = %.2f, p = %.3f, Cohen''s dz = %.3f, 95%% CI [%.3f, %.3f]\n', ...
        df_spec, t_spec, p_spec, dz_spec, dzL_spec, dzU_spec);
end

fprintf('\n(3) One-layer (1 kernel) vs Three-layer (64 kernels):\n');
if p_net < 1e-3
    fprintf('t(%d) = %.2f, p < 0.001, Cohen''s dz = %.3f, 95%% CI [%.3f, %.3f]\n', ...
        df_net, t_net, dz_net, dzL_net, dzU_net);
else
    fprintf('t(%d) = %.2f, p = %.3f, Cohen''s dz = %.3f, 95%% CI [%.3f, %.3f]\n', ...
        df_net, t_net, p_net, dz_net, dzL_net, dzU_net);
end



function [deltaL, deltaU] = local_nct_delta_ci(tObs, df, alpha)
% 100*(1-alpha)% CI for noncentrality parameter delta of noncentral t,
% inverted from observed tObs (two-sided).
%
% Solve:
%   nctcdf(tObs; df, deltaL) = 1 - alpha/2
%   nctcdf(tObs; df, deltaU) = alpha/2

    if exist('nctcdf','file') ~= 2
        error('nctcdf not found. Requires Statistics and Machine Learning Toolbox.');
    end

    targetL = 1 - alpha/2;  % 0.975
    targetU = alpha/2;      % 0.025

    deltaL = local_solve_delta_robust(tObs, df, targetL);
    deltaU = local_solve_delta_robust(tObs, df, targetU);

    if deltaL > deltaU
        tmp = deltaL; deltaL = deltaU; deltaU = tmp;
    end
end

function delta = local_solve_delta_robust(tObs, df, target)
% Robustly solve nctcdf(tObs; df, delta) = target for delta.
% Avoids non-finite endpoints and uses a safe fallback if bracketing fails.

    fun = @(d) nctcdf(tObs, df, d) - target;

    % Start near a reasonable center; delta is typically on the order of tObs
    center = tObs;

    % Candidate bracket half-widths (grow gradually)
    widths = [0.5 1 2 5 10 20 50 100 200 500 1000 2000];

    lo = NaN; hi = NaN;
    flo = NaN; fhi = NaN;

    % Try to find a finite bracket with sign change
    for w = widths
        lo_c = center - w;
        hi_c = center + w;

        flo_c = fun(lo_c);
        fhi_c = fun(hi_c);

        if isfinite(flo_c) && isreal(flo_c) && isfinite(fhi_c) && isreal(fhi_c)
            if sign(flo_c) ~= sign(fhi_c)
                lo = lo_c; hi = hi_c;
                flo = flo_c; fhi = fhi_c;
                break;
            end
        end
    end

    % If still not bracketed, try asymmetric expansion (sometimes needed)
    if isnan(lo)
        for w = widths
            lo_c = center - w;
            hi_c = center + 5*w;

            flo_c = fun(lo_c);
            fhi_c = fun(hi_c);

            if isfinite(flo_c) && isreal(flo_c) && isfinite(fhi_c) && isreal(fhi_c)
                if sign(flo_c) ~= sign(fhi_c)
                    lo = lo_c; hi = hi_c;
                    flo = flo_c; fhi = fhi_c;
                    break;
                end
            end
        end
    end

    % If we found a valid bracket, use fzero safely
    if ~isnan(lo)
        try
            delta = fzero(fun, [lo, hi]);
            return;
        catch
            % fall through to fallback
        end
    end

    % Fallback: bounded minimization of squared error on a safe delta range
    % (keeps code from crashing even in rare numeric corner cases)
    bound = 2000; % wide enough for typical cases, avoids nctcdf blowups
    obj = @(d) (fun(d)).^2;

    try
        delta = fminbnd(obj, center - bound, center + bound);
    catch
        % last resort: return center
        delta = center;
    end
end