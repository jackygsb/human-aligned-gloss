%--------------------------------------------------------------------------
% Kernel Fitting Script for One-Layer Network (24 Kernels)
%
% This script fits each of the 24 learned 15×15 kernels (from the one-layer
% network) with a parametric model composed of one 2D Gaussian and up to 
% two Gaussian ridge components. Each kernel is normalized, and kernel-specific
% constraints are provided for robust fitting. Ridge orientation is converted 
% to degrees (NaN for unfitted ridges), and all fitting parameters are saved 
% to a CSV file for downstream analysis and visualization. Optional figure 
% outputs allow visual inspection of the fits.
%
% Outputs:
%   - Fitted parameter table: 'data/gauss_ridge_parameters_kernel24.csv'
%   - Optional visualization of fitting for each kernel
%
% Author: TM, 2025
%--------------------------------------------------------------------------

clearvars; close all; clc % clean up

disp('Fitting 2-D Gauss and Gaussian ridge to 24 kernels... ')

rng(0)
show_colorbar = true;  % Set to false to hide colorbars
visualize_fitting = false;

% loading fig paremters
load(fullfile('data','fig_parameters'))

%% Load 15×15 kernel data
% Each kernel is normalized to [0,1] for fitting and visualization
kernel_24 = zeros(15,15,3,24);
cnt = 0;
for type = {'lighting','shape'}
    for N = 1:12
        cnt = cnt + 1;
        kernel_path = fullfile('data','networks','onelayer_models',...
            ['human_kernelN1_',type{1},num2str(N)],'kernel.mat');
        load(kernel_path);  % Loads variable 'kernel'
        kernel_24(:,:,:,cnt) = (kernel - min(kernel(:))) / (max(kernel(:)) - min(kernel(:)));
    end
end

%% Define Per-Kernel Fitting Parameter Constraints
% If the optimizer fails to find a second ridge, only one ridge is used
kernel24_nR = [1,1,1,1,2,2,2,1,1,1,2,1,2,2,1,2,2,1,1,1,2,1,1,1];

% --- FITTING PARAMETERS ---
% All kernel-specific fit parameter constraints are set here
fitParams = repmat(struct( ...
    'Phi_min', -1, 'Phi_max', 1, ...
    'gauss_s_min', 0.5, 'gauss_s_max', 5, ...
    'gauss_x0_min', -7, 'gauss_x0_max', 7, ...
    'gauss_y0_min', -7, 'gauss_y0_max', 7, ...
    'ridge_x0_min', -7, 'ridge_x0_max', 7, ...
    'ridge_y0_min', -7, 'ridge_y0_max', 7 ...
), 24, 1);

% Manually override constraints for specific kernels because some kernel needs different bounds to find a fit
fitParams(1).Phi_max = -0.5; fitParams(1).gauss_s_max = 2; fitParams(1).gauss_x0_max = -2;
fitParams(5).Phi_max = 0.5;
fitParams(8).ridge_x0_min = 5; fitParams(8).gauss_s_max = 2; fitParams(8).gauss_x0_min = 2;
fitParams(10).Phi_max = -0.5;
fitParams(16).gauss_s_max = 2; fitParams(16).gauss_x0_max = -2;
fitParams(19).Phi_min = 0.5; fitParams(19).gauss_s_max = 2; fitParams(19).ridge_x0_max = 6;
fitParams(20).Phi_max = 0.8;

% Initialize storage
paramsR_record = zeros(24,12);  % Ridge parameters (up to 2 ridges × 6 params)
A_g_record     = zeros(24,6);   % 2D Gaussian parameters

%% Fit Each Kernel
for N = 1:24
    % Get fitting params for this kernel
    I = kernel_24(:,:,:,N);
    if ndims(I)==3, I = rgb2gray(I); end

    % Normalize image
    I = (I - min(I(:)))/max(max((I - min(I(:)))));
    Data = I-mean(I(:));    % Data to fit
    
    % Create coordinate grid centered at image center
    [nRows,nCols] = size(I);
    [xGrid,yGrid] = meshgrid( -nCols/2:(nCols/2-1), -nRows/2:(nRows/2-1) );
    X = cat(3, xGrid, yGrid);

    %% 3) Fit two Gaussian ridges
    fRidge = @(B,X) B(6) + B(1)*exp( ...
        -(((X(:,:,1)-B(3))*sin(B(5)) - (X(:,:,2)-B(4))*cos(B(5))).^2)/(2*B(2)^2) ...
    );

    % Prepare ridge fitting
    p = fitParams(N);
    nR = kernel24_nR(N);
    
    lb_r = [0.0, p.gauss_s_min, p.ridge_x0_min, p.ridge_y0_min,  p.Phi_min,  -10];
    ub_r = [2,   p.gauss_s_max, p.ridge_x0_max, p.ridge_y0_max,  p.Phi_max,   10];

    % Initial orientations in radians
    ridgePhis = [+0.8848,  -0.8848]; 
    
    % Fit Gaussian ridge(s)
    resCurr = Data;
    modelsR = cell(1,nR);
    paramsR = zeros(nR,6);
    rssR    = zeros(1,nR);
    for k = 1:nR
        A0_r = [0.5, 2, 0, 0, ridgePhis(k), 0];
        [ B, rssR(k), model_k, resCurr ] = ...
            fitFilterKernel(fRidge, A0_r, lb_r, ub_r, X, resCurr);
        paramsR(k,:) = B;
        modelsR{k}   = model_k;
    end
    
    % Store ridge parameters
    if nR == 1
        paramsR_record(N,:) = [paramsR,zeros(1,6)];
    else
        paramsR_record(N,:) = [paramsR(1,:),paramsR(2,:)];
    end
        
    %% Fit 2D Gaussian (on residual after ridges)
    fGauss = @(A,X) A(6) + A(1)*exp( ...
        -(((X(:,:,1)-A(2)).^2)/(2*A(3)^2) + ((X(:,:,2)-A(4)).^2)/(2*A(5)^2)) ...
    );
    A0_g = [0.5, 0, 5, 0, 5, 0.1];
    lb_g = [0, p.gauss_x0_min, p.gauss_s_min, p.gauss_y0_min, p.gauss_s_min, -Inf];
    ub_g = [1, p.gauss_x0_max, p.gauss_s_max, p.gauss_y0_max, p.gauss_s_max, Inf];

    [A_g, rss_g, model_g, resAfterGauss] = ...
        fitFilterKernel(fGauss, A0_g, lb_g, ub_g, X, resCurr);

    A_g_record(N,:) = A_g;

    %% Visualization
    if visualize_fitting
        fullModel = model_g;
        for k = 1:nR
            fullModel = fullModel + modelsR{k};
        end
        finalRes = Data - fullModel;

        f = figure('Position', [100 100 1000 600]);
        f.Name = ['image',num2str(N)];
        colormap gray;

        subplot(3,4,1), imagesc(Data), axis image off, title('Data');
        if show_colorbar, colorbar; end

        for k = 1:nR
            phiDeg = 180 - mod(paramsR(k,5) * 180/pi, 180);
            subplot(3,4,1 + k)
            imagesc(modelsR{k}), axis image off
            title(sprintf('Ridge %d (φ=%.1f°)', k, phiDeg))
            if show_colorbar, colorbar; end
        end

        subplot(3,4,1 + nR + 1), imagesc(model_g), axis image off, title('Gaussian');
        if show_colorbar, colorbar; end

        res_after_ridge = Data;
        for k = 1:nR
            res_after_ridge = res_after_ridge - modelsR{k};
        end
        subplot(3,4,1 + nR + 2), imagesc(res_after_ridge), ...
            axis image off, title('Residual after Ridges');
        if show_colorbar, colorbar; end

        subplot(3,4,1 + nR + 3), imagesc(fullModel), axis image off, title('Combined Model');
        if show_colorbar, colorbar; end

        subplot(3,4,1 + nR + 4), imagesc(finalRes), axis image off, title('Final Residual');
        if show_colorbar, colorbar; end
    end
end

if visualize_fitting
    pause(2)
    close all
end

%% Convert radians to degrees for ridge orientation
% Replace zeros (no fit) with NaN for clarity
for n = [5, 11]
    col = paramsR_record(:, n);
    deg = 180 - mod(col * 180 / pi, 180);
    
    % Set result to NaN where original data was zero
    deg(col == 0) = NaN;
    
    paramsR_record(:, n) = deg;
end

%% Save Fitted Parameters to CSV
gaussHeaders = {'amp','x0','sx','y0','sy','offset'};
ridgeHeaders = {'amp1','sigma1','x01','y01','phi1_deg','offset1', ...
                'amp2','sigma2','x02','y02','phi2_deg','offset2'};
allHeaders = [gaussHeaders, ridgeHeaders];

T = array2table([A_g_record,paramsR_record], 'VariableNames', allHeaders);
writetable(T, fullfile('data','gauss_ridge_parameters_kernel24.csv'));
display(['Done. Fitting parameters saved... ', fullfile('data','gauss_ridge_parameters_kernel24.csv')])

%% Generic Least-Squares Fitting Function
function [A,resnorm,model,residual] = fitFilterKernel(fh,A0,lb,ub,X,Data)
    D   = Data(:);
    fun = @(p,~) reshape(fh(p,X), [], 1);
    opts = optimoptions('lsqcurvefit','MaxIterations',100000,'OptimalityTolerance',1.0e-9,'Display', 'off');
    [A,resnorm] = lsqcurvefit(fun, A0, [], D, lb, ub, opts);
    model    = fh(A,X);
    residual = Data - model;
end
