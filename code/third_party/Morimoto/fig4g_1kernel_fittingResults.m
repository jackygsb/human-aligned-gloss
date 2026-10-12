%--------------------------------------------------------------------------
% This script generates Figure 4e by fitting two Gaussian ridge functions
% and a 2D Gaussian to a trained 15x15 kernel (from a human one-layer model).
% The script outputs the fitted model components, residuals, and saves images
% of the model components for visualization.
%
% Author: TM, 2025
%--------------------------------------------------------------------------
clearvars; close all;

disp('Generating figure 4g...')

%% Parameters
show_colorbar = false;  % Set to false to hide all colorbars

% Load display and plotting parameters (adjust this file path as needed)
load(fullfile('data','fig_parameters'))

%% 1. Load and preprocess the 15×15 kernel
kernel_dir = fullfile('data', 'networks', 'human_onelayer_kernelN1_trainedby3888imgs');

% Load the trained RGB kernel and normalize to [0, 1]
temp = load(fullfile(kernel_dir, 'kernel'));
I_rgb = temp.kernel;
I = (I_rgb - min(I_rgb(:))) ./ (max(I_rgb(:)) - min(I_rgb(:)));

% Convert to grayscale if input is RGB
if ndims(I) == 3
    I = rgb2gray(I);
end

% Generate coordinate grid centered at zero
[nRows, nCols] = size(I);
[xGrid, yGrid] = meshgrid(-nCols/2:(nCols/2-1), -nRows/2:(nRows/2-1));
X = cat(3, xGrid, yGrid);

% Normalize image again after grayscale conversion
I = (I - min(I(:))) / max(max((I - min(I(:)))));

% Demean data for fitting
Data = I - mean(I(:));

%% 2. Fit two Gaussian ridge functions (with free orientation)
% Each ridge: Gaussian drop-off perpendicular to a line
% Parameterization: [amplitude, sigma, x0, y0, phi, offset]
fRidge = @(B, X) B(6) + B(1) * exp( ...
    -(((X(:, :, 1) - B(3)) * sin(B(5)) - (X(:, :, 2) - B(4)) * cos(B(5))).^2) ...
    / (2 * B(2)^2) ...
);

% Parameter bounds for the ridge
lb_r = [0.0, 0.5, -nCols/2, -nRows/2, -pi/2, -10];
ub_r = [2,   5,    nCols/2,  nRows/2,  pi/2,  10];

ridgePhis = [-1.1533,  1.0455]; % Initial angles for the two ridges
nR = numel(ridgePhis);

resCurr = Data;
modelsR = cell(1, nR);
paramsR = zeros(nR, 6);
rssR    = zeros(1, nR);

for k = 1:nR
    % Initial parameters: [amplitude, sigma, x0, y0, phi, offset]
    A0_r = [0.5, 2, 0, 0, ridgePhis(k), 0];
    [B, rssR(k), model_k, resCurr] = fitFilterKernel(fRidge, A0_r, lb_r, ub_r, X, resCurr);
    paramsR(k, :) = B;
    modelsR{k} = model_k;
end

%% 3. Fit a single 2D Gaussian to the remaining residual
% Parameterization: [amplitude, x0, sigma_x, y0, sigma_y, offset]
fGauss = @(A, X) A(6) + A(1) * exp( ...
    -(((X(:, :, 1) - A(2)).^2) / (2 * A(3)^2) + ((X(:, :, 2) - A(4)).^2) / (2 * A(5)^2)) ...
);

A0_g = [1, 0, 5, 0, 5, 0.1];     % Initial guess
lb_g = [0, -nCols/2, 0.5, -nRows/2, 0.5, -Inf];
ub_g = [Inf, nCols/2, nCols, nRows/2, nRows, Inf];

[A_g, rss_g, model_g, resAfterGauss] = fitFilterKernel(fGauss, A0_g, lb_g, ub_g, X, resCurr);

%% 4. Combine model components and plot results
fullModel = model_g;
for k = 1:nR
    fullModel = fullModel + modelsR{k};
end
finalRes = Data - fullModel;

% Prepare images and color ranges for plotting
img.ridge1   = modelsR{1};  crange.ridge1   = [0, 0.15];
img.ridge2   = modelsR{2};  crange.ridge2   = [0, 0.15];
img.gauss    = model_g;     crange.gauss    = [0, 0.5];
img.original = Data;        crange.original = [-0.2, 0.8];
img.residual = abs(finalRes); crange.residual = [-0.2, 0.5];

figure('Position', [100, 100, 1000, 600]);
colormap gray;

for type = {'ridge1', 'ridge2', 'gauss', 'original', 'residual'}
    imagesc(img.(type{1}));
    axis square; axis off; colormap gray;
    caxis(crange.(type{1}));
    if show_colorbar
        colorbar;
    end
    saveas(gca, fullfile('figs', ['fig4g_', type{1}, '.png']));
end

disp('Done.')
close all

%% 5. Helper function: Generic LSQ wrapper for kernel fitting
function [A, resnorm, model, residual] = fitFilterKernel(fh, A0, lb, ub, X, Data)
    % Fits a parametric model (fh) to image data using lsqcurvefit.
    % Returns best-fit parameters, residual norm, model, and residual image.
    D   = Data(:);
    fun = @(p, ~) reshape(fh(p, X), [], 1);
    opts = optimoptions('lsqcurvefit', ...
        'MaxIterations', 100000, ...
        'OptimalityTolerance', 1e-9, ...
        'Display', 'off'); % Suppress output
    [A, resnorm] = lsqcurvefit(fun, A0, [], D, lb, ub, opts);
    model    = fh(A, X);
    residual = Data - model;
end
