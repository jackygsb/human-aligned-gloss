%--------------------------------------------------------------------------
% This script generates Figure 4b for the manuscript.
% It visualizes the learned convolutional kernels from the single-kernel
% one-layer model by:
% - Loading and normalizing each kernel image (for both lighting and shape based validation ),
% - Saving individual visualizations for all 24 kernels,
% - Arranging all kernels into a 3 × 8 tiled composite image.
% The script outputs all images to the 'figs/onelayer_kernel' directory and
% saves the final composite figure in the 'figs' directory.
%
% Author: TM, 2025
%--------------------------------------------------------------------------

clearvars;close all; % cleaning

disp('Generating figure 4b...')
%% Load, visualize and save all kernel images from the single-kernel model
cnt = 0;

% create a folder
if ~exist(fullfile(pwd,'figs', 'onelayer_kernel'),'file')
    mkdir(fullfile(pwd,'figs', 'onelayer_kernel'))
end
    
for type = {'lighting', 'shape'}
    for N = 1:12
        kernel_dir = fullfile('data', 'networks', 'onelayer_models', ['human_kernelN1_', type{1}, num2str(N)]);
        temp = load(fullfile(kernel_dir, 'kernel'));
        kernel = temp.kernel;

        cnt = cnt + 1;

        % Normalize kernel values for visualization
        k_min = min(kernel(:));
        k_max = max(kernel(:));
        kernel_norm = (kernel - k_min) / (k_max - k_min);

        % Plot kernel image
        imagesc(kernel_norm);
        axis square off
        set(gca, 'Units', 'normalized');

        % Adjust plot position to remove white margins
        tightInset = get(gca, 'TightInset');
        newPos = [tightInset(1), tightInset(2), 1 - tightInset(1) - tightInset(3), 1 - tightInset(2) - tightInset(4)];
        set(gca, 'Position', newPos);

        % Save figure
        saveas(gca, fullfile('figs', 'onelayer_kernel', ...
            ['kernel', num2str(N), '_', type{1}, '.png']));
    end
end

%% Create a tiled image of all kernels (fig4b)
cnt = 0;
I_cropped = [];

for type = {'lighting', 'shape'}
    for N = 1:12
        cnt = cnt + 1;

        % Load and preprocess saved kernel image
        imgPath = fullfile('figs', 'onelayer_kernel', ...
            ['kernel', num2str(N), '_', type{1}, '.png']);
        I = double(imread(imgPath)) / 255;

        % Crop to remove axes and resize to standard dimension (range values were manually selected)
        thresh = 0.99;
        mask = any(I < thresh, 3);
        [row, col] = find(mask);
        I_cropped(:,:,:,cnt) = padarray(imresize(I(min(row):max(row), min(col):max(col), :), [256, 256]), [40, 20], 1, 'both'); % pad with white
    end
end

% Arrange into a 3 (rows) x 8 (columns) tiled image
s = size(I_cropped);
I_out = reshape(permute(reshape(I_cropped, [s(1), s(2), s(3), 3, 8]), [1, 4, 2, 5, 3]), [s(1)*3, s(2)*8, s(3)]);

% Display and save
figure;
imshow(I_out);
imwrite(I_out, fullfile('figs', 'fig4b(onelayer_24kernels).png'));

disp('Done.')
close all
