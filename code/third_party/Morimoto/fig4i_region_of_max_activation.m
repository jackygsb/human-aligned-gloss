%% Fig 2e: 3×3 tiled grid with max-activation boxes (Hi×Hi only)
% Author: TM (2025-10-20)

clear; clc; close all;
rng(10); % reproducible sampling

%% ---------------------- PATHS & PARAMS ----------------------
csvPath   = fullfile('data','pred_validation','prediction_humanANDonelayer.csv');
kernel_file = fullfile('data','networks','human_onelayer_kernelN1_trainedby3888imgs','kernel');
img_dir   = fullfile('data','imgs','imgs3888_nobg_png');
out_dir   = fullfile(pwd, 'figs');
mkdir(out_dir);
out_hihi_crop = fullfile(out_dir, 'fig4i_tiled_region_max_activation.png');

% Panel parameters
n_cell_rows   = 3; % Human (High at top)
n_cell_cols   = 3; % Model (Low→High)
n_img_per_cell = 72;
img_tile_rows = 6;
img_tile_cols = 6;

gap_ratio_img = 0.03;  % small internal gap
crop_size     = 75;
midgray       = [127 127 127]/255;

%% ---------------------- LOAD RESPONSES ----------------------
T = readtable(csvPath);
ih = find(contains(lower(T.Properties.VariableNames),'humanresponse'), 1);
im = find(contains(lower(T.Properties.VariableNames),'onearea'), 1);
assert(~isempty(ih) && ~isempty(im), 'Missing columns in CSV.');

human_resp = T{:, ih};
model_resp = T{:, im};
assert(numel(human_resp) == numel(model_resp), 'Mismatched response lengths.');

img_ids = (1:numel(human_resp))';

%% ---------------------- LOAD KERNEL ----------------------
Sker = load(kernel_file);
K = double(Sker.kernel);
for c = 1:3, K(:,:,c) = rot90(K(:,:,c),2); end
[kh, kw, ~] = size(K);

%% ---------------------- BINNING & SELECTION ----------------------
edgesH = linspace(min(human_resp), max(human_resp), n_cell_rows+1);
edgesM = linspace(min(model_resp), max(model_resp), n_cell_cols+1);
human_bins = discretize(human_resp, edgesH);
model_bins = discretize(model_resp, edgesM);

iH = n_cell_rows; jM = n_cell_cols;
idx = find(human_bins==iH & model_bins==jM);
if numel(idx) > n_img_per_cell
    idx = idx(randperm(numel(idx), n_img_per_cell));
end

[~, img_list_crop] = collect_annot_and_crops(idx, img_ids, img_dir, K, kw, kh, crop_size, midgray, n_img_per_cell);
img_list_crop = fillMissingWithWhite(img_list_crop);

tile_crop = tileImagesWithGapAdaptive(img_list_crop, img_tile_rows, img_tile_cols, gap_ratio_img);
imwrite(tile_crop, out_hihi_crop);
fprintf('Saved Hi×Hi (crop): %s\n', out_hihi_crop);

%% ====================== LOCAL FUNCTIONS ======================

function [img_list_full, img_list_crop] = collect_annot_and_crops(idx, img_ids, img_dir, K, kw, kh, crop_size, midgray, n_img_per_cell)
    img_list_full = cell(1, n_img_per_cell);
    img_list_crop = cell(1, n_img_per_cell);
    for k = 1:n_img_per_cell
        if k > numel(idx), break; end
        imgN = img_ids(idx(k));
        img_path = fullfile(img_dir, sprintf('img%d.png', imgN));
        if ~isfile(img_path), continue; end
        I = im2double(imread(img_path));
        if size(I,3)==1, I = repmat(I,[1 1 3]); end
        R = conv_valid_rgb(I, K);
        [~, linMax] = max(R(:));
        [r_top, c_left] = ind2sub(size(R), linMax);
        I_annot = drawRectBorder(I, c_left, r_top, kw, kh, [1 0 1], 2);
        cx = c_left + floor(kw/2);
        cy = r_top  + floor(kh/2);
        I_crop = crop_center_with_pad(I_annot, cy, cx, crop_size, midgray);
        img_list_full{k} = im2uint8(I_annot);
        img_list_crop{k} = im2uint8(I_crop);
    end
end

function R = conv_valid_rgb(I, K)
    R = sum(conv2(I(:,:,1), K(:,:,1), 'valid') + ...
            conv2(I(:,:,2), K(:,:,2), 'valid') + ...
            conv2(I(:,:,3), K(:,:,3), 'valid'), 3);
end

function out_list = fillMissingWithWhite(img_list)
    sz = [];
    for n = 1:numel(img_list)
        if ~isempty(img_list{n}), sz = size(img_list{n}); break; end
    end
    if isempty(sz), sz = [128 128 3]; end
    for n = 1:numel(img_list)
        if isempty(img_list{n}), img_list{n} = uint8(255*ones(sz)); end
    end
    out_list = img_list;
end

function tiled = tileImagesWithGapAdaptive(img_list, nrows, ncols, gap_ratio)
    ref = [];
    for q = 1:numel(img_list)
        if ~isempty(img_list{q}), ref = size(img_list{q}); break; end
    end
    if isempty(ref), ref = [128 128 3]; end
    h = ref(1); w = ref(2);
    gap_px = max(1, round(gap_ratio * min(h, w)));
    img_list = fillMissingWithWhite(img_list);
    idx = 1;
    vgap = uint8(255*ones(h, gap_px, 3));
    row_blocks = cell(1, nrows);
    for i = 1:nrows
        row_imgs = cell(1, 2*ncols-1);
        for j = 1:ncols
            row_imgs{2*j-1} = img_list{idx}; idx = idx + 1;
            if j < ncols, row_imgs{2*j} = vgap; end
        end
        row_blocks{i} = cat(2, row_imgs{:});
    end
    hgap = uint8(255*ones(gap_px, size(row_blocks{1},2), 3));
    for i = 2:nrows
        row_blocks{i} = cat(1, hgap, row_blocks{i});
    end
    tiled = cat(1, row_blocks{:});
end

function Iout = drawRectBorder(I, x, y, w, h, color, t)
    % Safely draw a rectangular RGB border of given thickness
    Iout = I;
    [H,W,~] = size(Iout);
    x = max(1,min(W,x));
    y = max(1,min(H,y));
    w = max(1,min(W - x + 1, w));
    h = max(1,min(H - y + 1, h));

    for d = 0:t-1
        % Top horizontal line
        r = max(1, y + d);
        c1 = max(1, x + d); c2 = min(W, x + w - 1 - d);
        if r <= H && c1 <= c2
            Iout(r, c1:c2, :) = repmat(reshape(color, [1 1 3]), [1, c2 - c1 + 1, 1]);
        end

        % Bottom horizontal line
        r = min(H, y + h - 1 - d);
        if r >= 1 && c1 <= c2
            Iout(r, c1:c2, :) = repmat(reshape(color, [1 1 3]), [1, c2 - c1 + 1, 1]);
        end

        % Left vertical line
        c = max(1, x + d);
        r1 = max(1, y + d); r2 = min(H, y + h - 1 - d);
        if c <= W && r1 <= r2
            Iout(r1:r2, c, :) = repmat(reshape(color, [1 1 3]), [r2 - r1 + 1, 1, 1]);
        end

        % Right vertical line
        c = min(W, x + w - 1 - d);
        if c >= 1 && r1 <= r2
            Iout(r1:r2, c, :) = repmat(reshape(color, [1 1 3]), [r2 - r1 + 1, 1, 1]);
        end
    end
end


function J = crop_center_with_pad(I, cy, cx, cropN, fillRGB)
    [H,W,~] = size(I);
    half = floor(cropN/2);
    r1 = cy - half; r2 = r1 + cropN - 1;
    c1 = cx - half; c2 = c1 + cropN - 1;
    J = repmat(reshape(fillRGB,1,1,3), [cropN, cropN, 1]);
    rs = max(1,r1); re = min(H,r2);
    cs = max(1,c1); ce = min(W,c2);
    if rs <= re && cs <= ce
        drs = rs - r1 + 1; dcs = cs - c1 + 1;
        dre = drs + (re - rs); dce = dcs + (ce - cs);
        J(drs:dre, dcs:dce, :) = I(rs:re, cs:ce, :);
    end
end
