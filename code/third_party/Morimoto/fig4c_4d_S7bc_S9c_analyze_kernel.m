%--------------------------------------------------------------------------
% This script generates Figures 4c, 4d, and 9c for the manuscript.
% It visualizes learned convolutional kernels from different training schemes
% ("original," "gamut rotation," and "texture") by:
% - Loading and displaying kernels for each model variant,
% - Saving kernel images and (for texture) all kernel components,
% - Analyzing and plotting the chromatic distribution of each kernel in CIELAB space,
% - Comparing the kernel color distributions to the daylight locus.
% All output figures are saved in the 'figs' directory.
%
% Author: TM, 2025
%--------------------------------------------------------------------------

clearvars;close all;clc % cleaning

disp('Generating figure 4c, 4d, S7bc and S9c...')

load(fullfile('data','fig_parameters'))

%%
for kernel_type = {'original','gamut_rotation','texture','lighting_elevation_shift_45deg','lighting_elevation_shift_90deg','lighting_rotation_45deg','object_rotation_90deg','object_rotation_45deg','lighting_rotation_90deg'}
%for kernel_type = {'texture'}
    % kernel directory
    switch kernel_type{1}
        case 'original'
            kernel_dir = fullfile(pwd,'data','networks','human_onelayer_kernelN1_trainedby3888imgs');
        case 'gamut_rotation'
            kernel_dir = fullfile(pwd,'data','networks','human_oneLayer_kernelN1_trainedby3888imgs_gamut_rotation');
        case 'texture'
            kernel_dir = fullfile(pwd,'data','networks','human_onelayer_kernelN2_trainedbytexturedimgs');
        otherwise
            kernel_dir = fullfile(pwd,'data','networks',['human_onelayer_kernelN1_',kernel_type{1}]);
    end
    
    % loading kernel
    temp = load(fullfile(kernel_dir,'kernel.mat'));
    I_kernel = temp.kernel;

    % mean and std used for normalization
    mean_rgb = [0.516186453743370, 0.4989957, 0.455484868501618];
    std_rgb = [0.231457708997510, 0.222412810219355, 0.235527730404435];
    fig = figure;
    if contains(kernel_type{1},'object') || contains(kernel_type{1},'lighting')
        kernel = (I_kernel-min(I_kernel(:)))/max(max(max(I_kernel-min(I_kernel(:)))));
        imagesc(kernel);axis square;axis off;
        print(fig, fullfile('figs',['figS7bc(',kernel_type{1},').pdf']), '-dpdf', '-fillpage')
    elseif ~strcmp(kernel_type{1},'texture')
        kernel = (I_kernel-min(I_kernel(:)))/max(max(max(I_kernel-min(I_kernel(:)))));
        imagesc(kernel);axis square;axis off;
        print(fig, fullfile('figs',['fig4cd(',kernel_type{1},').pdf']), '-dpdf', '-fillpage')
    elseif strcmp(kernel_type{1},'texture')
        for kernelN = 1:size(I_kernel,4)
            I_kernel_N = I_kernel(:,:,:,kernelN);
            kernel = (I_kernel_N-min(I_kernel_N(:)))/max(max(max(I_kernel_N-min(I_kernel_N(:)))));
            imagesc(kernel);axis square;axis off;
            print(fig, fullfile('figs',['figS9c(',kernel_type{1},'_kernelN',num2str(kernelN),').pdf']), '-dpdf', '-fillpage')
        end
    end

    %% Compute daylight color
    if strcmp(kernel_type{1},'original') || strcmp(kernel_type{1},'gamut_rotation')

        wp_d65 = [0.95047 1 1.08883];
        
        % a*b* for daylight locus
        ab_daylight =  [7.8542,35.2322;...
                        4.4413,25.8058;...
                        2.3027,17.7861;...
                        1.0069,10.9652;...
                        0.2632,5.1428;...
                        -0.1197,0.1471;...
                        -0.2674,-4.1634;...
                        -0.2624,-7.9201;...
                        -0.1588, -11.2005;...
                        0.0068,-14.0852;...
                        0.2103,-16.6379;...
                        0.4356,-18.9097;...
                        0.6718,-20.9422];
                    
        %% analyze color of kernels
        s = size(kernel);
        kernel_srgb_v = reshape(kernel,s(1)*s(2),s(3));
        kernel_linearsrgb = SRGBGammaUncorrect(kernel);
        kernel_xyz = SRGBPrimaryToXYZ(reshape(kernel_linearsrgb,s(1)*s(2),s(3))')';

        kernel_xyz = kernel_xyz/max(kernel_xyz(:));

        kernel_lab = XYZToLab(kernel_xyz',wp_d65')';

        fig = figure;

        scatter(kernel_lab(:,2),kernel_lab(:,3),20,power(kernel_srgb_v./sum(kernel_srgb_v,2)*1.5,1/2.2),'o','filled','MarkerEdgeColor',[.1 .1 .1],'LineWidth',0.1,'MarkerFaceAlpha',1);hold on;

        plot(ab_daylight(:,1),ab_daylight(:,2),'Color','k','LineWidth',0.5);hold on;

        fig.Units = 'centimeters';
        fig.Color = 'w';
        fig.InvertHardcopy = 'off';
        fig.PaperPosition   = [0,10,8.45,8.45];
        fig.Position = [10,10,figp.twocolumn/4*0.95,figp.twocolumn/4*0.95];

        ax = gca;
        range = 30;
        ax.XLim = [-range range];
        ax.XTick = [-range 0 range];
        ax.YLim = [-range range];
        ax.YTick = [-range 0 range];
        ax.XTickLabel = char(num2str(-range),'0',num2str(range));
        ax.YTickLabel = char(num2str(-range),'0',num2str(range));

        xlabel('a*','FontWeight', 'Bold','FontSize',figp.fontsize_axis);
        ylabel('b*','FontWeight', 'Bold','FontSize',figp.fontsize_axis);

        ax.FontName = figp.fontname;
        ax.FontSize = figp.fontsize;ax.Color = [.97 .97 .97];
        ax.XColor = 'k';ax.YColor = 'k';

        ax.LineWidth = 0.2;
        ax.Units = 'centimeters';
        ax.Position = [0.82 0.72 3.2 3.2];

        grid on;box off
        exportgraphics(fig,fullfile('figs',['fig4c(kernel_chromaticdistribution_',kernel_type{1},').pdf']),'ContentType','vector')
    end
end

close all

disp('Done.')
close all
