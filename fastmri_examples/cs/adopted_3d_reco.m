%% Necessary to use bart over wsl
setenv('TOOLBOX_PATH', '');

%% Load data
simulation = 1;
if ~simulation 
    kspace = h5read('data/file1000000.h5', '/kspace');
    kspace = kspace.r + 1j * kspace.i;

    kspace = permute(kspace, [1, 2, 4, 3]);
    kspace = fft(kspace, [], 3);
    slice_sel = 2;
else
    kspace = bart('phantom -3 -k -s 5');
    slice_sel = 64;
end

%% RSS for initial reco
% Root-of-sum-of-squares image
knee_imgs = kspace;
% fft for each channel
for k=1:size(kspace, 4)
    knee_imgs(:, :, :, k) = fftshift(knee_imgs(:, :, :, k));
    knee_imgs(:, :, :, k) = ifftn(knee_imgs(:, :, :, k));
    knee_imgs(:, :, :, k) = ifftshift(knee_imgs(:, :, :, k));
end
knee_imgs = (knee_imgs - min(knee_imgs(:))) / (max(knee_imgs(:)) - min(knee_imgs(:)));
knee_rss = squeeze(abs(sum(knee_imgs, 4) .^ 2));
imagesc(squeeze(knee_rss(:, :, slice_sel)))

%% Undersampling
accel = 4.0;
num_samples = 4800;
while true
    us_mask = zeros(size(knee_rss, 3), size(knee_rss, 2));
    mu = [round(size(knee_rss, 2) / 2) round(size(knee_rss, 3) / 2)];
    Sigma = [size(knee_rss, 2) * 10 0;
             0 size(knee_rss, 3) * 10];
    X = int32(mvnrnd(mu, Sigma, num_samples));
    for xi=1:length(X)
        idx = X(xi,:);
        if idx(1) < 1
            continue
        elseif idx(2) < 1
           continue
        elseif idx(1) > size(knee_rss, 2)
            continue
        elseif idx(2) > size(knee_rss, 3)
            continue
        end
        us_mask(idx(2), idx(1)) = 1;
    end
    R = 1 / (sum(us_mask(:)) / numel(us_mask));
    if abs(R - accel) < 0.01
        break
    elseif R > accel
        num_samples = num_samples + 50;
    elseif R < accel
        num_samples = num_samples - 50;
    end
end
us_mask = us_mask';

figure;
imshow(squeeze(us_mask(:, :)'))
us_mask = reshape(us_mask, 1, size(us_mask, 1), size(us_mask, 2));
us_mask = repmat(us_mask, size(kspace, 1), 1, 1, size(kspace, 4));
frac = 8;
acs = kspace(:, round(size(kspace, 2) * (frac/2-1)/frac):round(size(kspace, 2) * (frac/2+1)/frac), round(size(kspace, 3) * (frac/2-1)/frac):round(size(kspace, 3) * (frac/2+1)/frac), :);
pad = zeros(size(kspace));
pad(:, round(size(kspace, 2) * (frac/2-1)/frac):round(size(kspace, 2) * (frac/2+1)/frac), round(size(kspace, 3) * (frac/2-1)/frac):round(size(kspace, 3) * (frac/2+1)/frac), :) = acs;
acs = pad;
kspace = kspace .* us_mask;

%% Reconstruct undersampled kspace
knee_imgs = kspace;
for k=1:size(kspace, 4)
    knee_imgs(:, :, :, k) = fftshift(knee_imgs(:, :, :, k));
    knee_imgs(:, :, :, k) = ifftn(kspace(:, :, :, k));
    knee_imgs(:, :, :, k) = ifftshift(knee_imgs(:, :, :, k));
end

knee_rss_us = squeeze(abs(sum(knee_imgs, 4) .^ 2));
knee_imgs = (knee_imgs - min(knee_imgs(:))) / (max(knee_imgs(:)) - min(knee_imgs(:)));

figure;
subplot(2,1,1);
imagesc(squeeze(knee_rss(:, :, slice_sel)))
title('No US')
subplot(2,1,2);
imagesc(squeeze(knee_rss_us(:, :, slice_sel)))
title('With US')

%% ESPIRiT calibration
sens_maps = bart('ecalib -m 1', acs);
% sens_maps = bart('ecalib -m1 -k 4', kspace); does not work, too few info?

%%
%     0 readout
%     1 phase-encoding dimension 1
%     2 phase-encoding dimension 2
%     3 receive channels
%     4 ESPIRiT maps

kspace_l1 = bart('pics -l1', kspace, sens_maps);

%%% ACS free CS method????????????????????????????????????



%% Results
figure;
subplot(1,3,1);
imagesc(squeeze(knee_rss(:, :, slice_sel)))
title('No US')
subplot(1,3,2);
imagesc(squeeze(knee_rss_us(:, :, slice_sel)))
title('With US')
subplot(1,3,3)
imagesc(squeeze(abs(kspace_l1(:, :, slice_sel))))
title('With CS Reco')
