%% Necessary to use bart over wsl
setenv('TOOLBOX_PATH', '');

%% Load data
simulation = 0;
if ~simulation
    kspace = h5read('data/file1000000.h5', '/kspace');
    kspace = kspace.r + 1j * kspace.i;

    kspace = permute(kspace, [1, 2, 4, 3]);
    kspace = fftshift(kspace, 3);
    kspace = fft(kspace, [], 3);
    kspace = ifftshift(kspace, 3);
    % Manual undersampling
    kspace = kspace(...
                    round(size(kspace, 1)*1/4):round(size(kspace, 1)*3/4)-1,...
                    round(size(kspace, 2)*1/4):round(size(kspace, 2)*3/4)-1,...
                    round(size(kspace, 3) / 2)-4:round(size(kspace, 3) / 2)+4-1, :);
    slice_sel = round(size(kspace, 3) / 2 + 1);
else
    kspace = bart('phantom -3 -k -s 5');
    slice_sel = 64;
end

%% RSS for initial reco
knee_imgs = kspace;
% fft for each channel
for k=1:size(kspace, 4)
    knee_imgs(:, :, :, k) = fftshift(knee_imgs(:, :, :, k));
    knee_imgs(:, :, :, k) = ifftn(knee_imgs(:, :, :, k));
    knee_imgs(:, :, :, k) = ifftshift(knee_imgs(:, :, :, k));
end
knee_rss = bart('rss 8', knee_imgs);
knee_rss = (knee_rss - min(knee_rss(:))) / (max(knee_rss(:)) - min(knee_rss(:)));
figure;
imshow(squeeze(knee_rss(:, :, slice_sel)), [])

%% Undersampling
accel = 6.0;
num_samples = 3000;

while true
    us_mask = zeros(size(knee_rss, 3), size(knee_rss, 1));
    mu = [round(size(knee_rss, 1) / 2) round(size(knee_rss, 3) / 2)];
    Sigma = [size(knee_rss, 1) * 5 0;
             0 size(knee_rss, 3) * 10];
    X = int32(mvnrnd(mu, Sigma, num_samples));
    % Dense central mask sampling
    us_mask(round(size(us_mask, 1)*2/8):round(size(us_mask, 1)*6/8), round(size(us_mask, 2)*9/20):round(size(us_mask, 2)*11/20)) = 1;
    for xi=1:length(X)
        idx = X(xi,:);
        if idx(1) < 1
            continue
        elseif idx(2) < 1
           continue
        elseif idx(1) > size(knee_rss, 1)
            continue
        elseif idx(2) > size(knee_rss, 3)
            continue
        end
        us_mask(idx(2), idx(1)) = 1;
    end
    R = 1 / (sum(us_mask(:)) / numel(us_mask));
    if abs(R - accel) < 0.1
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
us_mask = reshape(us_mask, size(us_mask, 1), 1, size(us_mask, 2));
us_mask = repmat(us_mask, 1, size(kspace, 2), 1, size(kspace, 4));
kspace = kspace .* us_mask;

%% Reconstruct undersampled kspace
knee_imgs = kspace;
for k=1:size(kspace, 4)
    knee_imgs(:, :, :, k) = fftshift(knee_imgs(:, :, :, k));
    knee_imgs(:, :, :, k) = ifftn(kspace(:, :, :, k));
    knee_imgs(:, :, :, k) = ifftshift(knee_imgs(:, :, :, k));
end

knee_rss_us = bart('rss 8', knee_imgs);
knee_imgs = (knee_imgs - min(knee_imgs(:))) / (max(knee_imgs(:)) - min(knee_imgs(:)));

figure;
subplot(2,1,1);
imshow(squeeze(knee_rss(:, :, slice_sel)), [])
title('No US')
subplot(2,1,2);
imshow(squeeze(knee_rss_us(:, :, slice_sel)), [])
title('With US')

%% ESPIRiT calibration
sens_maps = bart('ecalib -k3 -c0. -m1', kspace);  % no -k for full size

figure;
subplot(2,1,1)
imshow3(abs(squeeze(sens_maps(:, :, slice_sel, :))), [],[3,5]);
title("Espirit Maps")
subplot(2,1,2)
imshow3(abs(squeeze(knee_imgs(:, :, slice_sel, :))), [],[3,5]);
title("Fully sampled images")
%%
%     0 readout
%     1 phase-encoding dimension 1
%     2 phase-encoding dimension 2
%     3 receive channels
%     4 ESPIRiT maps

kspace_l1 = bart('pics -l1 -r 0.01', kspace, sens_maps);
kspace_l1 = abs(kspace_l1);

%% Results
figure;
rss = squeeze(knee_rss(:, :, slice_sel));
rss = (rss - min(rss(:))) / (max(rss(:)) - min(rss(:)));
rss_us = squeeze(knee_rss_us(:, :, slice_sel));
rss_us = (rss_us - min(rss_us(:))) / (max(rss_us(:)) - min(rss_us(:)));
cs = squeeze(abs(kspace_l1(:, :, slice_sel)));
cs = (cs - min(cs(:))) / (max(cs(:)) - min(cs(:)));

imshow([rss, rss-rss; rss_us, rss_us-rss; cs, cs-rss], [])
title({'Original, US Reco, CS Reco', 'RECO | Error'})

%% Performance calculation
knee_rss = (knee_rss - min(knee_rss(:))) / (max(knee_rss(:)) - min(knee_rss(:)));
knee_rss_us = (knee_rss_us - min(knee_rss_us(:))) / (max(knee_rss_us(:)) - min(knee_rss_us(:)));
kspace_l1 = (kspace_l1 - min(kspace_l1(:))) / (max(kspace_l1(:)) - min(kspace_l1(:)));
disp("SSIM")
disp(ssim(knee_rss, knee_rss_us))
disp(ssim(knee_rss, kspace_l1))
disp("PSNR")
disp(psnr(knee_rss,knee_rss_us))
disp(psnr(knee_rss,kspace_l1))
disp("NRMSE")
disp(sqrt(mean((knee_rss(:) - knee_rss_us(:)).^2)))
disp(sqrt(mean((knee_rss(:) - kspace_l1(:)).^2)))
