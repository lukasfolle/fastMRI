function [samplingMask] = poissonUSMask(shape, acceleration)
    num_samples = 1000;
    while 1
        idx = poissonDisc(shape, 1, num_samples);
        image = zeros(shape(1), shape(2));
        for k=1:length(idx)
           image(round(idx(k, 1)), round(idx(k, 2))) = 1.0;
        end
        centerX = 50;
        centerY = 50;
        radius = 50;
        [X, Y] = meshgrid(0:100,0:100);
        circlePixels = (Y - centerY).^2 ...
            + (X - centerX).^2 <= radius.^2;
        circlePixels = imresize(circlePixels, [shape(1), shape(2)]);
        samplingMask = circlePixels .* image;
        center_y = round(size(samplingMask, 1) / 2);
        perc_y = round(size(samplingMask, 1)*0.05);
        center_x = round(size(samplingMask, 2) / 2);
        perc_x = round(size(samplingMask, 2)*0.05);
        samplingMask(center_y-perc_y:center_y+perc_y, center_x-perc_x:center_x+perc_x) = 1.0;
        R = 1 / (sum(samplingMask(:)) / numel(samplingMask));
        if abs(acceleration - R) < 0.1
            break
        elseif R > acceleration
            num_samples = num_samples - 100;
        elseif R < acceleration
            num_samples = num_samples + 100;
        end
        imshow(samplingMask)
        % disp(R)
        % disp(num_samples)
        % disp('')
    end
end