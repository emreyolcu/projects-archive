function result = stitch_two(images, H)

[x, y] = outputLimits(H, [1 size(images{1},2)], [1 size(images{1},1)]);

x_min = min([1 x]);
x_max = max([size(images{1},2) x]);

y_min = min([1 y]);
y_max = max([size(images{1},1) y]);

w = round(x_max-x_min);
h = round(y_max-y_min);

result = zeros([h w 3], 'like', images{1});
blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

x_lim = [x_min x_max];
y_lim = [y_min y_max];
view = imref2d([h w], x_lim, y_lim);

warped_1 = imwarp(images{1}, H, 'OutputView', view);
warped_2 = imwarp(images{2}, affine2d(), 'OutputView', view);

result = step(blender, result, warped_1, warped_1(:,:,1));
result = step(blender, result, warped_2, warped_2(:,:,1));

result = clip_image(result);

end

