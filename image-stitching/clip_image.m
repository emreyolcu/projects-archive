function image = clip_image(image)

s = sum(image,3);
image(~any(s,2), :, :) = [];
image(:, ~any(s,1), :) = [];

end

