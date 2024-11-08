function [f, d] = custom_sift(image)

[f, d] = vl_sift(image);
f = f(1:2,:);

end

