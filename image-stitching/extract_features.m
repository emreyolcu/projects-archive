function features = extract_features(images)

N = length(images);
features{N,1} = [];

for i = 1:N
    [f, d] = custom_sift(single(rgb2gray(images{i})));
    features{i}.f = f;  % Descriptor locations
    features{i}.d = d;  % Feature descriptors
end

end

