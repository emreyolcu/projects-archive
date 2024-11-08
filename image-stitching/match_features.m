function feature_matches = match_features(features)

N = length(features);
feature_matches{N,N} = [];

for i = 1:N
    di = features{i}.d;
    for j = 1:N
        dj = features{j}.d;
        [matches, scores] = vl_ubcmatch(di, dj);
        feature_matches{i,j}.m = matches;
        feature_matches{i,j}.s = scores;
    end
end

end

