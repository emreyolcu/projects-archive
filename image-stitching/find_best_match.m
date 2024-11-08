function [index, H] = find_best_match(I, images)

N = length(images);

% Feature extraction
I_feat = extract_features({I});
r_feat = extract_features(images);

% Feature matching
feature_matches{N} = [];

di = I_feat{1}.d;
for j = 1:N
    dj = r_feat{j}.d;
    [matches, scores] = vl_ubcmatch(di, dj);
    feature_matches{j}.m = matches;
    feature_matches{j}.s = scores;
end

% Candidate matching
n_matches = 6;
n_matches = min(n_matches, N);
matchcounts = zeros(1,N);
for j = 1:N
    matchcounts(j) = size(feature_matches{j}.m, 2);
end
[~, I] = sort(matchcounts, 'descend');
candidate_matches = I(1:n_matches);

% Determining the homography for the best match
max_in = -1;
for j = candidate_matches
    matches_1 = feature_matches{j}.m(1,:);
    matches_2 = feature_matches{j}.m(2,:);
    
    mp_1 = I_feat{1}.f(:,matches_1);
    mp_2 = r_feat{j}.f(:,matches_2);
    
    [cand_H, n_in, ~] = ransac(mp_1, mp_2);
    
    if n_in > max_in
        max_in = n_in;
        index = j;
        H = cand_H;
    end
end

end

