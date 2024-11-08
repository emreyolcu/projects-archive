function [candidate_matches, matchcounts] = match_candidates(feature_matches)

n_matches = 6;
N = size(feature_matches, 1);
matchcounts = zeros(N);

% Number of matches between ith and jth images
for i = 1:N
    for j = (i+1):N
        matchcounts(i,j) = size(feature_matches{i,j}.m, 2);
        matchcounts(j,i) = matchcounts(i,j);
    end
end

n_matches = min(n_matches, N-1);
candidate_matches = zeros(N, n_matches);

% Best matching images for each image
for i = 1:N
    [~, I] = sort(matchcounts(i,:), 'descend');
    candidate_matches(i,:) = I(1:n_matches);
end

end

