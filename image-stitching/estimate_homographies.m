function [homographies, adj, consmatch] = estimate_homographies(candidate_matches, feature_matches, features)

N = size(candidate_matches, 1);
homographies{N,N} = [];
adj = zeros(N);
consmatch = zeros(N);

for i = 1:N
    homographies{i,i} = affine2d();  % Identity transform
end

for i = 1:N
    for j = candidate_matches(i,:)
        [mp_1, mp_2] = determine_matchpoints(feature_matches, features, i, j);
        
        % Estimate a homography
        [H, n_in, n_out] = ransac(mp_1, mp_2);
        homographies{i,j} = H;
        
        % Verify the match
        if verify_match(n_in, n_out)
            adj(i,j) = 1;
            consmatch(i,j) = n_in;
        end
    end
end

end

