function [matchpoints_1, matchpoints_2] = determine_matchpoints(feature_matches, features, i, j)

% Indices of matches
matches_1 = feature_matches{i,j}.m(1,:);
matches_2 = feature_matches{i,j}.m(2,:);

% Image coordinates of matches
matchpoints_1 = features{i}.f(:,matches_1);
matchpoints_2 = features{j}.f(:,matches_2);

end

