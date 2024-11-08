clear;
clc;

path = 'data/mountain';
scale = 1;
save = false;

% -------------------------------------------------------------------------
run('vlfeat-0.9.20/toolbox/vl_setup.m');

disp('Loading images');
images = load_images(path, scale);

disp('Extracting SIFT features');
features = extract_features(images);

disp('Determining feature matches');
feature_matches = match_features(features);

disp('Determining candidate image matches');
[candidate_matches, matchcounts] = match_candidates(feature_matches);

disp('Estimating homographies and verifying matches');
[homographies, adj, consmatch] = estimate_homographies(candidate_matches, feature_matches, features);

disp('Finding connected components');
cc = connected_comps(adj, images);

disp('Stitching images and displaying panoramas');
display_panoramas(cc, save);

disp('Done');

