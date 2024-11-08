function [H, n_in, n_out] = ransac(points_1, points_2)

iter = 1000;
n_sample = 4;
epsilon = 5;

N = size(points_1, 2);
max_inliers = -1;
n_sample = min(n_sample, N);

for i = 1:iter
    r = randperm(N);
    
    samples = r(1:n_sample);  % Fit a homography to 4 randomly sampled points
    remaining = r((n_sample+1):end);  % Use the remaining points to determine inliers
    
    candidate_H = dlt_homography(points_1(:,samples), points_2(:,samples));
    fw = transformPointsForward(candidate_H, points_1(:,remaining)')';
    
    dists = sqrt(sum((points_2(:,remaining) - fw).^2, 1));
    n_inliers = sum(dists < epsilon);
    
    if n_inliers > max_inliers
        H = candidate_H;
        max_inliers = n_inliers;
        n_out = sum(dists >= epsilon);
    end
end

n_in = max_inliers+n_sample;

end

