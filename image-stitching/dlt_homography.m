function H = dlt_homography(u, u_)

N = size(u, 2);
A = [];

for i = 1:N
    Ui = [u(:,i)' 1];
    ui_ = u_(1,i);
    vi_ = u_(2,i);
    z3 = zeros(1,3);
    A = vertcat(A, [-Ui z3 Ui*ui_]);
    A = vertcat(A, [z3 -Ui Ui*vi_]);
end

[~, ~, V] = svd(A);
H = V(:,end);
H = reshape(H./H(9), [3 3]);

H = projective2d(H);

end

