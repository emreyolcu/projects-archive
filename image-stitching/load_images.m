function images = load_images(path, scale)

files = dir(path);
files(1:2) = [];  % Exclude `.` and `..`
N = length(files);

images{N,1} = [];

for i = 1:N
    images{i} = imresize(imread(fullfile(path, files(i).name)), scale);
end

end

