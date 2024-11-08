function panorama = create_panorama(images)

pair{1} = images{1};
images = images(2:end);

while ~isempty(images)
    [index, H] = find_best_match(pair{1}, images);
    pair{2} = images{index};
    images(index) = [];
    pair{1} = stitch_two(pair, H);
end

panorama = pair{1};

end

