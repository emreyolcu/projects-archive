function display_panoramas(cc, save)

N = length(cc);
for i = 1:N
    I = clip_image(create_panorama(cc{i}));
    if save
        imwrite(I, ['output/pa' num2str(i) '.jpg']);
    end
    figure;
    imshow(I);
end

end

