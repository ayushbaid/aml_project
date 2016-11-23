file_name = 'dict_high';

load(strcat(file_name, '.mat'));

patch_edge_length = 7;

img = viewColorPatches(U, patch_edge_length);

imwrite(imresize(img, 10, 'nearest'), strcat(file_name, '.png'));
