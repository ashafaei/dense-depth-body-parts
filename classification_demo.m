addpath(genpath('./models/'));
addpath(genpath('./ubc3vtoolkit'));

% Update this if you already have a matconvnet on your computer.
% It requires compilation.
run ./matconvnet/matlab/vl_setupnn.m

%% Load and prepare the image.

depth_image = importdata('sample.png');

% Find the boundaries of the person in the depth image.
margin_1 = sum(depth_image.alpha, 1);
margin_2 = sum(depth_image.alpha, 2);
top = find(margin_2, 1, 'first');
bottom = find(margin_2, 1, 'last');
left = find(margin_1, 1, 'first');
right = find(margin_1, 1, 'last');

assert(top<bottom && left<right, 'Top/Bottom & Left/Right are not consistent');
width = right-left+1;
height = bottom-top+1;

% Crop the image
depth_image.cdata = depth_image.cdata(top:bottom, left:right, 1);
depth_image.alpha = depth_image.alpha(top:bottom, left:right);

target_window = 190;
margin = 30;
if height > width,
    scaler = [target_window NaN];
else
    scaler = [NaN target_window];    
end

% Scale the image
depth_im = imresize(depth_image.cdata, scaler, 'lanczos2');
depth_mask = imresize(depth_image.alpha, scaler, 'nearest') == 255;

average_depth = mean(depth_im(depth_mask));
depth_im(depth_mask) = uint8(int64(depth_im(depth_mask)) + int64(55-average_depth));
depth_im(~depth_mask) = 255;

depth_im = single(depth_im)/255;
[height, width] = size(depth_im);

final_depth_image = ones(target_window + 2*margin, target_window + 2*margin, 'single');
final_depth_mask = zeros([target_window + 2*margin, target_window + 2*margin], 'uint8');
loc_x = floor((target_window + 2*margin - width)/2)+1;
loc_y = floor((target_window + 2*margin - height)/2)+1;

final_depth_image(loc_y:loc_y+height-1, loc_x:loc_x+width-1) = depth_im;
final_depth_mask(loc_y:loc_y+height-1, loc_x:loc_x+width-1) = depth_mask*255;

%% Load and prepare the network.
net = load('hardpose_69k_dag');
net = dagnn.DagNN.loadobj(net);

net.eval({'data', final_depth_image});

prediction = net.getVar('prob').value;
%% Visualize the predcitions

[~, classes] = max(prediction, [], 3);
classes(~final_depth_mask) = 0;
imagesc(classes);

