% Evaluate the content difference
clc; clear all; close all;
addpath('./evaluate3/ST_pub/');
addpath('./evaluate3/');

dirpath = './compares/global_transfer/color_transfer/result/test_outputs/2018-10-8';                      
files = dir(strcat(dirpath,'/*.png'));
scores = [];
for j = 1 : length( files )
    filepath = strcat(files(j).folder, '/', files(j).name);
    imHDR = imread(filepath);
    score = higrade_compute(imHDR);
    scores = [scores score];
    disp([num2str(j), '    ', files(j).name, '    ', num2str(score)]);
end
mean_score = mean(scores);

disp('good');



