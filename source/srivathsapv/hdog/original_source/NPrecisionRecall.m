%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Evaluation Measures
% 
% Author: Min Zhang
% Date: 12/01/2013 
% Email: mzhang33@asu.edu
% 
%
%
% [precision recall]=NPrecisionRecall(coord,gt,rho)
% INPUTS: coord - coordinates
%            gt - ground truth coordinates
%           rho - diameter
%
%
% OUTPUS:    precision - precision 
%                recall - recall
%
% Example: [precision recall]=NPrecisionRecall(coord,gt,10)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [precision recall]=NPrecisionRecall(coord,gt,rho)
num_PreSeg=length(coord(:,1));
num_sg=length(gt(:,1));
DisMatrix=zeros(num_PreSeg,num_sg);
for i=1:num_PreSeg
    for j=1:num_sg
        DisMatrix(i,j)=sqrt((coord(i,1)-gt(j,1))^2+(coord(i,2)-gt(j,2))^2);
    end
end
RecallMin=min(DisMatrix);
PrecisionMin=min(DisMatrix,[],2);
TP=min(length(find(PrecisionMin<=rho)),length(find(RecallMin<=rho)));
FN=length(find(PrecisionMin>rho));
%TP=min(length(find(PrecisionMin<=rho)),num_sg);

recall=TP/num_sg;
precision=TP/num_PreSeg;
%precision=TP/(TP+FN);