%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Hessian Prescreening Method
% Determin the Best Scale Section of LoG Space and Returns the initial
% Segmentation results.
% Author: Min Zhang
% Date: 12/01/2013 
% Email: mzhang33@asu.edu
% 
%
%
%[img2 blob_coords H rho idx]=HessianPreSeg(img,gamma,dark)
% INPUTS: img - Raw Grey Image (should be standardized to 0-1)
%       gamma - Normalizing Factor
%        dark - 1: detect dark blobs; 0: detect bright blobs
%
% OUTPUS: img2- best scale section of normalized LoG Space
%  blob_coords- centroids of candidate regions
%            H- Candidate regions
%          rho- estimated diameters 
%          idx- index of best section
%
% Example: [img2 blob_coords H rho idx]=HessianPreSeg(img,2,1)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [img2 blob_coords H rho idx]=HessianPreSeg(img,gamma,dark)
% disp(features);
if (nargin() < 3)
    dark=1;
end

if (nargin() < 2)
    gamma=2;
end

if dark~=1
    img=1-img;
end

j=1;

for i=2:0.5:9
    sigma=i;
    %n = ceil(sigma*3)*2+1;
    tmpim=sigma^gamma*imfilter(img,fspecial('gaussian',size(img),sigma));
    [dx dy]=gradient(tmpim);
    [dxx dxy]=gradient(dx);
    [dxy dyy]=gradient(dy);
    img1(:,:,j)=dxx+dyy;
    j=j+1;
end


clear  dx dy dz dxx dxy dyy ;
for i=1:j-1
    tmpH=Hessianclass2D(img1(:,:,i),3);

    tmp_r=img1(:,:,i).*tmpH;
    tmp_r=sum(abs(tmp_r(:)))./sum(tmpH(:));
    tpidx(i)=tmp_r;
    %plot(tpidx);drawnow;
end

[maxvalue idx]=max(tpidx);
clear tpidx maxvalue;
img2=img1(:,:,idx);

idx=2+(idx-1)*0.5;

H=Hessianclass2D(img2,3);
IMIN=imregionalmax(img2);

%img3=-single(img2);
%img3(IMIN)=-inf;
%L=watershed(img3);
%H=H & L;


IMIN=IMIN.*H;

BW=bwlabeln(H);
BWU=BW.*IMIN;
BWU=unique(BWU(:));
it=ismember(BW,BWU);
H=H.*it;
BWU(1)=[];

for t=1:length(BWU)
    idxtmp=find(BW==BWU(t));
    [tmpvalue tmpcoord ]=max(img2(idxtmp));
    mincoord(t) =idxtmp(tmpcoord);
end

rho=sqrt(sum(H(:))/(length(BWU))/pi);




[minj mini]=ind2sub(size(img2),mincoord);
blob_coords=[mini' minj'];