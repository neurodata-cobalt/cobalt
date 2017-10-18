%% Hessian Prescreening 

clc;clear all;

addpath('E:\Dropbox\ResearchProj\cell\FRST');

path='E:\Dropbox\ResearchProj\cell\groundTruth\PathImgs\';
gtdatapath=strcat(path,'*.mat');
gtdatapath=dir(gtdatapath);


path2='E:\Dropbox\ResearchProj\cell\groundTruth\PathImgs\no_threshold\';
glogdatapath=strcat(path2,'*.mat');
glogdatapath=dir(glogdatapath);

path3='E:\Dropbox\ResearchProj\cell\symradial\';
symradialdatapath=strcat(path3,'*.mat');
symradialdatapath=dir(symradialdatapath);

path4='E:\Dropbox\ResearchProj\cell\groundTruth\PathImgs\no_threshold\';
circularpath=strcat(path4,'*_aftrim.mat');
circularpath=dir(circularpath);
for i=1:length(gtdatapath)
    disp(['Processing image - ' num2str(i)]);
    gc=[];rd=[];clog=[];
    load(strcat(path,num2str(i),'.mat'));
    img=imread(strcat(path,num2str(i),'.tif'));
    img=double(rgb2gray(img));
    img=(img-min(img(:)))./(max(img(:))-min(img(:)));
    load(strcat(path2,num2str(i),'_blobCenters_dark.mat'));
    gc(:,1)=coord(:,2);
    gc(:,2)=coord(:,1);clear coord;
    
    load(strcat(path4,num2str(i),'_blobCenters_cirLoG_dark.mat'));
    clog(:,1)=coord(:,2);
    clog(:,2)=coord(:,1);clear coord;
    tic;
    %[blob_coords rho t]=HessianPrescreen(img,2,1,'',0.08);
    [img2 nblob_coords H rho t]=HessianPreSeg(img,2,1);
    
    S=FRST(img, floor(rho),0);
    S(S<=1)=0;
    RD=imregionalmax(S,18);
    [rd(:,2) rd(:,1)]=ind2sub(size(RD),find(RD>0));
    
    tm=toc;
        d(i,1)=2.*rho;
        d(i,2)=t;
        d(i,3)=tm;
        [shtruefalse(i,1) shtruefalse(i,2)]=NPrecisionRecall(nblob_coords,coords,2.*rho);
        [shtruefalse(i,3) shtruefalse(i,4)]=NPrecisionRecall(gc,coords,2.*rho);
        [shtruefalse(i,5) shtruefalse(i,6)]=NPrecisionRecall(rd,coords,2.*rho);
        [shtruefalse(i,7) shtruefalse(i,8)]=NPrecisionRecall(clog,coords,2.*rho);
end
parameter(:,1:2)=shtruefalse(:,1:2);
parameter(:,4:5)=shtruefalse(:,3:4);
parameter(:,7:8)=shtruefalse(:,5:6);
parameter(:,10:11)=shtruefalse(:,7:8);
parameter(:,3)= parameter(:,1).* parameter(:,2).*2./( parameter(:,1)+parameter(:,2));
parameter(:,6)= parameter(:,4).* parameter(:,5).*2./( parameter(:,4)+parameter(:,5));
parameter(:,9)= parameter(:,8).* parameter(:,7).*2./( parameter(:,7)+parameter(:,8));
parameter(:,12)= parameter(:,10).* parameter(:,11).*2./( parameter(:,10)+parameter(:,11));
save ComparisonResult15_Pre.mat