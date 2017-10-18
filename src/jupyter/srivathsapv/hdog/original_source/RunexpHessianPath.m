%% Hessian Prescreening 

clc;clear all;
path='E:\Dropbox\ResearchProj\cell\groundTruth\PathImgs\';
gtdatapath=strcat(path,'*.mat');
gtdatapath=dir(gtdatapath);



path2='E:\Dropbox\ResearchProj\cell\groundTruth\PathImgs\gLoGResult\';
glogdatapath=strcat(path2,'*.mat');
glogdatapath=dir(glogdatapath);

i=15;

for i=1:length(gtdatapath)
    disp(['Processing image - ' num2str(i)]);
    gc=[];
    load(strcat(path,num2str(i),'.mat'));
    img=imread(strcat(path,num2str(i),'.tif'));
    img=double(rgb2gray(img));
    img=(img-min(img(:)))./(max(img(:))-min(img(:)));
    load(strcat(path2,num2str(i),'_blobCenters.mat'));
    gc(:,1)=coord(:,2);
    gc(:,2)=coord(:,1);clear coord;
    tic;
    %[blob_coords rho t]=HessianPrescreen(img,2,1,'',0.08);
    [img2 nblob_coords H rho t]=HessianPreSeg(img,2,1); 
    tm=toc;
        parameter(i,1)=2.*rho;
        parameter(i,2)=t;
        [parameter(i,3) parameter(i,4)]=NPrecisionRecall(nblob_coords,coords,2*rho);
        parameter(i,5)=parameter(i,3).*parameter(i,4).*2./(parameter(i,3)+parameter(i,4));
end
save HessianPrescreenPath.mat