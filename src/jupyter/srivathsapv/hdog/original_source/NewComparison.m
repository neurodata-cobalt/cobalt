clc;clear all;
path='E:\Dropbox\ResearchProj\cell\groundTruth\PathImgs\';
gtdatapath=strcat(path,'*.mat');
gtdatapath=dir(gtdatapath);


path2='E:\Dropbox\ResearchProj\cell\groundTruth\PathImgs\gLoGResult\';
glogdatapath=strcat(path2,'*.mat');
glogdatapath=dir(glogdatapath);

path3='E:\Dropbox\ResearchProj\cell\symradial\';
symradialdatapath=strcat(path3,'*.mat');
symradialdatapath=dir(symradialdatapath);

path4='E:\Dropbox\ResearchProj\cell\circular\';
circularpath=strcat(path4,'*_aftrim.mat');
circularpath=dir(circularpath);
shtruefalse=zeros(length(gtdatapath),8,17);
for i=1:length(gtdatapath)
    disp(['Processing image - ' num2str(i)]);
    gc=[];rsym=[];clog=[];
    load(strcat(path,num2str(i),'.mat'));
    img=imread(strcat(path,num2str(i),'.tif'));
    img=double(rgb2gray(img));
    img=(img-min(img(:)))./(max(img(:))-min(img(:)));
    load(strcat(path2,num2str(i),'_blobCenters.mat'));
    gc(:,1)=coord(:,2);
    gc(:,2)=coord(:,1);clear coord;
    load(strcat(path3,num2str(i),'_blobCenters.mat'));
    rsym(:,1)=coord(:,2);
    rsym(:,2)=coord(:,1);clear coord;
    load(strcat(path4,num2str(i),'_blobCenters_aftrim.mat'));
    clog(:,1)=coord(:,2);
    clog(:,2)=coord(:,1);clear coord;
    tic;

    [img2 nblob_coords H rho t]=HessianPreSegDoG(img,2,1);
    [G L]=FeatureExtractionRegion(img2,img,H);
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 10 11]);
    tm=toc;
        d(i,1)=2.*rho;
        d(i,2)=t;
        d(i,3)=tm;
        u=1;
        for rho=0:1:16
            
            [shtruefalse(i,1,u) shtruefalse(i,2,u)]=NPrecisionRecall(blob_coords,coords,rho);
            [shtruefalse(i,3,u) shtruefalse(i,4,u)]=NPrecisionRecall(gc,coords,rho);
            [shtruefalse(i,5,u) shtruefalse(i,6,u)]=NPrecisionRecall(rsym,coords,rho);
            [shtruefalse(i,7,u) shtruefalse(i,8,u)]=NPrecisionRecall(clog,coords,rho);
            u=u+1;
        end
end
parameter=zeros(length(gtdatapath),12,17);
parameter(:,1:2,:)=shtruefalse(:,1:2,:);
parameter(:,4:5,:)=shtruefalse(:,3:4,:);
parameter(:,7:8,:)=shtruefalse(:,5:6,:);
parameter(:,10:11,:)=shtruefalse(:,7:8,:);
parameter(:,3,:)= parameter(:,1,:).* parameter(:,2,:).*2./( parameter(:,1,:)+parameter(:,2,:));
parameter(:,6,:)= parameter(:,4,:).* parameter(:,5,:).*2./( parameter(:,4,:)+parameter(:,5,:));
parameter(:,9,:)= parameter(:,8,:).* parameter(:,7,:).*2./( parameter(:,7,:)+parameter(:,8,:));
parameter(:,12,:)= parameter(:,10,:).* parameter(:,11,:).*2./( parameter(:,10,:)+parameter(:,11,:));
save DoG_ComparisonResult15Cell_range.mat