% Comparison Experiments
% 
clc;clear all;
addpath('E:\Dropbox\ResearchProj\cell\FRST');

path2='E:\cells\new\';

truefalse=zeros(200,8,17);
for i=1:200
    disp(['Processing image - ' num2str(i)]);
    load(strcat('E:\Dropbox\ResearchProj\cell\simulatingdata\',num2str(i),'_data.mat'));
    tic;
    [img2 nblob_coords H rho t]=HessianPreSegDoG(img,1,0);
    [G L]=FeatureExtractionRegion(img2,1-img,H);
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 10 11]);
    
    [A B]=NPrecisionRecall(nblob_coords,gt,2.*rho);
     if (A<=0.5) &&  (B<=0.5)
         [img2 nblob_coords H rho t]=HessianPreSegDoG(img,1,1);
         [G L]=FeatureExtractionRegion(img2,img,H);
         [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 10 11]);
     end
    tm=toc;
   
    gc=[];rd=[];clog=[];
    if i<10 
        load(strcat(path2,'00',num2str(i),'cell_blobCenters_bright.mat'));
    elseif i<100
        load(strcat(path2,'0',num2str(i),'cell_blobCenters_bright.mat'));
    else
        load(strcat(path2,num2str(i),'cell_blobCenters_bright.mat'));
    end
    gc(:,1)=coord(:,2);
    gc(:,2)=coord(:,1);clear coord;
    
    if i<10 
        load(strcat(path2,'00',num2str(i),'cell_blobCenters_cirLoG_bright_aftrim.mat'));
    elseif i<100
        load(strcat(path2,'0',num2str(i),'cell_blobCenters_cirLoG_bright_aftrim.mat'));
    else
        load(strcat(path2,num2str(i),'cell_blobCenters_cirLoG_bright_aftrim.mat'));
    end
    clog(:,1)=coord(:,2);
    clog(:,2)=coord(:,1);clear coord;
    
    S=FRST(img, floor(rho),0.0003);
    S(S<=1)=0;
    RD=imregionalmax(S,18);
    [rd(:,2) rd(:,1)]=ind2sub(size(RD),find(RD>0));
     d(i,1)=rho;
     d(i,2)=t;
     d(i,3)=tm;
     u=1;
     for rho=0:1:16
        [shtruefalse(i,1,u) shtruefalse(i,2,u)]=NPrecisionRecall(blob_coords,gt,rho);
        [shtruefalse(i,3,u) shtruefalse(i,4,u)]=NPrecisionRecall(gc,gt,rho);
        [shtruefalse(i,5,u) shtruefalse(i,6,u)]=NPrecisionRecall(rd,gt,rho);
        [shtruefalse(i,7,u) shtruefalse(i,8,u)]=NPrecisionRecall(clog,gt,rho);
        u=u+1;
     end
end
parameter=zeros(200,12,17);
parameter(:,1:2,:)=shtruefalse(:,1:2,:);
parameter(:,4:5,:)=shtruefalse(:,3:4,:);
parameter(:,7:8,:)=shtruefalse(:,5:6,:);
parameter(:,10:11,:)=shtruefalse(:,7:8,:);
parameter(:,3,:)= parameter(:,1,:).* parameter(:,2,:).*2./( parameter(:,1,:)+parameter(:,2,:));
parameter(:,6,:)= parameter(:,4,:).* parameter(:,5,:).*2./( parameter(:,4,:)+parameter(:,5,:));
parameter(:,9,:)= parameter(:,8,:).* parameter(:,7,:).*2./( parameter(:,7,:)+parameter(:,8,:));
parameter(:,12,:)= parameter(:,10,:).* parameter(:,11,:).*2./( parameter(:,10,:)+parameter(:,11,:));
save DoG_ComparisonResult200Cell_Range.mat
    