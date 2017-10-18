
clc;clear all;

for i=1:200
    disp(['Processing image - ' num2str(i)]);
    load(strcat('E:\Dropbox\ResearchProj\cell\simulatingdata\',num2str(i),'_data.mat'));
    tic;
    [img2 nblob_coords H rho t]=HessianPreSeg(img,1,0);
    [G L]=FeatureExtractionRegion(img2,1-img,H);
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 10 11]);
    coords=gt;
    [A B]=NPrecisionRecall(nblob_coords,gt,2.*rho);
     if (A+B)<1
         [img2 nblob_coords H rho t]=HessianPreSeg(img,1,1);
         [G L]=FeatureExtractionRegion(img2,img,H);
         [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 10 11]);
     end
    
  
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 3 4 5 6]);
    [parameter(i,1,1) parameter(i,1,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 3 4 5]);
    [parameter(i,2,1) parameter(i,2,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 3 5]);
    [parameter(i,3,1) parameter(i,3,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
     
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 3 7 8 9]);
    [parameter(i,4,1) parameter(i,4,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 3 7 8]);
    [parameter(i,5,1) parameter(i,5,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 3 9]);
    [parameter(i,6,1) parameter(i,6,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 3 10 11 12]);
    [parameter(i,7,1) parameter(i,7,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 3 10 11]);
    [parameter(i,8,1) parameter(i,8,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 3 12]);
    [parameter(i,9,1) parameter(i,9,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 4 5 6]);
    [parameter(i,10,1) parameter(i,10,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 4 5]);
    [parameter(i,11,1) parameter(i,11,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 5]);
    [parameter(i,12,1) parameter(i,12,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
     
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 7 8 9]);
    [parameter(i,13,1) parameter(i,13,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 7 8]);
    [parameter(i,14,1) parameter(i,14,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 9]);
    [parameter(i,15,1) parameter(i,15,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 10 11 12]);
    [parameter(i,16,1) parameter(i,16,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 10 11]);
    [parameter(i,17,1) parameter(i,17,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    
    [NL blob_coords]=FinalClustering(G,L,nblob_coords,[2 12]);
    [parameter(i,18,1) parameter(i,18,2)]=NPrecisionRecall(blob_coords,coords,2*rho);
    toc;
end
Fscore=parameter(:,:,1).*parameter(:,:,2).*2./(parameter(:,:,1)+parameter(:,:,2));
Fscore_mean=squeeze(mean(Fscore));
Fscore_mean=Fscore_mean';
F_std=std(Fscore);
F_std=F_std';
save testfeatures200.mat
%save testfeatures2-5featurs.mat parameter;