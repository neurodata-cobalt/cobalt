%% Hessian Prescreening 

clc;clear all;
addpath('E:\Dropbox\ResearchProj\cell\FRST');

path2='E:\cells\new_nothreshold\';

truefalse=zeros(200,8,17);
for i=1:200
    disp(['Processing image - ' num2str(i)]);
    load(strcat('E:\Dropbox\ResearchProj\cell\simulatingdata\',num2str(i),'_data.mat'));
    tic;
    [img2 nblob_coords H rho t]=HessianPreSeg(img,1,0);
    
    [A B]=NPrecisionRecall(nblob_coords,gt,2.*rho);
     if (A<=0.5) &&  (B<=0.5)
         [img2 nblob_coords H rho t]=HessianPreSeg(img,1,1);

     end
    tm=toc;
   coords=gt;
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
save ComparisonResult200_Pre.mat