%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Extract Regional Features 
% 
% Author: Min Zhang
% Date: 12/01/2013 
% Email: mzhang33@asu.edu
% 
%
%
% [G L]=FeatureExtractionRegion(img2,img,H)
% INPUTS: img2 - best section of normalized LoG Space
%          img - raw image
%            H - Candidates populated by Hessian Prescreening
%
% OUTPUS:    G - Feature set 
%       G(:,1) - index of candidates
%       G(:,2) - average intensity 
%       G(:,3) - size 
%       G(:,4) - max likelihood of blobness
%       G(:,5) - max structureness
%       G(:,6) - max blobness
%       G(:,7) - mean likelihood of blobness
%       G(:,8) - mean structureness
%       G(:,9) - mean blobness
%      G(:,10) - regional likelihood of blobness
%      G(:,11) - regional structureness
%      G(:,12) - regional blobness
%
%            L - segmentation of candidate 
%
% Example: [G L]=FeatureExtraction(img2,img,H)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [G L]=FeatureExtractionRegion(img2,img,H)
L=bwlabeln(H);

img2=(img2-min(img2(:)))./(max(img2(:))-min(img2(:)));
[dx dy]=gradient(img2);
[dxx dxy]=gradient(dx);
[dxy dyy]=gradient(dy);

infd=find(L>0);
LL=L;
LL=LL(infd);
LL=LL(:);
% zero order features
intensity=img(infd);
intensity=intensity(:);
ns=ones(size(LL));

% second order features
T=dxx+dyy;
D=dxx.*dyy-dxy.*dxy;
blob_like=abs(2.*D)./(T.^2-2.*D);


structureness=sqrt(T.^2-2.*D);

blobness=(1-exp(-blob_like.^2./0.5)).*(1-exp(-structureness.^2./0.5));


blob_like=blob_like(infd);
blob_like=blob_like(:);


blobness=blobness(infd);
blobness=blobness(:);


structureness=structureness(infd);
structureness=structureness(:);


dxx=dxx(infd);dxx=dxx(:);
dyy=dyy(infd);dyy=dyy(:);
dxy=dxy(infd);dxy=dxy(:);

Tol=max(LL);
G=zeros(Tol,14);
G(:,1)=1:Tol;
G(:,2)=accumarray(LL,intensity,[], @(x) mean(x));
G(:,3)=accumarray(LL,ns,[],@(x) sum(x));

G(:,4)=accumarray(LL,blob_like,[],@(x) max(x));
G(:,5)=accumarray(LL,structureness,[],@(x) max(x));
G(:,6)=accumarray(LL,blobness,[],@(x) max(x));

G(:,7)=accumarray(LL,blob_like,[],@(x) mean(x));
G(:,8)=accumarray(LL,structureness,[],@(x) mean(x));
G(:,9)=accumarray(LL,blobness,[],@(x) mean(x));

G(:,10)=accumarray(LL,dxx,[],@(x) sum(x));
G(:,11)=accumarray(LL,dyy,[],@(x) sum(x));
G(:,12)=accumarray(LL,dxy,[],@(x) sum(x));
G(:,13)=2.*abs(G(:,10).*G(:,11)-G(:,12).^2)./((G(:,10)+G(:,11)).^2-2.*(G(:,10).*G(:,11)-G(:,12).^2));
G(:,14)=sqrt((G(:,10)+G(:,11)).^2-2.*(G(:,10).*G(:,11)-G(:,12).^2));
G(:,15)=(1-exp(-G(:,13).^2./0.5)).*(1-exp(-G(:,14).^2./0.5));
G(:,[10 11 12])=[];