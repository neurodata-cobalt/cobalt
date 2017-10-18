function [NL nblob_coords]=FinalClustering(G,L,blobcoords,features)
IDX= kmeans(G(:,features),2);
[label, model, L1] = vbgmm(G(:,features)', IDX');  
label=label';
% label=kmeans(G(:,features),2);

if(mean(G(label==1,2))<=mean(G(label==2,2))) || (isnan( mean(G(label==2,2))))
    lb=1;
else
    lb=2;
end
D=G(label==lb,:);
it=ismember(L(:),D(:,1));
NL=it.*double(L(:));
NL=reshape(NL,size(L));
newmin=zeros(size(L));
mincoord=sub2ind(size(L),blobcoords(:,2),blobcoords(:,1));
newmin(mincoord)=1;
mincoord1=newmin.*NL;
newidx=find(mincoord1>0);
[nminj nmini]=ind2sub(size(L),newidx);
nblob_coords=[nmini nminj];