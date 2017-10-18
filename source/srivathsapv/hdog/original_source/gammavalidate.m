function [H2 img2]=gammavalidate(img,gamma)
 j=1;for i=1:0.5:8
    sigma=i;
    %n = ceil(sigma*3)*2+1;
    tmpim=sigma^gamma*imfilter(img,fspecial('gaussian',size(img),sigma));
    [dx dy]=gradient(tmpim);
    [dxx dxy]=gradient(dx);
    [dxy dyy]=gradient(dy);
    img2(:,:,j)=dxx+dyy;
    H2(:,:,j)=Hessianclass2D(img2(:,:,j),3);
    j=j+1;
 end
H2=bwlabeln(H2);