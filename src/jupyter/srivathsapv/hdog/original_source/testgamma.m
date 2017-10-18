for j=0:1:3
[H2 img2]=gammavalidate(img,j);
%A=[2561,1203,1314,631,401,654,1776,574,2116,996];
%A=[692,777,793,561,991,543,299,627,434,508];
A=[5,55,65,164,69,990,913,605,794,304]%130;
DL=H2; 
for i=1:10
    L=DL;
    L(find(L~=A(i)))=0;
    L(L>0)=1;
    ob(:,:,:,i)=img2.*L;
    oa(:,:,:,i)=L;
    %obj2=squeeze(mean(mean(ob(ob~=0,:,:,i))));
end
 LoG=squeeze(sum(sum(sum(ob)),2));
 AR= squeeze(sum(sum(sum(oa)),2));
 obj=LoG./AR;
 LoG=sum(LoG,2)./sum(AR,2);
 %LoG2=squeeze(mean(obj2,2));
 figure;plot(1:0.5:8,obj);hold on;plot(1:0.5:8,LoG,'*-','LineWidth',2);hold off;
 legend('Blob 1','Blob 2','Blob 3','Blob 4','Blob 5','Blob 6','Blob 7','Blob 8','Blob 9','Blob 10','C_r (t)');
 ylabel('Normalized LoG');
 xlabel(['scale t, \gamma = ',num2str(j)]);
end
%figure;plot(obj2);hold on;plot(LoG2,'*-','LineWidth',2);hold off;