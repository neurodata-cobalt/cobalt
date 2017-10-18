%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Hessian Filter
% 
% Author: Min Zhang
% Date: 12/01/2013 
% Email: mzhang33@asu.edu
% 
%
%
% H=Hessianclass2D(M,class)
% INPUTS: M - Input Image
%     class - 1: positive definite hessian
%             2: positive semidefinite hessian
%             3: negative definite
%             4: negative semidefinite hessian
%
% OUTPUS:    H - Region populated by Hessian Filter 
%
%
% Example: H=Hessianclass2D(img,3)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function H=Hessianclass2D(M,class)
[fx fy]=gradient(M);
[fxx fxy]=gradient(fx);
[fxy fyy]=gradient(fy);
H1=logical(zeros(size(M)));
H2=logical(zeros(size(M)));
H=logical(zeros(size(M)));
switch class
    case 1 % positive
        H1(fxx>0)=1;
        H2((fxx.*fyy-fxy.*fxy)>0)=1;
        H(H1 & H2)=1;
    case 2 % semipositive
        H1(fxx>=0)=1;
        H2((fxx.*fyy-fxy.*fxy)>=0)=1;
        H(H1 & H2)=1;
    case 3 % negative
        H1(fxx<0)=1;
        H2((fxx.*fyy-fxy.*fxy)>0)=1;
        H(H1 & H2)=1;
    case 4 % seminegative
         H1(fxx<=0)=1;
        H2((fxx.*fyy-fxy.*fxy)>=0)=1;
        H(H1 & H2)=1;
end