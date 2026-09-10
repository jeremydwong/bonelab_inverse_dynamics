function [Local] = G2L(LCS,Global)
% Transforms the global coordinates (Global) into the local coordinates
% (Local) using the inverse of the local coordinate system (LCS).

%LCS: Local coordinate system for the sement, x=columns 1:3, y=columns 4:6, z=columns 7:9
%           rows contain frames of data
%Global: The global coordinates of some quantity such as joint moments
%Local: The local coordinates of the same quantitiy as global


  for i=1:size(Global,1)
    Tinv=inv([LCS(i,1:3);LCS(i,4:6);LCS(i,7:9)]);
    Local(i,:) = Tinv*Global(i,:)';
  end
    
end