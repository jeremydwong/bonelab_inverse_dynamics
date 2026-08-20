function [NewPoint, RM, Tres] = VirtualPoint(Neut,pneut,Mov)
%This function creates a virtual point by forming a rotation matrix from
%at least 3 markers placed on a segment. The virtual point is defined in 
%the neutral position as pneut. It is then calculated in the moving 
%frames by multiplying the tranformation matrix by pneut. The 
%transformation matrix is calculated using SVD.

%Neut:  global coordinates of the marker clusters on the segment 
%       in a neutral position. Format: x1 y1 z1 x2 y2 z2 ...
%pneut: (1 x 3) vector that contains the global coordinates of the
%       virtual point for the nuetral frame of data. This would 
%       typically be the global coordinates of a joint center that 
%       were calculated while standing in a neutral position. 
%Mov:  global coordinates of the marker clusters on the segment during
%       the movement. Format is the same as Neut but rows are now 
%       frames of data.
%NewPoint: (frames x 3) matrix containing the global coordinates of the
%        virutual point for each frame of data. 

  pneut = [pneut 1]';
  RM = zeros(size(Mov,1),3,3);%allocate space for rotation matrix
  for i=1:size(Mov,1)
    if ~isnan(Neut)&~isnan(Mov(i,:));
        [T Tres(i)] = SODER([Neut;Mov(i,:)]); %T is the L2G matrix %Tres are the residuals
        NewPoint(i,:) = (T)*pneut;
        RM(i,:,:) = (T(1:3,1:3))';%export the transpose for a local to global
    else
        NewPoint(i,:)=[NaN NaN NaN NaN];
        Tres(i,:)=NaN;
    end
  end
  NewPoint(:,4)=[]; %delete the column of ones
    
end