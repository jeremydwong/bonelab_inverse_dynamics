function [NewData] = NewCS(origin,RM,olddata)
%This function transforms the kinematic markers in olddata to a new
%coordinate system that is created using the rotation matrix RM.

%origin is a 1 x 3
%RM is a 3 x 3
%olddata is a rows X (3 times # of points)

  TM=  [[RM  origin']; [0 0 0 1]];
  TMinv = inv(TM);
%   TMinv=repmat2(TMinv, size(olddata,1), [1 2], 2);
%   OD=[olddata repmat(1,size(olddata,1),1)];
%   NewData = multiprod(OD,TMinv,[2],[2 3]);
%   NewData=NewData(:,1:3);
%   
   [olddata_length numcols]=size(olddata);          
  
  %transform each 3D vector
  for c = 1:3:numcols;
    for r = 1:olddata_length;
      DP = [olddata(r,c:c+2) 1]; %DP = [1 ap ml v]
      DPtrans = DP*TMinv;
      NewData(r,c:c+2)= DPtrans(1:3);
    end; %for r
  end; %for c
end %function