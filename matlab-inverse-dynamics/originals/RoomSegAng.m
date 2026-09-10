function [AV AA] = RoomSegAng(TM,sf)
%units - radians
thresh=200;%assume discontinuity if abs(velocity) is > threshold and use other element of matrix

% calculate segment angles
% AP(:,1) = atan2(J1(:,3)  - J2(:,3),  J1(:,2)  - J2(:,2));	% about x
% AP(:,2) = atan2(J1(:,3)  - J2(:,3),  J1(:,1)  - J2(:,1));	% about y
% AP(:,3) = atan2(J1(:,2)  - J2(:,2),  J1(:,1)  - J2(:,1));	% about z

TMdot=FirstCentral([TM(:,1,:) TM(:,2,:) TM(:,3,:)],sf);
TMdot=reshape(TMdot,[length(TMdot) 3 3]);
S=multiprod(TMdot, TM, [2 3], [2 3]);

% remove -pi to pi transitions
% for i=2:n
%   for j=1:3 
%     while (AP(i,j)-AP(i-1,j) > 2.5),
% 	   AP(i,j) = AP(i,j) - 2*pi;
%     end
%     while (AP(i,j)-AP(i-1,j) < -2.5),
% 	   AP(i,j) = AP(i,j) + 2*pi;
%     end 
%   end
% end

AV(:,1)= S(:,3,2);if max(abs(AV(:,1)))>thresh; AV(:,1)=-S(:,2,3);end;
AV(:,2)= S(:,1,3);if max(abs(AV(:,2)))>thresh; AV(:,2)=-S(:,3,1);end;
AV(:,3)= S(:,2,1);if max(abs(AV(:,3)))>thresh; AV(:,3)=-S(:,1,2);end;
% AV=filterdata(AV,sf,0,6,2,0);

if size(TM,1)>3
  AA=FirstCentral(AV,sf);
else 
    AA=[0 0 0];
end;