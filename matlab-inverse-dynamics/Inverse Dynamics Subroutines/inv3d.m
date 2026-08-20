function [Rp, Mp] = inv3d(CM, Jp, Jd, Rd, Md, aCM, aANG, m, I,g)
% 3-D inverse dynamics calculation for one segment
%
% input:
% CM:		position(x,y,z) of center of mass of the segment
% Jp, Jd: 	position(x,y,z) of proximal and distal endpoint of segment (Jd is the COP for the foot)
% Rd, Md:	force(x,y,z) and moment acting on this segment at distal endpoint
% aCM:		acceleration(x,y,z) of segment center of mass
% aANG:	  angular acceleration (alpha, beta, gamma)
% m         	mass
% I               moment of inertia, relative position of CM
%
% output:
% Rp:		force(x,y,z) at proximal endpoint (force acting on proximal segment!)
% Mp:		moments about x,y,z at the proximal endpoint



%  m=[1.16 1.16 1.16];
%  I=[0.009684 0 0];
%  aANG=[12.18 0 0];
%  CM=[ 0 0.93  0.061];
%  Jp=[0  0.849 .110];
%  Jd=[ 0 .8563 0];
%  Rd=[ 0  -149.96  878.73];
%  aCM=[0 -.9  2.25];
%  Md=[0 0 0];
 rows=size(Jp,1); 
 m=repmat(m,rows,1);
 g=repmat(g,rows,1);

% calculate proximal reaction forces
Rp=m.*aCM - Rd - m.*g;
%Rp= - Rd - m.*g;

% calculate proximal moments
%Mp = I.*aANG - Md - cross(Jd-CM,Rd) - cross(Jp-CM,Rp);%use this if I is a 1 X 3 array
Mp = multiprod(I,aANG,[2 3],[2]) - Md - cross(Jd-CM,Rd) - cross(Jp-CM,Rp);%use this if I is a frames X 3 X 3 matrix
%Mp =  - Md - cross(Jd-CM,Rd) - cross(Jp-CM,Rp);%use this if I is a frames X 3 X 3 matrix




