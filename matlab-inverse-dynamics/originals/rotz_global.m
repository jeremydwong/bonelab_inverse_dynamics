%ROTZ Rotation about Z axis
%
%	TR = ROTZ(theta)
%
% Returns a homogeneous transformation representing a rotation of theta 
% about the Z axis.
%
% See also: ROTX, ROTY, ROTVEC.

% $Log: rotz.m,v $
% Revision 1.2  2002/04/01 11:47:16  pic
% General cleanup of code: help comments, see also, copyright, remnant dh/dyn
% references, clarification of functions.
%
% $Revision: 1.2 $
% Copyright (C) 1993-2002, by Peter I. Corke

function r = rotz_global(t,TM)
if ndims(TM)==2
    rows=1;
else
    rows = size(TM,1);
end;
t=deg2rad(t);
ct = cos(t);
st = sin(t);
rm =  [ct	st	0;-st	ct	0;0	0	1];
 rm=repmat2(rm, rows, [1 2], 2); 
 if size(TM,1)==3&size(TM,2)==3;TM=repmat2(TM, rows, [1 2], 2); end;

 r=multiprod(TM,rm, [2 3], [2 3]);
%r=squeeze(rm)*TM;


