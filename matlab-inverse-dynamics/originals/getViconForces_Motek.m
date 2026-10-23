%This subroutine is specific for someone running/walking away from the wall 
%on the Motek-Vicon configuration in the exercise physiology lab at the 
%University of Calgary (Winter 2025). This subroutine assumes the origin of
%the mocap system is the SW corner, marked using the three small reflective
%markers.
%
%This subroutine takes the AnalogSignals extracted from the .c3d file from 
%vicon and exports the reaction forces, reaction moments, and center of 
%pressure in the vicon coordinate system.

%Currently this program is written for someone running on a single force plate of a motek treadmill

%Inputs:
%AnalogSignals = raw analog force data from readC3D
%sf_Analog = sampling frequency of AnalogSignals data (likely the mocap
%sampling frequency, as AnalogSignals has likely already been downsampled)
%RM_fp = rotation matrix for converting force data into ISB (SIMM) CS
%FP_offsets = A 2 x 3 matrix containing the force plate offsets (in m) from Vicon
%Slope = slope of the treadmill

%Outputs:
%Rd = ground reaction force in ISB CS (Filtered)
%Md = moments in ISB CS (Filtered)
%COP = centre of pressure, adjusted for FP offset, in ISB CS (Filtered)
%plate = plate number that is being used

function [Rd, Md, COP, plate] = getViconForces_Motek(AnalogSignals,sf_Analog,RMfp,origin,FP_offsets,slope)

Fx1 = AnalogSignals(:,1);
Fy1 = AnalogSignals(:,2);
Fz1 = AnalogSignals(:,3);
Mx1 = AnalogSignals(:,4);
My1 = AnalogSignals(:,5);
Mz1 = AnalogSignals(:,6);
Fx2 = AnalogSignals(:,7);
Fy2 = AnalogSignals(:,8);
Fz2 = AnalogSignals(:,9);
Mx2 = AnalogSignals(:,10);
My2 = AnalogSignals(:,11);
Mz2 = AnalogSignals(:,12);

f_lowpass = 18;

%Combine force plate and moment data into single arrays account for
%analog/vicon discrepency in Fy and get reaction force (-)
Rd1 = [Fx1 Fy1 Fz1];
Rd2 = [Fx2 Fy2 Fz2];
Md1 = [Mx1 My1 Mz1]./1000;
Md2 = [Mx2 My2 Mz2]./1000;

%Rotate into ISB (SIMM) CS and Filter data with a lowpass filter to remove vibration from Motek
Rd1 = NewCS(origin,RMfp,filterdata(Rd1,sf_Analog,0,f_lowpass,4,0));
Rd2 = NewCS(origin,RMfp,filterdata(Rd2,sf_Analog,0,f_lowpass,4,0));
Md1 = NewCS(origin,RMfp,filterdata(Md1,sf_Analog,0,f_lowpass,4,0));
Md2 = NewCS(origin,RMfp,filterdata(Md2,sf_Analog,0,f_lowpass,4,0));

while length(Rd1) < length(Fx1) Rd1 = [Rd1;Rd1(end,:)]; end
while length(Rd2) < length(Fx2) Rd2 = [Rd2;Rd2(end,:)]; end
while length(Md1) < length(Mx1) Md1 = [Md1;Md1(end,:)]; end
while length(Md2) < length(Mx2) Md2 = [Md2;Md2(end,:)]; end

%determine which plate subject is running on
if mean(Rd1(:,2))>mean(Rd2(:,2))%they are running on 1st plate
    Rd = Rd1;
    Md = Md1;
    plate = 1;
else %they are running on 2nd plate
    Rd = Rd2;
    Md = Md2;
    plate = 2;
end

%Set up force plate offsets
az0 = -0.0000001;
offset = NewCS(origin,RMfp,FP_offsets(plate,:));

%Calculate center of pressure, accounting for FP offset
copMLLocal = (-Md(:,1)-(Rd(:,3)*az0))./Rd(:,2);
copMLGlobal = copMLLocal + offset(3);
copAPLocal = (Md(:,3)-(Rd(:,1)*az0))./Rd(:,2);
copAPGlobal = copAPLocal + offset(1);
COP = [copAPGlobal zeros(size(copAPGlobal,1),1) copMLGlobal];

%Calculate FM, exporting as external moment Md
FM = Md(:,2) - Rd(:,1).*copMLLocal + Rd(:,3).*copAPLocal;
Md=[zeros(length(COP),1) zeros(length(COP),1) FM];

%Adjust sence of GRF ap and ml if person is running downhill (i.e.,
%backwards).
% if slope<0; 
%     Rd = [-Rd(:,1) -Rd(:,2) Rd(:,3)];
%     Md = [-Md(:,1) -Md(:,2) Md(:,3)];
% end

%rotate and translate Rd, Md, and COP into a the global reference frame.
%Will not rotate (only translate) if treadmill slope if zero.
if slope == -10
    slope=10;
elseif slope == -5
    slope =5;
end
RMx = [1 0 0;0 cosd(slope) -sind(slope);0 sind(slope) cosd(slope)];
Rd = Rd*RMx';
Md = Md*RMx';
COP = COP*RMx';
end