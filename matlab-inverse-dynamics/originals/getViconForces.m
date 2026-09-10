%This subroutine is specific to the Bertec-Vicon configuration in the
%exercise physiology lab at the University of Calgary (Winter 2015). This
%subroutine takes the AnalogSignals extracted from the .c3d file from vicon
%and exports the reaction forces, reaction moments, and center of pressure
%in the vicon coordinate system.

%Currently this program is written for someone running on a signle force
%bertec force platform

function [Rd Md COP plate] = getViconForces(AnalogSignals,slope);

Fx1 = AnalogSignals(:,1);Fy1 = AnalogSignals(:,2);Fz1 = AnalogSignals(:,3);
Mx1 = AnalogSignals(:,4);My1 = AnalogSignals(:,5);Mz1 = AnalogSignals(:,6);
Fx2 = AnalogSignals(:,7);Fy2 = AnalogSignals(:,8);Fz2 = AnalogSignals(:,9);
Mx2 = AnalogSignals(:,10);My2 = AnalogSignals(:,11);Mz2 = AnalogSignals(:,12);

%Combine force plate and moment data into single arrays account for
%analog/vicon discrepency in Fy and get reaction force (-)
Rd1 = -[Fx1 -Fy1 Fz1];
Rd2 = -[Fx2 -Fy2 Fz2];
Md1 = -[Mx1 -My1 Mz1]./1000;
Md2 = -[Mx2 -My2 Mz2]./1000;

%determine which plate subject is running on
% if mean(Rd1(:,3))>mean(Rd2(:,3));%they are running on 1st plate
%     Rd = Rd1;
%     Md = Md1;
%     plate = 1;
% else %they are running on 2nd plate
    Rd = Rd2;
    Md = Md2;
    plate = 2;
% end

%Calculate center of pressure and free moments
h = 0.015;%15mm
COPap = (-h.*Rd(:,1)-Md(:,2))./Rd(:,3);
COPml = (h.*Rd(:,2)+Md(:,1))./Rd(:,3);
COP = [COPap COPml zeros(length(Rd),1)];

%now caculate the free moment and export as Md
FM = Md(:,3)-COPap.*Rd(:,2)+COPml.*Rd(:,1);
%External Moments
Md=[zeros(length(COP),1) zeros(length(COP),1) FM];

%rotate and translate Rd, Md, and COP into a the global reference frame.
%Will not rotate (only translate) if treadmill slope if zero.
RMx = [1 0 0;0 cosd(slope) -sind(slope);0 sind(slope) cosd(slope)];
Rd = Rd*RMx';
Md = Md*RMx';
COP = COP*RMx';

%now fix COP offset (this needs to be validated)
if plate==1;
    COP = COP+repmat([0.0363 -0.6334 -0.0068],length(COP),1);
else
    COP = COP+repmat([1.0210 -0.6365 -0.0060],length(COP),1);
end


end