function run_legacy_trial(staticMat, dynamicMat, c, spcode, outMat, A1)
%RUN_LEGACY_TRIAL  The inverse-dynamics path of Asics_Moments4JCF_v3.m on one
%trial, with the ORIGINAL routines, saved to a .mat for comparison.
%
%This harness exists because the original is a batch script over subjects,
%shoes and speeds with Z:\ paths, an anthropometrics spreadsheet, Excel
%outputs and SIMM exports, and because no Mac build of the ezc3d MEX exists.
%Inputs are .mat files written by boneid.legacy.export_for_matlab with the
%same fields the script reads from ezc3dRead (points [3xMxT] mm, labels,
%rates, per-plate force [3xTa] N / moment [3xTa] Nmm).  Every computational
%line below is copied from Asics_Moments4JCF_v3.m (line numbers in comments)
%except: the file/loop scaffolding, the optional SIMM/Excel/figure outputs,
%and A(1) which may be overridden (the spreadsheet is not available).
%
%  c      : shoe index into shcond = {'AM','AG','AH'} (1..3), for shoeHeight
%  spcode : 'RP' or 'PI' (only used for the S1 bad-step table)
%  A1     : body mass in kg, or [] to keep the script's default 53.5

%---- constants, Asics_Moments4JCF_v3.m lines 31-56 -----------------------
g=[0 -9.81 0];
origin = [0 0 0];
RMmocap = [0 -1 0; 0 0 1; -1 0 0];
RMfp = [0 -1 0; 0 0 1; -1 0 0];
offsetFP2 = [0.371453 1.040980 -0.0464284];
offsetFP1 = [0.872614 1.04162 -0.0477233];
shoeHeight = [0.03 0.02 0.04]; %m
event_force_threshold_n = 20; % N threshold used to trim HS/TO edges

%---- static trial, lines 74-128 -------------------------------------------
S = load(staticMat);
Markers = S.points;                 % 3 x M x T, as dataStatic.data.points
VideoFrameRate = S.point_rate;
Markers(find(Markers==0))=NaN;%no markers should be zero, replace with NaN (means gap exists in data)
labels = cellstr(S.labels);
for p=1:size(Markers,2)
    mname=labels(p);
    if strcmp('*',mname{1}(1))
        break
    else
        evalstr=[cell2mat(mname) '=squeeze(Markers(:,' num2str(p) ',:)./1000)'';'];
        markernames(p)=mname;
        eval(evalstr);
    end
end
for p=1:size(markernames,2)
    evalstr=[cell2mat(markernames(p)) '=NewCS(origin,RMmocap,nanmean(' cell2mat(markernames(p)) '));'];
    eval(evalstr);
end

%Get Anthopometrics — defaults (no spreadsheet), lines 114-129
A(1:15)=0;
A(1)=53.5; %Total Mass (kg)
A(2)=0.48;%Thigh Length (m)
A(3)=0.46;% Midthigh Circumference (m)
A(4)=0.43;% Calf Length (m)
A(5)=0.33;% Calf Circumference (m)
A(6)=0.25;% Foot Length (m) ???
A(7)=0.1;% Malleolus Height (m)
A(8)=0.08;% Malleolus Width (m)
A(9)=0.11;% Foot Breadth;
A(10)=0.75;% Belly circum;
A(11)=0.3;% Belly width;
A(12)=0.8;% Hip circum;
A(13)=0.3;% Hip width;
A(14)=0.18;% Belly-Hip length;
A(15)=1.8;% Total height;
if ~isempty(A1); A(1)=A1; end

%Define body weight
BW=A(1)*9.81;

%Define stance joint centers and virtual points (lines 137-141)
N_L5S1center=(0.75*((0.5*LPSIS)+(0.5*RPSIS)))+(0.25*((0.5*LASIS)+(0.5*RASIS)));
Nr_hipcenter = (0.70*RGT)+(0.30*LGT);
Nr_kneecenter = (0.50*KneeLat)+(0.50*KneeMed);
Nr_anklecenter = (0.50*AnkleLat)+(0.50*AnkleMed);
Nr_metcenter = (0.50*MTP1Head) + (0.50*MTP5Head);

%Define thigh length, calf length, malleolus height and width (144-147)
A(2) = sqrt(sum((Nr_hipcenter-Nr_kneecenter).^2)); %Thigh length (m)
A(4) = sqrt(sum((Nr_kneecenter-Nr_anklecenter).^2)); %Calf length (m)
A(7) = Nr_anklecenter(2)-shoeHeight(c)-0.0045;% Malleolus Height (m)
A(8) = AnkleLat(3)-AnkleMed(3)-0.009;% Malleolus Width (m)

%Define leg length
legLength = A(2)+A(4)+A(7);

%Define stance marker clusters (153-158)
N_pelvis = [RASIS LASIS RPSIS LPSIS RGT LGT];%neutral pelvis marker cluster
Nr_thigh = [ThighAntProx ThighPostDist ThighPostProx];%neutral thigh marker cluster
Nr_leg=[ShankAntProx ShankPostDist ShankPostProx];%neutral leg marker cluster
Nr_foot=[HeelLat HeelMed DorsalFoot];%neutral foot marker cluster
Nr_rearfoot = [HeelLat HeelMed DorsalFoot]; %neutral rearfoot marker cluster
Nr_forefoot = [MTP1Head MTP5Head ToeboxAnt]; %neutral forefoot marker cluster

%Define/create a virtual toe marker
Nr_footlength = ToeboxAnt(1) - HeelMed(1) ;%footlength + marker to heel distance
N_MTP1Head = MTP1Head;

%Segment Masses, From Vaughan, Dynamics of Human Gait (169-173)
m=(0.0083*A(1))+(254.5*A(6)*A(7)*A(8))-0.065;mfoot=[m m m];
m=(0.0223*A(1))+(31.33*A(4)*A(5)^2)-0.016;mleg=[m m m];
m=(0.1032*A(1))+(12.76*A(2)*A(3)^2)-1.023;mthigh=[m m m];
m=(0.1032*A(1))+(12.76*A(2)*A(3)^2)-1.023;mpelvis=[m m m];

%mass of hindfoot/forefoot are functions of segment x-length (176-177)
mrearfoot = (Nr_metcenter(1)-HeelMed(1))/(ToeboxAnt(1)-HeelMed(1))*mfoot;
mforefoot = (ToeboxAnt(1)-Nr_metcenter(1))/(ToeboxAnt(1)-HeelMed(1))*mfoot;

%Local segment moments of inertia for the rearfoot & forefoot (180-185)
Irfx = (1/12)*mrearfoot(1)*(3*((MTP5Head(3)-MTP1Head(3))*0.5)^2+((Nr_metcenter(1)-HeelMed(1)))^2);
Irfy = (1/12)*mrearfoot(1)*(3*(Nr_anklecenter(2)*0.5)^2+((Nr_metcenter(1)-HeelMed(1)))^2);
Irfz = (1/4)*mrearfoot(1)*(((MTP5Head(3)-MTP1Head(3))*0.5)^2+3*(Nr_anklecenter(2)*0.5)^2);
Iffx = (1/12)*mforefoot(1)*(3*((MTP5Head(3)-MTP1Head(3))*0.5)^2+((ToeboxAnt(1)-Nr_metcenter(1)))^2);
Iffy = (1/12)*mforefoot(1)*(3*(ToeboxAnt(2)*0.5)^2+((ToeboxAnt(1)-Nr_metcenter(1)))^2);
Iffz = (1/4)*mforefoot(1)*(((MTP5Head(3)-MTP1Head(3))*0.5)^2+(ToeboxAnt(2)*0.5)^2);

%makes virtual marker for Sacrum with ASIS vertical coordinate (188-190)
N_mid_ASIS =(.50*RASIS)+(.50*LASIS);
Sacrum = (0.50*RPSIS)+(0.50*LPSIS);
N_Sacrum_ASISy=[Sacrum(1) N_mid_ASIS(2) Sacrum(3)];

%Calculate segment lengths (193-195)
seg.R.thigh.length=norm(Nr_hipcenter-Nr_kneecenter);
seg.R.leg.length=norm(Nr_kneecenter-Nr_anklecenter);
seg.R.foot.length=norm(Nr_footlength);
SIMMTibialength = 0.439;  %SIMM tibia-femur contact to lateral maleolus
r_scaleTibia = seg.R.leg.length/SIMMTibialength;

%---- dynamic trial, lines 242-286 ------------------------------------------
D = load(dynamicMat);
Markers = D.points;
VideoFrameRate = D.point_rate;
AnalogFrameRate = D.analog_rate;
AnalogSignals = [D.force1',D.moment1',D.force2',D.moment2'];
clear markernames;
labels = cellstr(D.labels);
for p=1:size(Markers,2)
    mname=labels(p);
    if ~strcmp('*',mname{1}(1))
        evalstr=[cell2mat(mname) '=squeeze(Markers(:,' num2str(p) ',:)./1000)'';'];
        markernames(p)=mname;
        eval(evalstr);
    end
end
for p=1:size(markernames,2)
    evalstr=[cell2mat(markernames(p)) '=NewCS(origin,RMmocap,' cell2mat(markernames(p)) ');'];
    eval(evalstr);
end

%Define treadmill slope
slope = 0;

%interpolate GRF to be same as motion (277-279)
vtime = (0:1/VideoFrameRate:(length(Markers)-1)/VideoFrameRate)';
atime = (0:1/AnalogFrameRate:(length(AnalogSignals)-1)/AnalogFrameRate)';
intAnalogSignals = interp1(atime,AnalogSignals,vtime,'pchip');

%Extract Ground reaction force data (284-285)
FP_offsets = [offsetFP1;offsetFP2];
[Rd, Md, COP, plate] = getViconForces_Motek(intAnalogSignals, VideoFrameRate, RMfp, origin, FP_offsets, slope);

%Initial event detection (288)
[HS, TO, ind_off] = ContactEvents_Motek(Rd(:,2),COP(:,3),VideoFrameRate);

%Tighten HS/TO boundaries to avoid low-force transition samples. (291-299)
for ev = 1:length(HS)
    idx = HS(ev):TO(ev);
    in_contact = find(Rd(idx,2) > event_force_threshold_n,1,'first');
    out_contact = find(Rd(idx,2) > event_force_threshold_n,1,'last');
    if ~isempty(in_contact) && ~isempty(out_contact)
        HS(ev) = idx(in_contact);
        TO(ev) = idx(out_contact);
    end
end

%Rebuild off-ground indices from refined right-foot events. (302-310)
if ~isempty(HS)
    ind_off = [(1:HS(1))'; (TO(end):length(Rd))'];
    for ev = 1:length(TO)-1
        ind_off = [ind_off; (TO(ev):HS(ev+1))'];
    end
    ind_off = unique(ind_off);
else
    ind_off = (1:length(Rd))';
end

%Remove GRF data for right-foot off-ground periods (313-315)
Rd(ind_off,:)=0;
Md(ind_off,:)=0;
COP(ind_off,:)=[DorsalFoot(ind_off,1) zeros(length(ind_off),1) DorsalFoot(ind_off,3)];

%Define dynamic marker clusters (325-330)
pelvis = [RASIS LASIS RPSIS LPSIS RGT LGT];%pelvis marker cluster
r_thigh = [ThighAntProx ThighPostDist ThighPostProx];%thigh marker cluster
r_leg=[ShankAntProx ShankPostDist ShankPostProx];%leg marker cluster
r_foot=[HeelLat HeelMed DorsalFoot];%foot marker cluster
r_rearfoot = [HeelLat HeelMed DorsalFoot]; %neutral rearfoot marker cluster
r_forefoot = [MTP1Head MTP5Head ToeboxAnt]; %neutral forefoot marker cluster

%Calculate virtual points (334-342)
[R_Jmtp,R_ForefootTM,~] = VirtualPoint_v2(Nr_forefoot,Nr_metcenter,r_forefoot);
[~, R_RearfootTM,~] = VirtualPoint_v2(Nr_rearfoot,Nr_anklecenter,r_rearfoot);
[R_Jankle, R_FootTM, R_FootRes] = VirtualPoint_v2(Nr_foot,Nr_anklecenter,r_foot);
[R_Jknee, R_LegTM, R_LegRes] = VirtualPoint_v2(Nr_leg,Nr_kneecenter,r_leg);
[R_Jhip, R_ThighTM, R_ThighRes] = VirtualPoint_v2(Nr_thigh,Nr_hipcenter,r_thigh);
[JL5S1, PelvisTM, R_PelvisRes] = VirtualPoint_v2(N_pelvis,N_L5S1center,pelvis);
[VPSacrum_ASISy] = VirtualPoint_v2(N_pelvis,N_Sacrum_ASISy,pelvis);
[VPmid_ASIS] = VirtualPoint_v2(N_pelvis,N_mid_ASIS,pelvis);
mid_Hip=(0.5*RGT)+(0.5*LGT);

%Local segment moments of inertia, From Vaughan (346-354)
Ifx=0.00021*A(1)*(4*A(9)^2+3*A(6)^2)+0.00067;
Ifz=(0.00023*A(1))*(4*A(7)^2+3*A(6)^2)+0.00022;
Ify=0.00141*A(1)*(A(7)^2+A(9)^2)-0.00008;
Ilx=0.00387*A(1)*(A(4)^2+.076*A(5)^2)+0.00138;
Ilz=(0.00347*A(1))*(A(4)^2+0.076*A(5)^2)+0.00511;
Ily=0.00041*A(1)*A(9)^2+0.00012;
Itx=0.00726*A(1)*(A(2)^2+0.076*A(3)^2)+0.01186;
Itz=(0.00762 * A(1)) * (A(2)^2 + 0.076 * A(3)^2) + 0.01153;
Ity= 0.00151*A(1)*A(6)^2+0.00305;

%Local segment moments of inertia for pelvis,, From Zatsiorsky-Seluyanov (357-359)
Ipx=-1568+(12*A(1))+(7.741*A(15)*100);
Ipz=-934+(11.8*A(1))+(3.44*A(15)*100);
Ipy=-775+(14.7*A(1))+(1.685*A(15)*100);

Ifoot=[Ifx 0 0;0 Ify 0; 0 0 Ifz];
Ileg=[Ilx 0 0;0 Ify 0; 0 0 Ilz];
Ithigh=[ Itx 0 0;0 Ity 0; 0 0 Itz];
Ipelvis=[Ipx 0 0;0 Ipy 0; 0 0 Ipz]*1e-4;
Irearfoot = [Irfx 0 0;0 Irfy 0; 0 0 Irfz];
Iforefoot = [Iffx 0 0;0 Iffy 0; 0 0 Iffz];

%Calculate global moments of inertia (369-380)
Irearfoot = matexp(Irearfoot, length(R_RearfootTM), [1 2], 2);%replicate the I matrix for each frame of data
IrearfootG = multiprod(R_RearfootTM, Irearfoot, [2 3], [2 3]);%multiply I matrix by TM to get global moments of inertia
Iforefoot = matexp(Iforefoot, length(R_ForefootTM), [1 2], 2);%replicate the I matrix for each frame of data
IforefootG = multiprod(R_ForefootTM, Iforefoot, [2 3], [2 3]);%multiply I matrix by TM to get global moments of inertia
Ifoot = matexp(Ifoot, length(R_FootTM), [1 2], 2);%replicate the I matrix for each frame of data
IfootG=multiprod(R_FootTM,Ifoot,[2 3], [2 3]);%multiply I matrix by TM to get global moments of inertia
Ileg = matexp(Ileg, length(R_LegTM), [1 2], 2);%replicate the I matrix for each frame of data
IlegG=multiprod(R_LegTM, Ileg, [2 3], [2 3]);%multiply I matrix by TM to get global moments of inertia
Ithigh = matexp(Ithigh, length(R_ThighTM), [1 2], 2);%replicate the I matrix for each frame of data
IthighG=multiprod(R_ThighTM, Ithigh, [2 3], [2 3]);%multiply I matrix by TM to get global moments of inertia
Ipelvis = matexp(Ipelvis, length(PelvisTM), [1 2], 2);%replicate the I matrix for each frame of data
IpelvisG=multiprod(PelvisTM, Ipelvis, [2 3], [2 3]);%multiply I matrix by TM to get global moments of inertia

%Segment COM locations, From Vaughn (383-386)
R_CMfoot=0.44+(0.0195/A(6));%correct percentage for heel to heel marker distance
R_CMleg=0.42;
R_CMthigh=0.39;
CMpelvis=0.50;%just a guess

%COM Locations (389-394)
R_CMrearfoot = HeelMed + 0.5*(R_Jmtp - HeelMed);
R_CMforefoot = R_Jmtp + 0.5*(ToeboxAnt - R_Jmtp);
R_CMfoot=HeelLat-(R_CMfoot*(HeelLat-ToeboxAnt));
R_CMleg=R_Jknee-(R_CMleg*(R_Jknee-R_Jankle));
R_CMthigh=R_Jhip-(R_CMthigh*(R_Jhip-R_Jknee));
CMpelvis=JL5S1-(CMpelvis*(JL5S1-mid_Hip));

%COM Accelerations (397-402)
aR_CMrearfoot = FirstCentral(FirstCentral(R_CMrearfoot,VideoFrameRate),VideoFrameRate);
aR_CMforefoot = FirstCentral(FirstCentral(R_CMforefoot,VideoFrameRate),VideoFrameRate);
aR_CMfoot=FirstCentral(FirstCentral(R_CMfoot,VideoFrameRate),VideoFrameRate);
aR_CMleg=FirstCentral(FirstCentral(R_CMleg,VideoFrameRate),VideoFrameRate);
aR_CMthigh=FirstCentral(FirstCentral(R_CMthigh,VideoFrameRate),VideoFrameRate);
a_CMpelvis=FirstCentral(FirstCentral(CMpelvis,VideoFrameRate),VideoFrameRate);

%Cardan Segment Angles (405-411)
sequence = 'zxy';
FFA = CardanSegAngles(R_ForefootTM,sequence);
RFA = CardanSegAngles(R_RearfootTM, sequence);
FA = CardanSegAngles(R_FootTM, sequence);
LA = CardanSegAngles(R_LegTM, sequence);
TA = CardanSegAngles(R_ThighTM, sequence);
PA = CardanSegAngles(PelvisTM, sequence);

%Cardan Joint Angles (414-417)
MTPA = CardanJointAngles(R_RearfootTM, R_ForefootTM, sequence);
AA = CardanJointAngles(R_LegTM, R_FootTM, sequence);
KA = CardanJointAngles(R_ThighTM, R_LegTM, sequence);
HA = CardanJointAngles(PelvisTM, R_ThighTM, sequence);

%Get PatellaCS from knee angle (420-421)
TP_TM = SIMMpatellaCS(KA(:,3),r_scaleTibia); %patellar CS
PatellaTM = multiprod(TP_TM,R_ThighTM,[2 3],[2 3]);

%Joint Angles in Radians (424-427)
MTPA_Rad = deg2rad(MTPA);
AA_Rad = deg2rad(AA);
KA_Rad = deg2rad(KA);
HA_Rad = deg2rad(HA);

%Joint Angular Velocity in Radians/s (436-439)
MTPV_Rad = filterdata1(FirstCentral(MTPA_Rad,VideoFrameRate),VideoFrameRate,0,12,4,0);
AV_Rad = filterdata1(FirstCentral(AA_Rad,VideoFrameRate),VideoFrameRate,0,12,4,0);
KV_Rad = filterdata1(FirstCentral(KA_Rad,VideoFrameRate),VideoFrameRate,0,12,4,0);
HV_Rad = filterdata1(FirstCentral(HA_Rad,VideoFrameRate),VideoFrameRate,0,12,4,0);

%Global Angular Accelerations (442-447)
[FFV, aAngforefoot] = RoomSegAng(R_ForefootTM, VideoFrameRate);
[RFV, aAngrearfoot] = RoomSegAng(R_RearfootTM, VideoFrameRate);
[FV, aAngfoot] = RoomSegAng(R_FootTM, VideoFrameRate);
[LV, aAngleg] = RoomSegAng(R_LegTM, VideoFrameRate);
[TV, aAngthigh] = RoomSegAng(R_ThighTM, VideoFrameRate);
[PV, aAngpelvis] = RoomSegAng(PelvisTM, VideoFrameRate);

%If COP is proximal to mtp joint ... (451-464)
COPfilt = filterdata1(COP,VideoFrameRate,0,12,4,0);
COPfilt = [COPfilt; zeros(length(COP)-length(COPfilt),3)];

COPmtpdist = (((MTP5Head(:,1)-MTP1Head(:,1)).*COPfilt(:,3))-((MTP5Head(:,3)-MTP1Head(:,3)).*COPfilt(:,1))+(MTP5Head(:,3).*MTP1Head(:,1))-(MTP5Head(:,1).*MTP1Head(:,3)))./sqrt((MTP5Head(:,1)-MTP1Head(:,1)).^2+(MTP5Head(:,3)-MTP1Head(:,3)).^2);
for b = 1:length(COPmtpdist)
    if COPmtpdist(b) > 0
        [GlobalFankle(b,:), GlobalMankle(b,:)] = inv3d(R_CMfoot(b,:), R_Jankle(b,:), COP(b,:), Rd(b,:), Md(b,:), aR_CMfoot(b,:), aAngfoot(b,:), mfoot, IfootG(b,:,:), g);
        GlobalFmtp(b,:) = [0 0 0];
        GlobalMmtp(b,:) = [0 0 0];
    else
        [GlobalFmtp(b,:), GlobalMmtp(b,:)] = inv3d(R_CMforefoot(b,:), R_Jmtp(b,:), COP(b,:), Rd(b,:), Md(b,:), aR_CMforefoot(b,:), aAngforefoot(b,:), mforefoot, IforefootG(b,:,:), g);
        [GlobalFankle(b,:), GlobalMankle(b,:)] = inv3d(R_CMrearfoot(b,:), R_Jankle(b,:), R_Jmtp(b,:), -GlobalFmtp(b,:), -GlobalMmtp(b,:), aR_CMrearfoot(b,:), aAngrearfoot(b,:), mrearfoot, IrearfootG(b,:,:), g);
    end
end

%Knee Forces and Moments (467)
[GlobalFknee, GlobalMknee] = inv3d(R_CMleg, R_Jknee, R_Jankle, -GlobalFankle, -GlobalMankle, aR_CMleg, aAngleg, mleg, IlegG, g);

%Hip Forces and Moments (470)
[GlobalFhip, GlobalMhip] = inv3d(R_CMthigh, R_Jhip, R_Jknee, -GlobalFknee, -GlobalMknee, aR_CMthigh, aAngthigh, mthigh, IthighG, g);

%Transform from global to local moments (473-478)
LocalMmtp = G2L(R_RearfootTM, GlobalMmtp);
LocalMmtp(isnan(LocalMmtp),:) = 0;
LocalMankle = G2L(R_LegTM, GlobalMankle);
LocalMknee = G2L(R_ThighTM, GlobalMknee);
LocalMhip = G2L(PelvisTM, GlobalMhip);
LocalMhip(isinf(LocalMhip)) = 0;

%Transform from global to local reaction forces (481-484)
LocalFmtp = G2L(R_RearfootTM, GlobalFmtp);
LocalFankle = G2L(R_LegTM, GlobalFankle);
LocalFknee = G2L(R_ThighTM, GlobalFknee);
LocalFhip = G2L(PelvisTM, GlobalFhip);

%Joint Powers (487-490)
RMTP_P = filterdata1(LocalMmtp(:,3).*MTPV_Rad(:,3),VideoFrameRate,0,12,4,0);
RA_P = filterdata1(LocalMankle(:,3).*AV_Rad(:,3),VideoFrameRate,0,12,4,0);
RK_P = filterdata1(LocalMknee(:,3).*KV_Rad(:,3),VideoFrameRate,0,12,4,0);
RH_P = filterdata1(LocalMhip(:,3).*HV_Rad(:,3),VideoFrameRate,0,12,4,0);

%---- per-stride loop: the parts that produce exported kinetics -------------
trials = 18;
% Labelling bad trials — subject 1 only (lines 508-524)
shcond = {'AM','AG','AH'};
if strcmp(shcond{c},'AM') && strcmp(spcode,'RP')
    trialsNaN= 8;
elseif strcmp(shcond{c},'AM') && strcmp(spcode,'PI')
    trialsNaN= [1 7 8 9 10 11 12 13 14 15 16 17 18];
elseif strcmp(shcond{c},'AG') && strcmp(spcode,'RP')
    trialsNaN= [5 6 8];
elseif strcmp(shcond{c},'AG') && strcmp(spcode,'PI')
    trialsNaN= [2 3 4 5 8 16];
elseif strcmp(shcond{c},'AH') && strcmp(spcode,'RP')
    trialsNaN= [5 8];
elseif strcmp(shcond{c},'AH') && strcmp(spcode,'PI')
    trialsNaN= 5;
else
    trialsNaN= [];
end

nsteps = min(trials, length(HS)-1);
percent_new = (0:1:100);
Fper = 0.90; Mper = 0.90;
step_ankle_Mloc = nan(nsteps,101,3); step_knee_Mloc = step_ankle_Mloc; step_hip_Mloc = step_ankle_Mloc; step_mtp_Mloc = step_ankle_Mloc;
step_ankle_Mglob = step_ankle_Mloc; step_knee_Mglob = step_ankle_Mloc; step_hip_Mglob = step_ankle_Mloc;
step_ankle_Floc = step_ankle_Mloc; step_knee_Floc = step_ankle_Mloc; step_hip_Floc = step_ankle_Mloc;
step_AA = step_ankle_Mloc; step_KA = step_ankle_Mloc; step_HA = step_ankle_Mloc; step_MTPA = step_ankle_Mloc;
step_A_P = nan(nsteps,101); step_K_P = step_A_P; step_H_P = step_A_P; step_MTP_P = step_A_P; step_Rd = step_ankle_Mloc;
step_bad = false(nsteps,1); step_contact = nan(nsteps,1); step_flight = step_contact; step_stride = step_contact;
PeakMankle = nan(nsteps,3); PeakMknee = PeakMankle; PeakMhip = PeakMankle; PeakMmtp = PeakMankle;
for i = 1:nsteps
    step_bad(i) = ismember(i,trialsNaN);
    time = vtime(HS(i):TO(i));
    trialInds = HS(i):TO(i);
    ContactT=(TO(i)-HS(i))/VideoFrameRate;
    FlightT = (HS(i+1)-TO(i)-length(trialInds))/2/VideoFrameRate;
    StrideT = ((HS(i+1)-HS(i))/VideoFrameRate);
    step_contact(i)=ContactT; step_flight(i)=FlightT; step_stride(i)=StrideT;
    ind_stance = 1:(TO(i)-HS(i))+1;
    sMTPA = MTPA(HS(i):TO(i),:); sAA = AA(HS(i):TO(i),:); sKA = KA(HS(i):TO(i),:); sHA = HA(HS(i):TO(i),:);
    sRd = Rd(HS(i):TO(i),:);

    %MTP JOINT (743-751)
    Fcutoff = 12; Mcutoff = 12;
    sLocalFmtp = filterdata(LocalFmtp(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
    while length(sLocalFmtp) < length(ind_stance) sLocalFmtp = [sLocalFmtp;sLocalFmtp(end,:)]; end
    sLocalMmtp = filterdata(LocalMmtp(HS(i):TO(i),:), VideoFrameRate,0,Mcutoff,4,0);
    while length(sLocalMmtp) < length(ind_stance) sLocalMmtp = [sLocalMmtp;sLocalMmtp(end,:)]; end
    sGlobalFmtp = filterdata(GlobalFmtp(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
    sGlobalMmtp = filterdata(GlobalMmtp(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);

    %ANKLE JOINT (754-765)
    [PSD, freq] = MyPSD(sRd(ind_stance,2),VideoFrameRate);
    PSDcum=cumsum(PSD);
    PSDcum=PSDcum./PSDcum(end);
    Fcutoff=interp1(PSDcum,freq,Fper);
    Mcutoff=interp1(PSDcum,freq,Mper);
    Fcutoff = 12;
    Mcutoff = 12;
    sLocalFankle = filterdata(LocalFankle(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
    sLocalMankle = filterdata(LocalMankle(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);
    sGlobalFankle = filterdata(GlobalFankle(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
    sGlobalMankle = filterdata(GlobalMankle(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);

    %KNEE JOINT (768-778)
    Fcutoff = 12; Mcutoff = 12;
    sLocalFknee = filterdata(LocalFknee(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
    sLocalMknee = filterdata(LocalMknee(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);
    sGlobalFknee = filterdata(GlobalFknee(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
    sGlobalMknee = filterdata(GlobalMknee(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);

    %HIP JOINT (781-792)
    Fcutoff = 12; Mcutoff = 12;
    sLocalFhip = filterdata(LocalFhip(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
    sLocalMhip = filterdata(LocalMhip(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);
    sGlobalFhip = filterdata(GlobalFhip(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
    sGlobalMhip = filterdata(GlobalMhip(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);

    %Joint Powers (808-811)
    sMTP_P = RMTP_P(HS(i):TO(i)); sA_P = RA_P(HS(i):TO(i)); sK_P = RK_P(HS(i):TO(i)); sH_P = RH_P(HS(i):TO(i));

    %Interpolate data (942-989)
    percent_old = (0:1/VideoFrameRate:(length(time)-1)/VideoFrameRate);
    percent_old = (percent_old/percent_old(end))*100;
    step_MTPA(i,:,:) = interp1(percent_old,sMTPA,percent_new,'pchip');
    step_AA(i,:,:) = interp1(percent_old,sAA,percent_new,'pchip');
    step_KA(i,:,:) = interp1(percent_old,sKA,percent_new,'pchip');
    step_HA(i,:,:) = interp1(percent_old,sHA,percent_new,'pchip');
    step_ankle_Floc(i,:,:) = interp1(percent_old,sLocalFankle,percent_new,'pchip');
    step_knee_Floc(i,:,:) = interp1(percent_old,sLocalFknee,percent_new,'pchip');
    step_hip_Floc(i,:,:) = interp1(percent_old,sLocalFhip,percent_new,'pchip');
    step_mtp_Mloc(i,:,:) = interp1(percent_old,sLocalMmtp,percent_new,'pchip');
    step_ankle_Mloc(i,:,:) = interp1(percent_old,sLocalMankle,percent_new,'pchip');
    step_knee_Mloc(i,:,:) = interp1(percent_old,sLocalMknee,percent_new,'pchip');
    step_hip_Mloc(i,:,:) = interp1(percent_old,sLocalMhip,percent_new,'pchip');
    step_ankle_Mglob(i,:,:) = interp1(percent_old,sGlobalMankle,percent_new,'pchip');
    step_knee_Mglob(i,:,:) = interp1(percent_old,sGlobalMknee,percent_new,'pchip');
    step_hip_Mglob(i,:,:) = interp1(percent_old,sGlobalMhip,percent_new,'pchip');
    step_Rd(i,:,:) = interp1(percent_old,sRd,percent_new);
    step_MTP_P(i,:) = interp1(percent_old,sMTP_P,percent_new,'pchip');
    step_A_P(i,:) = interp1(percent_old,sA_P,percent_new,'pchip');
    step_K_P(i,:) = interp1(percent_old,sK_P,percent_new,'pchip');
    step_H_P(i,:) = interp1(percent_old,sH_P,percent_new,'pchip');
    PeakMmtp(i,:) = min(squeeze(step_mtp_Mloc(i,:,:)));
    PeakMankle(i,:) = min(squeeze(step_ankle_Mloc(i,:,:)));
    PeakMknee(i,:) = max(squeeze(step_knee_Mloc(i,:,:)));
    PeakMhip(i,:) = min(squeeze(step_hip_Mloc(i,:,:)));
end

matlab_version = version;
generated = datestr(now,'yyyy-mm-dd HH:MM:SS');
save(outMat, '-v7', 'A', 'BW', 'plate', 'HS', 'TO', 'Rd', 'Md', 'COP', 'vtime', ...
    'mfoot', 'mleg', 'mthigh', 'mrearfoot', 'mforefoot', ...
    'R_Jmtp', 'R_Jankle', 'R_Jknee', 'R_Jhip', 'JL5S1', ...
    'R_FootRes', 'R_LegRes', 'R_ThighRes', 'R_PelvisRes', ...
    'R_FootTM', 'R_LegTM', 'R_ThighTM', 'PelvisTM', ...
    'aAngfoot', 'aAngleg', 'aAngthigh', 'COPmtpdist', ...
    'GlobalFmtp', 'GlobalMmtp', 'GlobalFankle', 'GlobalMankle', 'GlobalFknee', 'GlobalMknee', 'GlobalFhip', 'GlobalMhip', ...
    'LocalMmtp', 'LocalMankle', 'LocalMknee', 'LocalMhip', 'LocalFankle', 'LocalFknee', 'LocalFhip', ...
    'MTPA', 'AA', 'KA', 'HA', 'RMTP_P', 'RA_P', 'RK_P', 'RH_P', ...
    'step_*', 'PeakM*', 'trialsNaN', 'matlab_version', 'generated');
disp(['saved ' outMat]);
end
