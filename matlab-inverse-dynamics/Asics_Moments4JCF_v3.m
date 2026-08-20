%This program takes the motion capture data preps it for TimSimm_V1

%Written by:
%Brent Edwards (Winter 2015)

%start with clean slate
clear all
% close all

%specify directory, filename, file type
mainDir = 'Z:\Individual Research Projects\Maria Rodriguez\AFT Joint Loading\';
prefix = 'S';
ext = '.c3d';
anthro_file='Subject Anthropometrics.xlsx';

%Set up constancts
subj = [1 2 4 7 8 9 10 11 12 13 15 16 17 18 19 20 21 23 24 25]; %subject numbers
% subj = [15 16 17 18 19 20 21 23 24 25];
shcond = [{'AM'},{'AG'},{'AH'}]; %Shoe conditions (AG = Asics Gel-Cumulus, AH = Asics Hyperspeed, AM = Asics Metaspeed Sky Tokyo, NS = Nike Structure, NI = Nike Invincible, NA = Nike Alphafly 3)
% shcond = {'AM'};
spcond = [{'RP'},{'PI'}]; %Speed conditions (RP = race pace, PI = percent improved pace)
% spcond = {'RP'};

doOutputScaleFile=1;%if you want to output scale file
doOutputSIMMTrials=1;%if you want to output files for simm
doOutputTimSIMMTrials=1; %data by trial for use with TimSIMM
dofigs_stride = 0;%if you want to individual strides
dofigs_trial = 1;%if you want to plot all strides in trial
doCSplot = 0;
doStickFig = 0;
stickview= [0 0 1];
g=[0 -9.81 0];
origin = [0 0 0];
RMmocap = [0 -1 0; 0 0 1; -1 0 0];
RMfp = [0 -1 0; 0 0 1; -1 0 0];
az0 = -0.000001; %mm
offsetFP2 = [0.371453 1.040980 -0.0464284];
offsetFP1 = [0.872614 1.04162 -0.0477233];
VideoFrameRate = 300;
sf_fp = 3000;
Fper = 0.90; %percentile cutoffs for force filtering
Mper = 0.90; %percentile cutoffs for moment filtering
standard_mass=82.7; %from Arnold, 2010
cutoff_moments = 14;
shoeHeight = [0.03 0.02 0.04]; %m
alphabet = 'ABCDEFGHIJKLM';

% Contact event and baseline correction settings
event_force_threshold_n = 20; % N threshold used to trim HS/TO edges
min_off_samples = 20; % minimum off-ground samples for event indexing

% Dynamic moment filtering settings
moment_fc_init_hz = 6; % heavy filtering cutoff for first 20% of stance
moment_fc_midstance_hz = 12; % lighter filtering cutoff for remaining stance
moment_filter_order = 4;
moment_blend_frac = 0.05; % blending window around the 20% transition

for s = 1:length(subj) %subject loop

    %Setting up filenames
    subjDir = [prefix num2str(subj(s)) '\'];
    timsimmDir = [subjDir 'TimSIMM Files\'];
    if ~exist([mainDir timsimmDir],'dir')
        mkdir([mainDir timsimmDir]);
    end

    for c = 1:length(shcond) %shoe condition loop

        %first read in static data to create virtual markers
        filenameStatic = [prefix num2str(subj(s)) '_' shcond{c} '_Static'];
        %read in .c3d and parse out parameters
        % [Markers,VideoFrameRate,AnalogSignals,AnalogFrameRate,Event,ParameterGroup]=readC3D([mainDir stancename ext]);
        % [Markers,VideoFrameRate,AnalogSignals,AnalogFrameRate,Event,ParameterGroup]=readC3D([mainDir subjDir filenameStatic ext]);
        [dataStatic,dataStaticAnalog] = ezc3dRead([mainDir subjDir filenameStatic ext]);
        Markers = dataStatic.data.points;
        VideoFrameRate = dataStatic.parameters.POINT.RATE.DATA;
        AnalogSignals = [dataStaticAnalog(1).force',dataStaticAnalog(1).moment',dataStaticAnalog(2).force',dataStaticAnalog(2).moment'];
        AnalogFrameRate = dataStatic.parameters.ANALOG.RATE.DATA;

        Markers(find(Markers==0))=NaN;%no markers should be zero, replace with NaN (means gap exists in data)

        %extract individual marker data (also divide by 1000 to put in units (m)
        % GroupNumber=3;
        % ParameterNumber=11;
        for p=1:size(Markers,2)
            mname=dataStatic.parameters.POINT.LABELS.DATA(p);
            if strcmp('*',mname{1}(1))
                break
            else
                evalstr=[cell2mat(mname) '=squeeze(Markers(:,' num2str(p) ',:)./1000)'';'];
                markernames(p)=mname;
                eval(evalstr);
            end
        end

        % Transform  marker coordinates to SIMM Coordinate system
        for p=1:size(markernames,2)
            evalstr=[cell2mat(markernames(p)) '=NewCS(origin,RMmocap,nanmean(' cell2mat(markernames(p)) '));'];
            eval(evalstr);
        end

        %Get Anthopometrics
        A(1:15)=0;
        if exist([mainDir anthro_file])%there is an anthro file
            anthro = readmatrix([mainDir anthro_file],'FileType','spreadsheet','Range','A2:M27');
            A(1)=anthro(subj(s),5); %Total Mass (kg)
            A(3)=anthro(subj(s),7);% Midthigh Circumference (m)
            A(5)=anthro(subj(s),6);% Calf Circumference (m)
            A(6)=anthro(subj(s),9);% Foot Length (m)
            A(9)=anthro(subj(s),8);% Foot Breadth (m)
            A(15)=anthro(subj(s),4);% Height (m)
            speed_RP= anthro(subj(s),10);%Race pace speed (m/s)
            speed_PI = anthro(subj(s),11);%AFT_improved speed (m/s)
        else %use defaults
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
            Attention = ('Default anthropometrics used')
        end

        %Define body weight
        BW=A(1)*9.81;

        %Define stance joint centers and virtual points
        N_L5S1center=(0.75*((0.5*LPSIS)+(0.5*RPSIS)))+(0.25*((0.5*LASIS)+(0.5*RASIS)));
        Nr_hipcenter = (0.70*RGT)+(0.30*LGT);
        Nr_kneecenter = (0.50*KneeLat)+(0.50*KneeMed);
        Nr_anklecenter = (0.50*AnkleLat)+(0.50*AnkleMed);
        Nr_metcenter = (0.50*MTP1Head) + (0.50*MTP5Head);

        %Define thigh length, calf length, malleolus height and width
        A(2) = sqrt(sum((Nr_hipcenter-Nr_kneecenter).^2)); %Thigh length (m)
        A(4) = sqrt(sum((Nr_kneecenter-Nr_anklecenter).^2)); %Calf length (m)
        A(7) = Nr_anklecenter(2)-shoeHeight(c)-0.0045;% Malleolus Height (m)
        A(8) = AnkleLat(3)-AnkleMed(3)-0.009;% Malleolus Width (m)

        %Define leg length
        legLength = A(2)+A(4)+A(7);

        %Define stance marker clusters
        N_pelvis = [RASIS LASIS RPSIS LPSIS RGT LGT];%neutral pelvis marker cluster
        Nr_thigh = [ThighAntProx ThighPostDist ThighPostProx];%neutral thigh marker cluster
        Nr_leg=[ShankAntProx ShankPostDist ShankPostProx];%neutral leg marker cluster
        Nr_foot=[HeelLat HeelMed DorsalFoot];%neutral foot marker cluster
        Nr_rearfoot = [HeelLat HeelMed DorsalFoot]; %neutral rearfoot marker cluster
        Nr_forefoot = [MTP1Head MTP5Head ToeboxAnt]; %neutral forefoot marker cluster

        %Define/create a virtual toe marker
        Nr_footlength = ToeboxAnt(1) - HeelMed(1) ;%footlength + marker to heel distance
        
        %Define static MTP1head
        N_MTP1Head = MTP1Head;
        % footvec=unitvec(DFoot-Heel);
        % Nr_toe=Heel+Nr_footlength*footvec;
        % Nr_toe(2)=Heel(2);

        %Segment Masses, From Vaughan, Dynamics of Human Gait
        m=(0.0083*A(1))+(254.5*A(6)*A(7)*A(8))-0.065;mfoot=[m m m];
        m=(0.0223*A(1))+(31.33*A(4)*A(5)^2)-0.016;mleg=[m m m];
        m=(0.1032*A(1))+(12.76*A(2)*A(3)^2)-1.023;mthigh=[m m m];
        m=(0.1032*A(1))+(12.76*A(2)*A(3)^2)-1.023;mpelvis=[m m m];

        %mass of hindfoot/forefoot are functions of segment x-length
        mrearfoot = (Nr_metcenter(1)-HeelMed(1))/(ToeboxAnt(1)-HeelMed(1))*mfoot;
        mforefoot = (ToeboxAnt(1)-Nr_metcenter(1))/(ToeboxAnt(1)-HeelMed(1))*mfoot;

        %Local segment moments of inertia for the rearfoot & forefoot, From Zatsiorsky, Kinetics of Human Motion page 594-597 (x = abd/add, y = int/ext, z = flex/ext)
        Irfx = (1/12)*mrearfoot(1)*(3*((MTP5Head(3)-MTP1Head(3))*0.5)^2+((Nr_metcenter(1)-HeelMed(1)))^2);
        Irfy = (1/12)*mrearfoot(1)*(3*(Nr_anklecenter(2)*0.5)^2+((Nr_metcenter(1)-HeelMed(1)))^2);
        Irfz = (1/4)*mrearfoot(1)*(((MTP5Head(3)-MTP1Head(3))*0.5)^2+3*(Nr_anklecenter(2)*0.5)^2);
        Iffx = (1/12)*mforefoot(1)*(3*((MTP5Head(3)-MTP1Head(3))*0.5)^2+((ToeboxAnt(1)-Nr_metcenter(1)))^2);
        Iffy = (1/12)*mforefoot(1)*(3*(ToeboxAnt(2)*0.5)^2+((ToeboxAnt(1)-Nr_metcenter(1)))^2);
        Iffz = (1/4)*mforefoot(1)*(((MTP5Head(3)-MTP1Head(3))*0.5)^2+(ToeboxAnt(2)*0.5)^2);

        %makes virtual marker for Sacrum with ASIS vertical coordinate
        N_mid_ASIS =(.50*RASIS)+(.50*LASIS);
        Sacrum = (0.50*RPSIS)+(0.50*LPSIS);
        N_Sacrum_ASISy=[Sacrum(1) N_mid_ASIS(2) Sacrum(3)];

        %Calculate segment lengths
        seg.R.thigh.length=norm(Nr_hipcenter-Nr_kneecenter);
        seg.R.leg.length=norm(Nr_kneecenter-Nr_anklecenter);
        seg.R.foot.length=norm(Nr_footlength);

        if doOutputScaleFile
            %SIMM scaling factors to ouput SIMM scaling file
            SIMMmass=standard_mass; %height = 168.4 cm
            SIMMTorsolength = 0.5246;  %ArnoldHamner v2.1
            SIMMASISwidth = 0.243;
            SIMMHipwidth = 0.318;
            SIMMFemurlength = 0.434-0.017;%SIMM greater trochanter to tibia-femur contact
            SIMMTibialength = 0.439;  %SIMM tibia-femur contact to lateral maleolus
            SIMMFootlength = 0.2607;%
            Hipwidth = pdist([RGT;LGT])-(0.014);
            ASISwidth = (SIMMASISwidth/SIMMHipwidth)*Hipwidth;
            scaleMass=A(1)/SIMMmass;
            scaleASIS=ASISwidth/SIMMASISwidth;
            scalePelvis =scaleASIS ;
            r_scaleFemur = seg.R.thigh.length/SIMMFemurlength;
            r_scaleTibia = seg.R.leg.length/SIMMTibialength;
            r_scaleFoot = seg.R.foot.length/SIMMFootlength;

            %Output scale factors to a scale file
            outfile = [mainDir timsimmDir 'S' num2str(subj(s)) '.txt'] ;
            dlmwrite(outfile,['pelvis ' num2str(scaleASIS) ' ' num2str(scaleASIS) ' ' num2str(scaleASIS) ], 'delimiter','', 'newline', 'pc');
            dlmwrite(outfile,['femur ' num2str(r_scaleFemur) ' ' num2str(r_scaleFemur) ' ' num2str(r_scaleFemur)  ],'-append', 'delimiter',  '', 'newline', 'pc');
            dlmwrite(outfile,['tibia ' num2str(r_scaleTibia) ' ' num2str(r_scaleTibia) ' ' num2str(r_scaleTibia)  ],'-append', 'delimiter',  '', 'newline', 'pc');
            dlmwrite(outfile,['foot ' num2str(r_scaleFoot) ' ' num2str(r_scaleFoot) ' ' num2str(r_scaleFoot)  ],'-append', 'delimiter',  '', 'newline', 'pc');
            dlmwrite(outfile,['mass ' num2str(scaleMass)],'-append', 'delimiter',  '', 'newline', 'pc');
        end

        for t = 1:length(spcond) %Speed loop
            disp(['S' num2str(subj(s)) ' ' shcond{c} ' ' spcond{t}])
            filenameDynamic = [prefix num2str(subj(s)) '_' shcond{c}  '_' spcond{t}];

            outCol = 2;

            %Set speed to either race pace or AFT-improved
            if strcmp(spcond{t},'RP')
                speed = speed_RP;
            elseif strcmp(spcond{t},'PI')
                speed = speed_PI;
            end

            if exist([mainDir subjDir filenameDynamic ext]) %only does analysis if dynamic trial exists

                %Read in C3D file
                % [Markers,VideoFrameRate,AnalogSignals,AnalogFrameRate,Event,ParameterGroup]=readC3D([mainDir subjDir filenameDynamic ext]);
                %Markers(find(Markers==0))=NaN;%no markers should be zero, replace with NaN (means gap exists in data)
                [dataDynamic,dataDynamicAnalog] = ezc3dRead([mainDir subjDir filenameDynamic ext]);
                Markers = dataDynamic.data.points;
                VideoFrameRate = dataDynamic.parameters.POINT.RATE.DATA;
                AnalogFrameRate = dataDynamic.parameters.ANALOG.RATE.DATA;
                AnalogSignals = [dataDynamicAnalog(1).force',dataDynamicAnalog(1).moment',dataDynamicAnalog(2).force',dataDynamicAnalog(2).moment'];
                % analogSt = 10*AnalogFrameRate/VideoFrameRate;
                % analogEnd = length(AnalogSignals)-analogSt+1;
                % AnalogSignals = AnalogSignals(analogSt:analogEnd,:);

                clear markernames;

                %extract individual marker data (also divide by 1000 to put in units (m)
                GroupNumber=3;ParameterNumber=11;
                for p=1:size(Markers,2)
                    mname=dataDynamic.parameters.POINT.LABELS.DATA(p);
                    if ~strcmp('*',mname{1}(1))
                        evalstr=[cell2mat(mname) '=squeeze(Markers(:,' num2str(p) ',:)./1000)'';'];
                        markernames(p)=mname;
                        eval(evalstr);
                    end
                end

                % Transform  marker coordinates to SIMM Coordinate system
                for p=1:size(markernames,2)
                    evalstr=[cell2mat(markernames(p)) '=NewCS(origin,RMmocap,' cell2mat(markernames(p)) ');'];
                    eval(evalstr);
                end

                %Define treadmill slope
                slope = 0;

                %Take care of any potential offsets in forceplateform
                % AnalogSignals = TreadOffsets(AnalogSignals);

                %interpolate GRF to be same as motion
                vtime = (0:1/VideoFrameRate:(length(Markers)-1)/VideoFrameRate)';
                atime = (0:1/AnalogFrameRate:(length(AnalogSignals)-1)/AnalogFrameRate)';
                intAnalogSignals = interp1(atime,AnalogSignals,vtime,'pchip');

                %Extract Ground reaction force data, filter it at 18 Hz to
                %get rid of vibration from Motek, and output rotated GRF
                %(Rd), rotated moment (Md), and rotated COP data
                FP_offsets = [offsetFP1;offsetFP2];
                [Rd, Md, COP, plate] = getViconForces_Motek(intAnalogSignals, VideoFrameRate, RMfp, origin, FP_offsets, slope);

                %Initial event detection
                [HS, TO, ind_off] = ContactEvents_Motek(Rd(:,2),COP(:,3),VideoFrameRate);

                %Tighten HS/TO boundaries to avoid low-force transition samples.
                for ev = 1:length(HS)
                    idx = HS(ev):TO(ev);
                    in_contact = find(Rd(idx,2) > event_force_threshold_n,1,'first');
                    out_contact = find(Rd(idx,2) > event_force_threshold_n,1,'last');
                    if ~isempty(in_contact) && ~isempty(out_contact)
                        HS(ev) = idx(in_contact);
                        TO(ev) = idx(out_contact);
                    end
                end

                %Rebuild off-ground indices from refined right-foot events.
                if ~isempty(HS)
                    ind_off = [(1:HS(1))'; (TO(end):length(Rd))'];
                    for ev = 1:length(TO)-1
                        ind_off = [ind_off; (TO(ev):HS(ev+1))'];
                    end
                    ind_off = unique(ind_off);
                else
                    ind_off = (1:length(Rd))';
                end

                %Remove GRF data for right-foot off-ground periods
                Rd(ind_off,:)=0;
                Md(ind_off,:)=0;
                COP(ind_off,:)=[DorsalFoot(ind_off,1) zeros(length(ind_off),1) DorsalFoot(ind_off,3)];

                %generate grf.mot file for opensimm
                %                 GRFMatrix = [vtime Rd COP zeros(length(Rd),6) Md zeros(length(Rd),3)];
                %                 colnames = {'time' 'ground_force_vx' 'ground_force_vy' 'ground_force_vz' 'ground_force_px' 'ground_force_py' 'ground_force_pz' 'l_ground_force_vx'...
                %                     'l_ground_force_vy' 'l_ground_force_vz' 'l_ground_force_px' 'l_ground_force_py' 'l_1ground_force_pz' 'ground_torque_x' 'ground_torque_y'...
                %                     'ground_torque_z' 'l_ground_torque_x' 'l_ground_torque_y' 'l_ground_torque_z'};
                %                 generateMotFile(GRFMatrix, colnames, [dir filename '.mot']);

                %Define dynamic marker clusters
                pelvis = [RASIS LASIS RPSIS LPSIS RGT LGT];%pelvis marker cluster
                r_thigh = [ThighAntProx ThighPostDist ThighPostProx];%thigh marker cluster
                r_leg=[ShankAntProx ShankPostDist ShankPostProx];%leg marker cluster
                r_foot=[HeelLat HeelMed DorsalFoot];%foot marker cluster
                r_rearfoot = [HeelLat HeelMed DorsalFoot]; %neutral rearfoot marker cluster
                r_forefoot = [MTP1Head MTP5Head ToeboxAnt]; %neutral forefoot marker cluster

                %Calculate virtual points
                % [R_VPtoe] = VirtualPoint_v2(Nr_foot,Nr_toe,r_foot);
                [R_Jmtp,R_ForefootTM,~] = VirtualPoint_v2(Nr_forefoot,Nr_metcenter,r_forefoot);
                [~, R_RearfootTM,~] = VirtualPoint_v2(Nr_rearfoot,Nr_anklecenter,r_rearfoot);
                [R_Jankle, R_FootTM, R_FootRes] = VirtualPoint_v2(Nr_foot,Nr_anklecenter,r_foot);
                [R_Jknee, R_LegTM, R_LegRes] = VirtualPoint_v2(Nr_leg,Nr_kneecenter,r_leg);
                [R_Jhip, R_ThighTM, R_ThighRes] = VirtualPoint_v2(Nr_thigh,Nr_hipcenter,r_thigh);
                [JL5S1, PelvisTM, R_PelvisRes] = VirtualPoint_v2(N_pelvis,N_L5S1center,pelvis);
                [VPSacrum_ASISy] = VirtualPoint_v2(N_pelvis,N_Sacrum_ASISy,pelvis);
                [VPmid_ASIS] = VirtualPoint_v2(N_pelvis,N_mid_ASIS,pelvis);
                mid_Hip=(0.5*RGT)+(0.5*LGT);

                %Local segment moments of inertia, From Vaughan, Dynamics of Human Gait (needs to
                %be global eventually).
                Ifx=0.00021*A(1)*(4*A(9)^2+3*A(6)^2)+0.00067;
                Ifz=(0.00023*A(1))*(4*A(7)^2+3*A(6)^2)+0.00022;
                Ify=0.00141*A(1)*(A(7)^2+A(9)^2)-0.00008;
                Ilx=0.00387*A(1)*(A(4)^2+.076*A(5)^2)+0.00138;
                Ilz=(0.00347*A(1))*(A(4)^2+0.076*A(5)^2)+0.00511;
                Ily=0.00041*A(1)*A(9)^2+0.00012;
                Itx=0.00726*A(1)*(A(2)^2+0.076*A(3)^2)+0.01186;
                Itz=(0.00762 * A(1)) * (A(2)^2 + 0.076 * A(3)^2) + 0.01153;
                Ity= 0.00151*A(1)*A(6)^2+0.00305;

                %Local segment moments of inertia for pelvis,, From Zatsiorsky-Seluyanov
                Ipx=-1568+(12*A(1))+(7.741*A(15)*100);
                Ipz=-934+(11.8*A(1))+(3.44*A(15)*100);
                Ipy=-775+(14.7*A(1))+(1.685*A(15)*100);

                Ifoot=[Ifx 0 0;0 Ify 0; 0 0 Ifz];
                Ileg=[Ilx 0 0;0 Ify 0; 0 0 Ilz];
                Ithigh=[ Itx 0 0;0 Ity 0; 0 0 Itz];
                Ipelvis=[Ipx 0 0;0 Ipy 0; 0 0 Ipz]*1e-4;
                Irearfoot = [Irfx 0 0;0 Irfy 0; 0 0 Irfz];
                Iforefoot = [Iffx 0 0;0 Iffy 0; 0 0 Iffz];

                %Calculate global moments of inertia by multiplying local moments of inertia by transformation matrices
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

                %Segment COM locations, From Vaughn
                R_CMfoot=0.44+(0.0195/A(6));%correct percentage for heel to heel marker distance
                R_CMleg=0.42;
                R_CMthigh=0.39;
                CMpelvis=0.50;%just a guess

                %COM Locations
                R_CMrearfoot = HeelMed + 0.5*(R_Jmtp - HeelMed);
                R_CMforefoot = R_Jmtp + 0.5*(ToeboxAnt - R_Jmtp);
                R_CMfoot=HeelLat-(R_CMfoot*(HeelLat-ToeboxAnt));
                R_CMleg=R_Jknee-(R_CMleg*(R_Jknee-R_Jankle));
                R_CMthigh=R_Jhip-(R_CMthigh*(R_Jhip-R_Jknee));
                CMpelvis=JL5S1-(CMpelvis*(JL5S1-mid_Hip));

                %COM Accelerations
                aR_CMrearfoot = FirstCentral(FirstCentral(R_CMrearfoot,VideoFrameRate),VideoFrameRate);
                aR_CMforefoot = FirstCentral(FirstCentral(R_CMforefoot,VideoFrameRate),VideoFrameRate);
                aR_CMfoot=FirstCentral(FirstCentral(R_CMfoot,VideoFrameRate),VideoFrameRate);
                aR_CMleg=FirstCentral(FirstCentral(R_CMleg,VideoFrameRate),VideoFrameRate);
                aR_CMthigh=FirstCentral(FirstCentral(R_CMthigh,VideoFrameRate),VideoFrameRate);
                a_CMpelvis=FirstCentral(FirstCentral(CMpelvis,VideoFrameRate),VideoFrameRate);

                %Cardan Segment Angles
                sequence = 'zxy';
                FFA = CardanSegAngles(R_ForefootTM,sequence);
                RFA = CardanSegAngles(R_RearfootTM, sequence);
                FA = CardanSegAngles(R_FootTM, sequence);
                LA = CardanSegAngles(R_LegTM, sequence);
                TA = CardanSegAngles(R_ThighTM, sequence);
                PA = CardanSegAngles(PelvisTM, sequence);

                %Cardan Joint Angles
                MTPA = CardanJointAngles(R_RearfootTM, R_ForefootTM, sequence);
                AA = CardanJointAngles(R_LegTM, R_FootTM, sequence);
                KA = CardanJointAngles(R_ThighTM, R_LegTM, sequence);
                HA = CardanJointAngles(PelvisTM, R_ThighTM, sequence);

                %Get PatellaCS from knee angle
                TP_TM = SIMMpatellaCS(KA(:,3),r_scaleTibia); %patellar CS
                PatellaTM = multiprod(TP_TM,R_ThighTM,[2 3],[2 3]);

                %Joint Angles in Radians
                MTPA_Rad = deg2rad(MTPA);
                AA_Rad = deg2rad(AA);
                KA_Rad = deg2rad(KA);
                HA_Rad = deg2rad(HA);

                %Joint Angular Velocity
                MTPV = filterdata1(FirstCentral(MTPA,VideoFrameRate),VideoFrameRate,0,12,4,0);
                AV = filterdata1(FirstCentral(AA,VideoFrameRate),VideoFrameRate,0,12,4,0);
                KV = filterdata1(FirstCentral(KA,VideoFrameRate),VideoFrameRate,0,12,4,0);
                HV = filterdata1(FirstCentral(HA,VideoFrameRate),VideoFrameRate,0,12,4,0);

                %Joint Angular Velocity in Radians/s
                MTPV_Rad = filterdata1(FirstCentral(MTPA_Rad,VideoFrameRate),VideoFrameRate,0,12,4,0);
                AV_Rad = filterdata1(FirstCentral(AA_Rad,VideoFrameRate),VideoFrameRate,0,12,4,0);
                KV_Rad = filterdata1(FirstCentral(KA_Rad,VideoFrameRate),VideoFrameRate,0,12,4,0);
                HV_Rad = filterdata1(FirstCentral(HA_Rad,VideoFrameRate),VideoFrameRate,0,12,4,0);

                %Global Angular Accelerations
                [FFV, aAngforefoot] = RoomSegAng(R_ForefootTM, VideoFrameRate);
                [RFV, aAngrearfoot] = RoomSegAng(R_RearfootTM, VideoFrameRate);
                [FV, aAngfoot] = RoomSegAng(R_FootTM, VideoFrameRate);
                [LV, aAngleg] = RoomSegAng(R_LegTM, VideoFrameRate);
                [TV, aAngthigh] = RoomSegAng(R_ThighTM, VideoFrameRate);
                [PV, aAngpelvis] = RoomSegAng(PelvisTM, VideoFrameRate);

                %If COP is proximal to mtp joint, apply Rd & Md to hindfoot. If COP is distal to mtp joint, apply Rd & Md to forefoot and apply mtp forces/moments to hindfoot
                %Filter the shit out of the cop data before usin it to calculate which joint to perform IDyn on.
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

                %Knee Forces and Moments
                [GlobalFknee, GlobalMknee] = inv3d(R_CMleg, R_Jknee, R_Jankle, -GlobalFankle, -GlobalMankle, aR_CMleg, aAngleg, mleg, IlegG, g);

                %Hip Forces and Moments
                [GlobalFhip, GlobalMhip] = inv3d(R_CMthigh, R_Jhip, R_Jknee, -GlobalFknee, -GlobalMknee, aR_CMthigh, aAngthigh, mthigh, IthighG, g);

                %Transform from global to local moments at the distal end of the proximal segment
                LocalMmtp = G2L(R_RearfootTM, GlobalMmtp);
                LocalMmtp(isnan(LocalMmtp),:) = 0;
                LocalMankle = G2L(R_LegTM, GlobalMankle);
                LocalMknee = G2L(R_ThighTM, GlobalMknee);
                LocalMhip = G2L(PelvisTM, GlobalMhip);
                LocalMhip(isinf(LocalMhip)) = 0;

                %Transform from global to local reaction forces
                LocalFmtp = G2L(R_RearfootTM, GlobalFmtp);
                LocalFankle = G2L(R_LegTM, GlobalFankle);
                LocalFknee = G2L(R_ThighTM, GlobalFknee);
                LocalFhip = G2L(PelvisTM, GlobalFhip);

                %Joint Powers
                RMTP_P = filterdata1(LocalMmtp(:,3).*MTPV_Rad(:,3),VideoFrameRate,0,12,4,0);
                RA_P = filterdata1(LocalMankle(:,3).*AV_Rad(:,3),VideoFrameRate,0,12,4,0);
                RK_P = filterdata1(LocalMknee(:,3).*KV_Rad(:,3),VideoFrameRate,0,12,4,0);
                RH_P = filterdata1(LocalMhip(:,3).*HV_Rad(:,3),VideoFrameRate,0,12,4,0);               

                %Create an individual stride loop
                clear trials
                %Select which steps to use based on subject, shoe, and
                %speed (default = steps 1:18)

                if (subj(s) ==5 && strcmp('AG',shcond{c}) && strcmp('RP',spcond{t})) || (subj(s) ==5 && strcmp('AM',shcond{c}) && strcmp('PI',spcond{t}))
                    trials = 2 ;
                else
                    trials = 18;
                end
                n=1;

                %Labelling bad trials
                if subj(s) == 1
                    if strcmp(shcond{c},'AM') && strcmp(spcond{t},'RP')
                        trialsNaN= 8;
                    elseif strcmp(shcond{c},'AM') && strcmp(spcond{t},'PI')
                        trialsNaN= [1 7 8 9 10 11 12 13 14 15 16 17 18];
                    elseif strcmp(shcond{c},'AG') && strcmp(spcond{t},'RP')
                        trialsNaN= [5 6 8];
                    elseif strcmp(shcond{c},'AG') && strcmp(spcond{t},'PI')
                        trialsNaN= [2 3 4 5 8 16];
                    elseif strcmp(shcond{c},'AH') && strcmp(spcond{t},'RP')
                        trialsNaN= [5 8];
                    elseif strcmp(shcond{c},'AH') && strcmp(spcond{t},'PI')
                        trialsNaN= 5;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 2
                    if strcmp(shcond{c},'AM') && strcmp(spcond{t},'RP')
                        trialsNaN= 17;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 3
                    if strcmp(shcond{c},'AG') && strcmp(spcond{t},'PI')
                        trialsNaN= [9 10];
                    elseif strcmp(shcond{c},'AH') && strcmp(spcond{t},'RP')
                        trialsNaN= 13;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 4
                    if strcmp(shcond{c},'AG') && strcmp(spcond{t},'RP')
                        trialsNaN= 4;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 8
                    if strcmp(shcond{c},'AH') && strcmp(spcond{t},'RP')
                        trialsNaN= 7;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 9
                    if strcmp(shcond{c},'AG') && strcmp(spcond{t},'RP')
                        trialsNaN= 4;
                    elseif strcmp(shcond{c},'AH') && strcmp(spcond{t},'PI')
                        trialsNaN = [3 10 11 13];
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 10
                    if strcmp(shcond{c},'AM') && strcmp(spcond{t},'PI')
                        trialsNaN= 8;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 11
                    if strcmp(shcond{c},'AG') && strcmp(spcond{t},'PI')
                        trialsNaN= 8;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 12
                    if strcmp(shcond{c},'AM') && strcmp(spcond{t},'RP')
                        trialsNaN= 4;
                    elseif strcmp(shcond{c},'AG') && strcmp(spcond{t},'PI')
                        trialsNaN= 17;
                    elseif strcmp(shcond{c},'AH') && strcmp(spcond{t},'PI')
                        trialsNaN= [2 18];
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 13
                    if strcmp(shcond{c},'AG') && strcmp(spcond{t},'RP')
                        trialsNaN= 14;
                    elseif strcmp(shcond{c},'AH') && strcmp(spcond{t},'PI')
                        trialsNaN= [13 14];
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 14
                    if strcmp(shcond{c},'AM') && strcmp(spcond{t},'RP')
                        trialsNaN= [7 13];
                    elseif strcmp(shcond{c},'AG') && strcmp(spcond{t},'PI')
                        trialsNaN= 12;
                    elseif strcmp(shcond{c},'AH') && strcmp(spcond{t},'RP')
                        trialsNaN= 14;
                    elseif strcmp(shcond{c},'AH') && strcmp(spcond{t},'PI')
                        trialsNaN= 6;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 15
                    if strcmp(shcond{c},'AM') && strcmp(spcond{t},'RP')
                        trialsNaN= 5;
                    elseif strcmp(shcond{c},'AM') && strcmp(spcond{t},'PI')
                        trialsNaN= 2;
                    elseif strcmp(shcond{c},'AG') && strcmp(spcond{t},'PI')
                        trialsNaN= 16;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 16
                    if strcmp(shcond{c},'AH') && strcmp(spcond{t},'PI')
                        trialsNaN= 15;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 17
                    if strcmp(shcond{c},'AG') && strcmp(spcond{t},'RP')
                        trialsNaN= [14 18];
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 19
                    if strcmp(shcond{c},'AM') && strcmp(spcond{t},'RP')
                        trialsNaN= 13;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 20
                    if strcmp(shcond{c},'AG') && strcmp(spcond{t},'PI')
                        trialsNaN= 4;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 21
                    if strcmp(shcond{c},'AH') && strcmp(spcond{t},'PI')
                        trialsNaN= 5;
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 24
                    if strcmp(shcond{c},'AG') && strcmp(spcond{t},'RP')
                        trialsNaN= [1 14];
                    elseif strcmp(shcond{c},'AG') && strcmp(spcond{t},'PI')
                        trialsNaN= [14 16];
                    else
                        trialsNaN= [];
                    end
                elseif subj(s) == 25
                    if strcmp(shcond{c},'AG') && strcmp(spcond{t},'PI')
                        trialsNaN= [10 15];
                    elseif strcmp(shcond{c},'AH') && strcmp(spcond{t},'PI')
                        trialsNaN= [1 10 16];
                    else
                        trialsNaN= [];
                    end
                else
                    trialsNaN = [];
                end

                nanFlag = 0;

                for i = 1:trials  %First 18 strides

                    if ismember(i,trialsNaN)
                        nanFlag = 1;
                    end

                    disp(['Step ' num2str(i)])
                    time = vtime(HS(i):TO(i));
                    trialInds = HS(i):TO(i);
                    ContactT=(TO(i)-HS(i))/VideoFrameRate;
                    FlightT = (HS(i+1)-TO(i)-length(trialInds))/2/VideoFrameRate;
                    StrideT = ((HS(i+1)-HS(i))/VideoFrameRate);
                    Stride_Length = StrideT.*speed;%stride length when speed = 3.3
                    Step_Length = (Stride_Length)/2; %assumes equal step length between limbs
                    pLeg_StepsPERkm(i,1) = (1000/Stride_Length);%this gives number of strides per KM which is equivalent to single leg steps per km
                    SFreq= (60/StrideT).*2;%Step frequency calculated as twice the stride frequency
                    Fmax = A(1)*9.81*pi()/2*(FlightT/ContactT+1);
                    deltaYc = (Fmax*ContactT^2/A(1)/pi()^2)+(9.81*ContactT^2/8);
                    deltaL = legLength-sqrt(legLength^2-(speed*ContactT/2)^2)+deltaYc;
                    legStiffness = Fmax/deltaL;
                    verticalStiffness = Fmax/deltaYc;

                    All_verticalStiffness(s,c,t,i) = verticalStiffness;
                    All_legStiffness(s,c,t,i) = legStiffness;
                    All_FlightT(s,c,t,i) = FlightT;
                    All_ContactT(s,c,t,i) = ContactT;
                    All_StrideLength(s,c,t,i) = Stride_Length;
                    All_SFreq(s,c,t,i) = SFreq;

                    %stance indices
                    ind_stance = 1:(TO(i)-HS(i))+1;

                    %local kinematic segment and joint angle data and parameters
                    sFFA = FFA(HS(i):TO(i),:);
                    sRFA = RFA(HS(i):TO(i),:);
                    sFA = FA(HS(i):TO(i),:);
                    sLA = LA(HS(i):TO(i),:);
                    sTA = TA(HS(i):TO(i),:);
                    sPA = PA(HS(i):TO(i),:);
                    sMTPA = MTPA(HS(i):TO(i),:);
                    sAA = AA(HS(i):TO(i),:);
                    sKA = KA(HS(i):TO(i),:);
                    sHA = HA(HS(i):TO(i),:);

                    %calculate segment angles at contact
                    contFFA(s,c,t,i,:) = sFFA(1,:);
                    contRFA(s,c,t,i,:) = sRFA(1,:);
                    contFA(s,c,t,i,:) = sFA(1,:);
                    contLA(s,c,t,i,:) = sLA(1,:);
                    contTA(s,c,t,i,:) = sTA(1,:);
                    contPA(s,c,t,i,:) = sPA(1,:);

                    %calculate joint angles at contact
                    All_cMTPA(s,c,t,i,:) = MTPA(HS(i),:);
                    All_cAA(s,c,t,i,:) = AA(HS(i),:);
                    All_cKA(s,c,t,i,:) = KA(HS(i),:);
                    All_cHA(s,c,t,i,:) = HA(HS(i),:);

                    %Foot segment angle at contact
                    All_cFA(s,c,t,i,:)= FA(HS(i),:);

                    %calculate joint angles at toeoff
                    All_tMTPA(s,c,t,i,:) = MTPA(TO(i),:);
                    All_tAA(s,c,t,i,:) = AA(TO(i),:);
                    All_tKA(s,c,t,i,:) = KA(TO(i),:);
                    All_tHA(s,c,t,i,:) = HA(TO(i),:);

                    %calculate segment and joint max angles across stride
                    [maxFFA, maxFFA_ind] = nanmax(sFFA,[],1);maxFFA_ind = maxFFA_ind./length(sFFA).*100;
                    [maxRFA, maxRFA_ind] = nanmax(sRFA,[],1);maxRFA_ind = maxRFA_ind./length(sRFA).*100;
                    [maxFA, maxFA_ind] = nanmax(sFA,[],1);maxFA_ind = maxFA_ind./length(sFA).*100;
                    [maxLA, maxLA_ind] = nanmax(sLA,[],1);maxLA_ind = maxLA_ind./length(sLA).*100;
                    [maxTA, maxTA_ind] = nanmax(sTA,[],1);maxTA_ind = maxTA_ind./length(sTA).*100;
                    [maxPA, maxPA_ind] = nanmax(sPA,[],1);maxPA_ind = maxPA_ind./length(sPA).*100;
                    [maxMTPA, maxMTPA_ind] = nanmax(sMTPA,[],1);maxMTPA_ind = maxMTPA_ind./length(sMTPA).*100;
                    [maxAA, maxAA_ind] = nanmax(sAA,[],1);maxAA_ind = maxAA_ind./length(sAA).*100;
                    [maxKA, maxKA_ind] = nanmax(sKA,[],1);maxKA_ind = maxKA_ind./length(sKA).*100;
                    [maxHA, maxHA_ind] = nanmax(sHA,[],1);maxHA_ind = maxHA_ind./length(sHA).*100;

                    %calculate segment and joint min angles across stride
                    [minFFA, minFFA_ind] = nanmin(sFFA,[],1);minFFA_ind = minFFA_ind./length(sFFA).*100;
                    [minRFA, minRFA_ind] = nanmin(sRFA,[],1);minRFA_ind = minRFA_ind./length(sRFA).*100;
                    [minFA, minFA_ind] = nanmin(sFA,[],1);minFA_ind = minFA_ind./length(sFA).*100;
                    [minLA, minLA_ind] = nanmin(sLA,[],1);minLA_ind = minLA_ind./length(sLA).*100;
                    [minTA, minTA_ind] = nanmin(sTA,[],1);minTA_ind = minTA_ind./length(sTA).*100;
                    [minPA, minPA_ind] = nanmin(sPA,[],1);minPA_ind = minPA_ind./length(sPA).*100;
                    [minMTPA, minMTPA_ind] = nanmin(sMTPA,[],1);minMTPA_ind = minMTPA_ind./length(sMTPA).*100;
                    [minAA, minAA_ind] = nanmin(sAA,[],1);minAA_ind = minAA_ind./length(sAA).*100;
                    [minKA, minKA_ind] = nanmin(sKA,[],1);minKA_ind = minKA_ind./length(sKA).*100;
                    [minHA, minHA_ind] = nanmin(sHA,[],1);minHA_ind = minHA_ind./length(sHA).*100;

                    %calculate segment and joint ranges of motion
                    rangeFA(s,c,t,i,:) = maxFA-minFA;
                    rangeLA(s,c,t,i,:) = maxLA-minLA;
                    rangeTA(s,c,t,i,:) = maxTA-minTA;
                    rangePA(s,c,t,i,:) = maxPA-minPA;
                    All_rangeMTPA(s,c,t,i,:) = maxMTPA-minMTPA;
                    All_rangeAA(s,c,t,i,:) = maxAA-minAA;
                    All_rangeKA(s,c,t,i,:) = maxKA-minKA;
                    All_rangeHA(s,c,t,i,:) = maxHA-minHA;

                    %reaction force data
                    sRd = Rd(HS(i):TO(i),:);
                    [pkGRF, pkGRF_idx] = max(sRd(:,2));
                    sMd = Md(HS(i):TO(i),:);
                    sCOP = COP(HS(i):TO(i),:);

                    %calculate joint angles at Rhip minimum displacement
                    [minHip, minHip_idx] = min(RGT(HS(i):TO(i),2));
                    All_mMTPA(s,c,t,i,:) = MTPA(HS(i)+minHip_idx,:);
                    All_mAA(s,c,t,i,:) = AA(HS(i)+minHip_idx,:);
                    All_mKA(s,c,t,i,:) = KA(HS(i)+minHip_idx,:);
                    All_mHA(s,c,t,i,:) = HA(HS(i)+minHip_idx,:);

                    %MTP JOINT
                    Fcutoff = 12;
                    Mcutoff = 12;

                    sLocalFmtp = filterdata(LocalFmtp(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
                    while length(sLocalFmtp) < length(ind_stance) sLocalFmtp = [sLocalFmtp;sLocalFmtp(end,:)]; end
                    sLocalMmtp = filterdata(LocalMmtp(HS(i):TO(i),:), VideoFrameRate,0,Mcutoff,4,0);
                    while length(sLocalMmtp) < length(ind_stance) sLocalMmtp = [sLocalMmtp;sLocalMmtp(end,:)]; end
                    sGlobalFmtp = filterdata(GlobalFmtp(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
                    sGlobalMmtp = filterdata(GlobalMmtp(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);

                    %ANKLE JOINT
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

                    %KNEE JOINT
                    PSDcum = cumsum(PSD);
                    PSDcum = PSDcum./PSDcum(end);
                    Fcutoff = interp1(PSDcum,freq,Fper);
                    Mcutoff = interp1(PSDcum,freq,Mper);
                    Fcutoff = 12;
                    Mcutoff = 12;

                    sLocalFknee = filterdata(LocalFknee(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
                    sLocalMknee = filterdata(LocalMknee(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);
                    sGlobalFknee = filterdata(GlobalFknee(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
                    sGlobalMknee = filterdata(GlobalMknee(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);

                    %HIP JOINT
                    [PSD, freq] = MyPSD(sLocalFknee(ind_stance,2),VideoFrameRate);
                    PSDcum = cumsum(PSD);
                    PSDcum = PSDcum./PSDcum(end);
                    Fcutoff = interp1(PSDcum,freq,Fper);
                    Mcutoff = interp1(PSDcum,freq,Mper);
                    Fcutoff = 12;
                    Mcutoff = 12;

                    sLocalFhip = filterdata(LocalFhip(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
                    sLocalMhip = filterdata(LocalMhip(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);
                    sGlobalFhip = filterdata(GlobalFhip(HS(i):TO(i),:),VideoFrameRate,0,Fcutoff,4,0);
                    sGlobalMhip = filterdata(GlobalMhip(HS(i):TO(i),:),VideoFrameRate,0,Mcutoff,4,0);

                    %Additional parameters to export for TimmSimm joint centers
                    sR_Jankle = R_Jankle(HS(i):TO(i),:);
                    sR_Jknee = R_Jknee(HS(i):TO(i),:);
                    sR_Jhip = R_Jhip(HS(i):TO(i),:);

                    %Rotation matrices
                    sR_FootTM = R_FootTM(HS(i):TO(i),:,:);
                    sR_LegTM = R_LegTM(HS(i):TO(i),:,:);
                    sR_ThighTM = R_ThighTM(HS(i):TO(i),:,:);
                    sPelvisTM = PelvisTM(HS(i):TO(i),:,:);
                    sPatellaTM = PatellaTM(HS(i):TO(i),:,:);

                    %Joint Powers
                    sMTP_P = RMTP_P(HS(i):TO(i));
                    sA_P = RA_P(HS(i):TO(i));
                    sK_P = RK_P(HS(i):TO(i));
                    sH_P = RH_P(HS(i):TO(i));

                    %Total Joint Work
                    All_MTP_W(s,c,t,i) = trapz((HS(i):TO(i))/VideoFrameRate,sMTP_P);
                    All_A_W(s,c,t,i) = trapz((HS(i):TO(i))/VideoFrameRate,sA_P);
                    All_K_W(s,c,t,i) = trapz((HS(i):TO(i))/VideoFrameRate,sK_P);
                    All_H_W(s,c,t,i) = trapz((HS(i):TO(i))/VideoFrameRate,sH_P);

                    %Positive Joint Work
                    MTP_Wp_ind = find(sMTP_P > 0);
                    if length(MTP_Wp_ind) > 1
                        MTP_Wp = trapz((1:length(MTP_Wp_ind))/VideoFrameRate,sMTP_P(MTP_Wp_ind));
                    else
                        MTP_Wp = 0;
                    end
                    All_MTP_Wp(s,c,t,i) = MTP_Wp;

                    A_Wp_ind = find(sA_P > 0);
                    if length(A_Wp_ind) > 1
                        A_Wp = trapz((1:length(A_Wp_ind))/VideoFrameRate,sA_P(A_Wp_ind));
                    else
                        A_Wp = 0;
                    end
                    All_A_Wp(s,c,t,i) = A_Wp;

                    K_Wp_ind = find(sK_P > 0);
                    if length(K_Wp_ind) > 1
                        K_Wp = trapz((1:length(K_Wp_ind))/VideoFrameRate,sK_P(K_Wp_ind));
                    else
                        K_Wp = 0;
                    end
                    All_K_Wp(s,c,t,i) = K_Wp;

                    H_Wp_ind = find(sH_P > 0);
                    if length(H_Wp_ind) > 1
                        H_Wp = trapz((1:length(H_Wp_ind))/VideoFrameRate,sH_P(H_Wp_ind));
                    else
                        H_Wp = 0;
                    end
                    All_H_Wp(s,c,t,i) = H_Wp;

                    %Negative Joint Work
                    MTP_Wn_ind = find(sMTP_P < 0);
                    if length(MTP_Wn_ind) > 1
                        MTP_Wn = trapz((1:length(MTP_Wn_ind))/VideoFrameRate,sMTP_P(MTP_Wn_ind));
                    else
                        MTP_Wn = 0;
                    end
                    All_MTP_Wn(s,c,t,i) = MTP_Wn;

                    A_Wn_ind = find(sA_P < 0);
                    if length(A_Wn_ind) > 1
                        A_Wn = trapz((1:length(A_Wn_ind))/VideoFrameRate,sA_P(A_Wn_ind));
                    else
                        A_Wn = 0;
                    end
                    All_A_Wn(s,c,t,i) = A_Wn;

                    K_Wn_ind = find(sK_P < 0);
                    if length(K_Wn_ind) > 1
                        K_Wn = trapz((1:length(K_Wn_ind))/VideoFrameRate,sK_P(K_Wn_ind));
                    else
                        K_Wn = 0;
                    end
                    All_K_Wn(s,c,t,i) = K_Wn;

                    H_Wn_ind = find(sH_P < 0);
                    if length(H_Wn_ind) > 1
                        H_Wn = trapz((1:length(H_Wn_ind))/VideoFrameRate,sH_P(H_Wn_ind));
                    else
                        H_Wn = 0;
                    end
                    All_H_Wn(s,c,t,i) = H_Wn;

                    %Gut checks using StickFigure and mtpcopaxis
                    % COP_mtpaxis_plot(COPfilt(trialInds,:),MTP5Head(trialInds,:),MTP1Head(trialInds,:))
                    % StickFigure1([ToeboxAnt(trialInds,:) HeelLat(trialInds,:) R_Jankle(trialInds,:) R_Jknee(trialInds,:) R_Jhip(trialInds,:) R_CMfoot(trialInds,:) R_CMleg(trialInds,:) R_CMthigh(trialInds,:)],[1:5],[6:8],Rd(trialInds,:),COP(trialInds,:),[0,0,1],[-2, 1,  0, 1.5 ,-1, 1]);


                    if dofigs_stride
                        figure(1);
                        set(gcf,'name',filenameDynamic);
                        subplot(3,1,1);plot(time,sHA(:,2)); xlabel('Time [s]');ylabel('Local Hip In/Ex  [degrees]');
                        subplot(3,1,2);plot(time,sKA(:,2));xlabel('Time [s]');ylabel('Local Knee In/Ex  [degrees]');
                        subplot(3,1,3);plot(time,sAA(:,2));xlabel('Time [s]');ylabel('Local Ankle In/Ex  [degrees]');

                        figure(2);
                        set(gcf,'name',filenameDynamic);
                        subplot(3,1,1);plot(time,sHA(:,1)); xlabel('Time [s]');ylabel('Local Hip Ab/Ad  [degrees]');
                        subplot(3,1,2);plot(time,sKA(:,1));xlabel('Time [s]');ylabel('Local Knee Ab/Ad  [degrees]');
                        subplot(3,1,3);plot(time,sAA(:,1));xlabel('Time [s]');ylabel('Local Ankle Ab/Ad  [degrees]');

                        figure(3);
                        set(gcf,'name',filenameDynamic);
                        subplot(3,1,1);plot(time,sHA(:,3)); xlabel('Time [s]');ylabel('Local Hip Fl/Ex  [degrees]');
                        subplot(3,1,2);plot(time,sKA(:,3));xlabel('Time [s]');ylabel('Local Knee Fl/Ex  [degrees]');
                        subplot(3,1,3);plot(time,sAA(:,3));xlabel('Time [s]');ylabel('Local Ankle Fl/Ex  [degrees]');

                    end

                    %Interpolate data
                    percent_old = (0:1/VideoFrameRate:(length(time)-1)/VideoFrameRate);
                    percent_old = (percent_old/percent_old(end))*100;
                    percent_new = (0:1:100);
                    new_st = ((length(time)-1)/VideoFrameRate)/101;
                    interpFoot_Angle = interp1(percent_old,sFA,percent_new,'pchip');
                    interpLeg_Angle = interp1(percent_old,sLA,percent_new,'pchip');
                    interpThigh_Angle = interp1(percent_old,sTA,percent_new,'pchip');
                    interpPelvis_Angle = interp1(percent_old,sPA,percent_new,'pchip');
                    interpPelvis_Location = interp1(percent_old,(0.50*RASIS(HS(i):TO(i),:))+(0.50*LASIS(HS(i):TO(i),:)),percent_new,'pchip');

                    interpMTP_Angle = interp1(percent_old,sMTPA,percent_new,'pchip');
                    interpAnkle_Angle = interp1(percent_old,sAA,percent_new,'pchip');
                    interpKnee_Angle = interp1(percent_old,sKA,percent_new,'pchip');
                    interpHip_Angle = interp1(percent_old,sHA,percent_new,'pchip');

                    interpMTP_Reaction_Local = interp1(percent_old,sLocalFmtp,percent_new,'pchip');
                    interpAnkle_Reaction_Local = interp1(percent_old,sLocalFankle,percent_new,'pchip');
                    interpKnee_Reaction_Local = interp1(percent_old,sLocalFknee,percent_new,'pchip');
                    interpHip_Reaction_Local = interp1(percent_old,sLocalFhip,percent_new,'pchip');
                    interpAnkle_Reaction_Global = interp1(percent_old,sGlobalFankle,percent_new,'pchip');
                    interpKnee_Reaction_Global = interp1(percent_old,sGlobalFknee,percent_new,'pchip');
                    interpHip_Reaction_Global = interp1(percent_old,sGlobalFhip,percent_new,'pchip');

                    interpMTP_Moment_Local = interp1(percent_old,sLocalMmtp,percent_new,'pchip');
                    interpAnkle_Moment_Local = interp1(percent_old,sLocalMankle,percent_new,'pchip');
                    interpKnee_Moment_Local = interp1(percent_old,sLocalMknee,percent_new,'pchip');
                    interpHip_Moment_Local= interp1(percent_old,sLocalMhip,percent_new,'pchip');
                    interpAnkle_Moment_Global= interp1(percent_old,sGlobalMankle,percent_new,'pchip');
                    interpKnee_Moment_Global= interp1(percent_old,sGlobalMknee,percent_new,'pchip');
                    interpHip_Moment_Global= interp1(percent_old,sGlobalMhip,percent_new,'pchip');

                    interpAnkle_Joint = interp1(percent_old,sR_Jankle,percent_new);
                    interpKnee_Joint = interp1(percent_old,sR_Jknee,percent_new);
                    interpHip_Joint = interp1(percent_old,sR_Jhip,percent_new);

                    interpGround_Reaction_Global = interp1(percent_old,sRd,percent_new);
                    interpGround_Moment_Global = interp1(percent_old,sMd,percent_new);
                    interpCOP_Global = interp1(percent_old,sCOP,percent_new);

                    interpPelvisRM = interp1(percent_old,sPelvisTM,percent_new);
                    interpThighRM = interp1(percent_old,sR_ThighTM,percent_new);
                    interpLegRM = interp1(percent_old,sR_LegTM,percent_new);
                    interpFootRM = interp1(percent_old,sR_FootTM,percent_new);
                    interpPatellaRM = interp1(percent_old,sPatellaTM,percent_new);

                    interpMTP_P = interp1(percent_old,sMTP_P,percent_new,'pchip');
                    interpA_P = interp1(percent_old,sA_P,percent_new,'pchip');
                    interpK_P = interp1(percent_old,sK_P,percent_new,'pchip');
                    interpH_P = interp1(percent_old,sH_P,percent_new,'pchip');

                    if doOutputTimSIMMTrials
                        outfile2=([mainDir timsimmDir filenameDynamic 'T' num2str(i) '_SD']);
                        save (outfile2, 'int*', 'J*', '*RM', 'new_st', 'BW');
                        display(' saved');
                    end

                    if doOutputSIMMTrials
                        outfilename = [filenameDynamic 'T' num2str(i)];
                        outfile1 = [mainDir timsimmDir outfilename '.mot'];
                        dlmwrite(outfile1,['name ' outfilename '.mot'], 'delimiter','', 'newline', 'pc');
                        %dlmwrite(outfile1,['datacolumns 22'],'-append', 'delimiter',  '', 'newline', 'pc');
                        dlmwrite(outfile1,'datacolumns 16','-append', 'delimiter',  '', 'newline', 'pc');
                        dlmwrite(outfile1,['datarows ' num2str(length(percent_new))],'-append', 'delimiter',  '', 'newline', 'pc');
                        dlmwrite(outfile1,'range 0.0 100.0','-append', 'delimiter',  '', 'newline', 'pc');
                        dlmwrite(outfile1,'keys g_key','-append', 'delimiter',  '', 'newline', 'pc');
                        dlmwrite(outfile1,'wrap','-append', 'delimiter',  '', 'newline', 'pc');
                        dlmwrite(outfile1,['calc_derivatives ' num2str(new_st)],'-append', 'delimiter',  '', 'newline', 'pc');
                        dlmwrite(outfile1,'endheader','-append', 'delimiter',  '', 'newline', 'pc');
                        dlmwrite(outfile1,' ','-append', 'delimiter',  '', 'newline', 'pc');

                        %header line
                        dlmwrite(outfile1,'time hipx hipy hipz pelvis_rx pelvis_ry pelvis_rz hip_adduction hip_rotation hip_flexion knee_adduction knee_rotation knee_flexion ankle_adduction ankle_rotation ankle_flexion','-append', 'delimiter',  '', 'newline', 'pc');
                        dlmwrite(outfile1,' ','-append', 'delimiter',  '', 'newline', 'pc');

                        %data
                        dlmwrite(outfile1,[percent_new' interpPelvis_Location(:,1) interpPelvis_Location(:,2) interpPelvis_Location(:,3) interpPelvis_Angle(:,1) interpPelvis_Angle(:,2) interpPelvis_Angle(:,3) interpHip_Angle(:,1) interpHip_Angle(:,2) interpHip_Angle(:,3) interpKnee_Angle(:,1) interpKnee_Angle(:,2) interpKnee_Angle(:,3) interpAnkle_Angle(:,1) interpAnkle_Angle(:,2) interpAnkle_Angle(:,3)],'-append', 'delimiter',  ' ', 'newline', 'pc');
                    end

                    %Compile all of the interpolated data
                    All_FA(s,c,t,i,:,:) = interpFoot_Angle;
                    All_LA(s,c,t,i,:,:) = interpLeg_Angle;
                    All_TA(s,c,t,i,:,:) = interpThigh_Angle;
                    All_PA(s,c,t,i,:,:) = interpPelvis_Angle;
                    All_MTPA(s,c,t,i,:,:) = interpMTP_Angle;
                    All_AA(s,c,t,i,:,:) = interpAnkle_Angle;
                    All_KA(s,c,t,i,:,:) = interpKnee_Angle;
                    All_HA(s,c,t,i,:,:) = interpHip_Angle;
                    All_LocalFmtp(s,c,t,i,:,:) = interpMTP_Reaction_Local;
                    All_LocalFankle(s,c,t,i,:,:) = interpAnkle_Reaction_Local;
                    All_LocalFknee(s,c,t,i,:,:) = interpKnee_Reaction_Local;
                    All_LocalFhip(s,c,t,i,:,:) = interpHip_Reaction_Local;
                    All_LocalMmtp(s,c,t,i,:,:) = interpMTP_Moment_Local;
                    All_LocalMankle(s,c,t,i,:,:) = interpAnkle_Moment_Local;
                    All_LocalMknee(s,c,t,i,:,:) = interpKnee_Moment_Local;
                    All_LocalMhip(s,c,t,i,:,:) = interpHip_Moment_Local;
                    All_Rd(s,c,t,i,:,:)=interpGround_Reaction_Global;
                    All_PeakMmtp(s,c,t,i,:) = min(interpMTP_Moment_Local);
                    All_PeakMankle(s,c,t,i,:) = min(interpAnkle_Moment_Local);
                    All_PeakMknee(s,c,t,i,:) = max(interpKnee_Moment_Local);
                    All_PeakMhip(s,c,t,i,:) = min(interpHip_Moment_Local);
                    All_MTP_P(s,c,t,i,:) = interpMTP_P;
                    All_A_P(s,c,t,i,:) = interpA_P;
                    All_K_P(s,c,t,i,:) = interpK_P;
                    All_K_P(s,c,t,i,:) = interpK_P;

                    %Save all required data and remove bad trials
                    if nanFlag == 1
                        All_MTP_P(s,c,t,i,:) = NaN;
                        All_A_P(s,c,t,i,:) = NaN;
                        All_K_P(s,c,t,i,:) = NaN;
                        All_H_P(s,c,t,i,:) = NaN;
                        All_MTP_W(s,c,t,i) = NaN;
                        All_A_W(s,c,t,i) = NaN;
                        All_K_W(s,c,t,i) = NaN;
                        All_H_W(s,c,t,i) = NaN;
                        All_MTP_Wp(s,c,t,i) = NaN;
                        All_A_Wp(s,c,t,i) = NaN;
                        All_K_Wp(s,c,t,i) = NaN;
                        All_H_Wp(s,c,t,i) = NaN;
                        All_MTP_Wn(s,c,t,i) = NaN;
                        All_A_Wn(s,c,t,i) = NaN;
                        All_K_Wn(s,c,t,i) = NaN;
                        All_H_Wn(s,c,t,i) = NaN;
                        All_verticalStiffness(s,c,t,i) = NaN;
                        All_legStiffness(s,c,t,i) = NaN;
                        All_FlightT(s,c,t,i) = NaN;
                        All_ContactT(s,c,t,i) = NaN;
                        All_StrideLength(s,c,t,i) = NaN;
                        All_SFreq(s,c,t,i) = NaN;
                        All_cAA(s,c,t,i,:) = NaN;
                        All_cKA(s,c,t,i,:) = NaN;
                        All_cHA(s,c,t,i,:) = NaN;
                        All_cFA(s,c,t,i,:) = NaN;
                        All_tAA(s,c,t,i,:) = NaN;
                        All_tKA(s,c,t,i,:) = NaN;
                        All_tHA(s,c,t,i,:) = NaN;
                        All_rangeMTPA(s,c,t,i,:) = NaN;
                        All_rangeAA(s,c,t,i,:) = NaN;
                        All_rangeKA(s,c,t,i,:) = NaN;
                        All_rangeHA(s,c,t,i,:) = NaN;
                        All_mAA(s,c,t,i,:) = NaN;
                        All_mKA(s,c,t,i,:) = NaN;
                        All_mHA(s,c,t,i,:) = NaN;
                        All_FA(s,c,t,i,:,:) = NaN;
                        All_LA(s,c,t,i,:,:) = NaN;
                        All_TA(s,c,t,i,:,:) = NaN;
                        All_PA(s,c,t,i,:,:) = NaN;
                        All_MTPA(s,c,t,i,:,:) = NaN;
                        All_AA(s,c,t,i,:,:) = NaN;
                        All_KA(s,c,t,i,:,:) = NaN;
                        All_HA(s,c,t,i,:,:) = NaN;
                        All_LocalFmtp(s,c,t,i,:,:) = NaN;
                        All_LocalFankle(s,c,t,i,:,:) = NaN;
                        All_LocalFknee(s,c,t,i,:,:) = NaN;
                        All_LocalFhip(s,c,t,i,:,:) = NaN;
                        All_LocalMmtp(s,c,t,i,:,:) = NaN;
                        All_LocalMankle(s,c,t,i,:,:) = NaN;
                        All_LocalMknee(s,c,t,i,:,:) = NaN;
                        All_LocalMhip(s,c,t,i,:,:) = NaN;
                        All_Rd(s,c,t,i,:,:) = NaN;
                        All_PeakMmtp(s,c,t,i,:) = NaN;
                        All_PeakMankle(s,c,t,i,:) = NaN;
                        All_PeakMknee(s,c,t,i,:) = NaN;
                        All_PeakMhip(s,c,t,i,:) = NaN;
                    end

                    % n = n+1;

                    if doStickFig
                        StickFigure1([ToeboxAnt(trialInds,:) MTP5Head(trialInds,:) HeelLat(trialInds,:) R_Jankle(trialInds,:) R_Jknee(trialInds,:) R_Jhip(trialInds,:) R_CMfoot(trialInds,:) R_CMleg(trialInds,:) R_CMthigh(trialInds,:)],1:6,7:9,Rd(trialInds,:),COP(trialInds,:),[0,0,1],[-2,1,0,1.5,-1,1]);
                    end

                    if doCSplot
                        figure(100);
                        for f=trialInds
                            axis([-2 1 0 1.5 -1 1]);view(stickview);
                            set(gca,'CameraUpVector',  [0, 1, 0]);
                            hold on;
                            axis manual;% freezes the axis, so that as you go through the animation the axis doesn't change
                            axis equal;
                            plotCS(squeeze(R_CMforefoot(f,:)), squeeze(R_ForefootTM(f,:,:)),'^r',.1);
                            plotCS(squeeze(R_CMrearfoot(f,:)), squeeze(R_RearfootTM(f,:,:)),'^r',.1);
                            plotCS(squeeze(R_CMleg(f,:)), squeeze(R_LegTM(f,:,:)),'^r',.1);
                            plotCS(squeeze(R_CMthigh(f,:)), squeeze(R_ThighTM(f,:,:)),'^r',.1);
                            plotCS(squeeze(CMpelvis(f,:)), squeeze(PelvisTM(f,:,:)),'^r',.1);
                            drawnow;
                            clf
                        end
                    end
                    clear trialInds
                    nanFlag = 0;
                end %step loop

                if dofigs_trial
                    set(gcf,'name',filenameDynamic);
                    subplot(4,2,1);myeb(squeeze(All_HA(s,c,t,:,:,:)));xlabel('Stride [%]');ylabel('Local Hip Angle [deg]');xlim([0 100]);
                    subplot(4,2,3);myeb(squeeze(All_KA(s,c,t,:,:,:)));xlabel('Stride [%]');ylabel('Local Knee Angle  [deg]');xlim([0 100]);
                    subplot(4,2,5);myeb(squeeze(All_AA(s,c,t,:,:,:)));xlabel('Stride [%]');ylabel('Local Ankle Angle  [deg]');xlim([0 100]);
                    subplot(4,2,7);myeb(squeeze(All_MTPA(s,c,t,:,:,:)));xlabel('Stride [%]');ylabel('Local MTP Angle  [deg]');xlim([0 100]);
                    %                     pause(0.5);
                    subplot(4,2,2);myeb(squeeze(All_LocalMhip(s,c,t,:,:,:)));xlabel('Stride [%]');ylabel('Local Hip Moment  [Nm]');xlim([0 100]);
                    subplot(4,2,4);myeb(squeeze(All_LocalMknee(s,c,t,:,:,:)));xlabel('Stride [%]');ylabel('Local Knee Moment  [Nm]');xlim([0 100]);
                    subplot(4,2,6);myeb(squeeze(All_LocalMankle(s,c,t,:,:,:)));xlabel('Stride [%]');ylabel('Local Ankle Moment  [Nm]');xlim([0 100]);
                    subplot(4,2,8);myeb(squeeze(All_LocalMmtp(s,c,t,:,:,:)));xlabel('Stride [%]');ylabel('Local MTP Moment  [Nm]');xlim([0 100]);
                end

                %Clear variables to avoid additional data from previous data being used
                clear GlobalFankle GlobalMankle GlobalFmtp GlobalMmtp
                clear interpFoot_Angle interpLeg_Angle interpThigh_Angle intterpPelvis_Angle interpPelvis_Location;
                clear interpAnkle_Angle interpKnee_Angle interpHip_Angle interpAnkle_Reaction_Local interpKnee_Reaction_Local interpHip_Reaction_Local interpAnkle_Reaction_Global interpKnee_Reaction_Global interpHip_Reaction_Global;
                clear interpAnkle_Joint interpKnee_Joint interpHip_Joint interpAnkle_Moment_Local interpKnee_Moment_Local interpHip_Moment_Local interpAnkle_Moment_Global interpKnee_Moment_Global interpHip_Moment_Global;
                clear interpGround_Reaction_Global interpGround_Moment_Global interpCOP_Global;
                clear interpPelvisRM interpThighRM interpLegRM interpFootRM interpPatellaRM;
                clear HS TO ind_off;
                clear AP_ind;
                clear sMTP_P sA_p sK_p sH_p
                numIP = 0;
            else
                disp('File not found')
            end %file exist loop

            %Trial-averaging all data
            All_verticalStiffness_avg(s,c,t) = squeeze(nanmean(All_verticalStiffness(s,c,t,:),4));
            All_legStiffness_avg(s,c,t) = squeeze(nanmean(All_legStiffness(s,c,t,:),4));
            All_FlightT_avg(s,c,t) = squeeze(nanmean(All_FlightT(s,c,t,:),4));
            All_ContactT_avg(s,c,t) = squeeze(nanmean(All_ContactT(s,c,t,:),4));
            All_StrideLength_avg(s,c,t) = squeeze(nanmean(All_StrideLength(s,c,t,:),4));
            All_SFreq_avg(s,c,t) = squeeze(nanmean(All_SFreq(s,c,t,:),4));
            All_cAA_avg(s,c,t,:) = squeeze(nanmean(All_cAA(s,c,t,:,:),4));
            All_cKA_avg(s,c,t,:)= squeeze(nanmean(All_cKA(s,c,t,:,:),4));
            All_cHA_avg(s,c,t,:) = squeeze(nanmean(All_cHA(s,c,t,:,:),4));
            All_cFA_avg(s,c,t,:) = squeeze(nanmean(All_cFA(s,c,t,:,:),4));
            All_tAA_avg(s,c,t,:) = squeeze(nanmean(All_tAA(s,c,t,:,:),4));
            All_tKA_avg(s,c,t,:) = squeeze(nanmean(All_tKA(s,c,t,:,:),4));
            All_tHA_avg(s,c,t,:) = squeeze(nanmean(All_tHA(s,c,t,:,:),4));
            All_rangeMTPA_avg(s,c,t,:) = squeeze(nanmean(All_rangeMTPA(s,c,t,:,:),4));
            All_rangeAA_avg(s,c,t,:) = squeeze(nanmean(All_rangeAA(s,c,t,:,:),4));
            All_rangeKA_avg(s,c,t,:) = squeeze(nanmean(All_rangeKA(s,c,t,:,:),4));
            All_rangeHA_avg(s,c,t,:) = squeeze(nanmean(All_rangeHA(s,c,t,:,:),4));
            All_mAA_avg(s,c,t,:) = squeeze(nanmean(All_mAA(s,c,t,:,:),4));
            All_mKA_avg(s,c,t,:) = squeeze(nanmean(All_mKA(s,c,t,:,:),4));
            All_mHA_avg(s,c,t,:) = squeeze(nanmean(All_mHA(s,c,t,:,:),4));
            All_FA_avg(s,c,t,:,:) = squeeze(nanmean(All_FA(s,c,t,:,:),4));
            All_LA_avg(s,c,t,:,:) = squeeze(nanmean(All_LA(s,c,t,:,:),4));
            All_TA_avg(s,c,t,:,:) = squeeze(nanmean(All_TA(s,c,t,:,:),4));
            All_PA_avg(s,c,t,:,:) = squeeze(nanmean(All_PA(s,c,t,:,:),4));
            All_MTPA_avg(s,c,t,:,:) = squeeze(nanmean(All_MTPA(s,c,t,:,:,:),4));
            All_AA_avg(s,c,t,:,:) = squeeze(nanmean(All_AA(s,c,t,:,:,:),4));
            All_KA_avg(s,c,t,:,:) = squeeze(nanmean(All_KA(s,c,t,:,:,:),4));
            All_HA_avg(s,c,t,:,:) = squeeze(nanmean(All_HA(s,c,t,:,:,:),4));
            All_LocalFmtp_avg(s,c,t,:,:) = squeeze(nanmean(All_LocalFmtp(s,c,t,:,:,:),4));
            All_LocalFankle_avg(s,c,t,:,:) = squeeze(nanmean(All_LocalFankle(s,c,t,:,:,:),4));
            All_LocalFknee_avg(s,c,t,:,:) = squeeze(nanmean(All_LocalFknee(s,c,t,:,:,:),4));
            All_LocalFhip_avg(s,c,t,:,:) = squeeze(nanmean(All_LocalFhip(s,c,t,:,:,:),4));
            All_LocalMmtp_avg(s,c,t,:,:) = squeeze(nanmean(All_LocalMmtp(s,c,t,:,:,:),4));
            All_LocalMankle_avg(s,c,t,:,:) = squeeze(nanmean(All_LocalMankle(s,c,t,:,:,:),4));
            All_LocalMknee_avg(s,c,t,:,:) = squeeze(nanmean(All_LocalMknee(s,c,t,:,:,:),4));
            All_LocalMhip_avg(s,c,t,:,:) = squeeze(nanmean(All_LocalMhip(s,c,t,:,:,:),4));
            All_Rd_avg(s,c,t,:,:) = squeeze(nanmean(All_Rd(s,c,t,:,:),4));
            All_PeakMmtp_avg(s,c,t,:) = squeeze(nanmean(All_PeakMmtp(s,c,t,:,:),4));
            All_PeakMankle_avg(s,c,t,:) = squeeze(nanmean(All_PeakMankle(s,c,t,:,:),4));
            All_PeakMknee_avg(s,c,t,:) = squeeze(nanmean(All_PeakMknee(s,c,t,:,:),4));
            All_PeakMhip_avg(s,c,t,:) = squeeze(nanmean(All_PeakMhip(s,c,t,:,:),4));
            All_MTP_P_avg(s,c,t,:) = squeeze(nanmean(All_MTP_P(s,c,t,:,:),4));
            All_A_P_avg(s,c,t,:) = squeeze(nanmean(All_A_P(s,c,t,:,:),4));
            All_K_P_avg(s,c,t,:) = squeeze(nanmean(All_K_P(s,c,t,:,:),4));
            All_H_P_avg(s,c,t,:) = squeeze(nanmean(All_K_P(s,c,t,:,:),4));
            All_MTP_W_avg(s,c,t) = squeeze(nanmean(All_MTP_W(s,c,t,:),4));
            All_A_W_avg(s,c,t) = squeeze(nanmean(All_A_W(s,c,t,:),4));
            All_K_W_avg(s,c,t) = squeeze(nanmean(All_K_W(s,c,t,:),4));
            All_H_W_avg(s,c,t) = squeeze(nanmean(All_H_W(s,c,t,:),4));
            All_MTP_Wp_avg(s,c,t) = squeeze(nanmean(All_MTP_Wp(s,c,t,:),4));
            All_A_Wp_avg(s,c,t) = squeeze(nanmean(All_A_Wp(s,c,t,:),4));
            All_K_Wp_avg(s,c,t) = squeeze(nanmean(All_K_Wp(s,c,t,:),4));
            All_H_Wp_avg(s,c,t) = squeeze(nanmean(All_H_Wp(s,c,t,:),4));
            All_MTP_Wn_avg(s,c,t) = squeeze(nanmean(All_MTP_Wn(s,c,t,:),4));
            All_A_Wn_avg(s,c,t) = squeeze(nanmean(All_A_Wn(s,c,t,:),4));
            All_K_Wn_avg(s,c,t) = squeeze(nanmean(All_K_Wn(s,c,t,:),4));
            All_H_Wn_avg(s,c,t) = squeeze(nanmean(All_H_Wn(s,c,t,:),4));


            %% Writing data

            %Save the data to an excel spreadsheet
            if strcmp(shcond{c},'AH')
                outCol = outCol+2;
            elseif strcmp(shcond{c},'AM')
                outCol = outCol+4;
            end

            if strcmp(spcond{t},'PI')
                outCol = outCol+1;
            end


            header = [{'Subject'},{'GelCumulus, RacePace'},{'GelCumulus, PercentImproved'},{'HyperSpeed, RacePace'},{'HyperSpeed, PercentImproved'},{'MetaspeedSky, RacePace'},{'MetaspeedSky, PercentImproved'}];
            %Vertical stiffness
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','vertStiffness');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','vertStiffness');
            writematrix(All_verticalStiffness_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','vertStiffness');
            %Leg stiffness
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','legStiffness');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','legStiffness');
            writematrix(All_legStiffness_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','legStiffness');
            %Flight phase
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','FlightTime');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','FlightTime');
            writematrix(All_FlightT_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','FlightTime');
            %All_ContactT_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','ContactTime');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','ContactTime');
            writematrix(All_ContactT_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','ContactTime');
            %All_StrideLength_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','StrideLength');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','StrideLength');
            writematrix(All_StrideLength_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','StrideLength');
            %All_SFreq_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','StrideFrequency');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','StrideFrequency');
            writematrix(All_SFreq_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','StrideFrequency');
            %All_cAA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','AnkleAngleContact');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','AnkleAngleContact');
            writematrix(All_cAA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','AnkleAngleContact');
            %All_cKA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','KneeAngleContact');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','KneeAngleContact');
            writematrix(All_cKA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','KneeAngleContact');
            %All_cHA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','HipAngleContact');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','HipAngleContact');
            writematrix(All_cHA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','HipAngleContact');
            %All_cFA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','FootAngleContact');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','FootAngleContact');
            writematrix(All_cFA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','FootAngleContact');
            %All_tAA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','AnkleAngleToeOff');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','AnkleAngleToeOff');
            writematrix(All_tAA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','AnkleAngleToeOff');
            %All_tKA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','KneeAngleToeOff');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','KneeAngleToeOff');
            writematrix(All_tKA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','KneeAngleToeOff');
            %All_tHA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','HipAngleToeOff');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','HipAngleToeOff');
            writematrix(All_tHA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','HipAngleToeOff');
            %All_rangeMTPA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','MTPROM');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','MTPROM');
            writematrix(All_rangeMTPA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','MTPROM');
            %All_rangeAA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','AnkleROM');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','AnkleROM');
            writematrix(All_rangeAA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','AnkleROM');
            %All_rangeKA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','KneeROM');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','KneeROM');
            writematrix(All_rangeKA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','KneeROM');
            %All_rangeHA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','HipROM');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','HipROM');
            writematrix(All_rangeHA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','HipROM');
            %All_mAA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','AnkleAngleMinHipHeight');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','AnkleAngleMinHipHeight');
            writematrix(All_mAA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','AnkleAngleMinHipHeight');
            %All_mKA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','KneeAngleMinHipHeight');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','KneeAngleMinHipHeight');
            writematrix(All_mKA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','KneeAngleMinHipHeight');
            %All_mHA_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','HipAngleMinHipHeight');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','HipAngleMinHipHeight');
            writematrix(All_mHA_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','HipAngleMinHipHeight');
            %All_PeakMmtp_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','MTPPeakMoment');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','MTPPeakMoment');
            writematrix(All_PeakMmtp_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','MTPPeakMoment');
            %All_PeakMankle_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','AnklePeakMoment');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','AnklePeakMoment');
            writematrix(All_PeakMankle_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','AnklePeakMoment');
            %All_PeakMknee_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','KneePeakMoment');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','KneePeakMoment');
            writematrix(All_PeakMknee_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','KneePeakMoment');
            %All_PeakMhip_avg
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','HipPeakMoment');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','HipPeakMoment');
            writematrix(All_PeakMhip_avg(s,c,t,3), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','HipPeakMoment');
            %All_MTP_Wp
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','MTPWpos');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','MTPWpos');
            writematrix(All_MTP_Wp_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','MTPWpos');
            %All_Ankle_Wp
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','AnkleWpos');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','AnkleWpos');
            writematrix(All_A_Wp_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','AnkleWpos');
            %All_Knee_Wp
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','KneeWpos');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','KneeWpos');
            writematrix(All_K_Wp_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','KneeWpos');
            %All_Hip_Wp
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','HipWpos');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','HipWpos');
            writematrix(All_H_Wp_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','HipWpos');
            %All_MTP_Wn
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','MTPWneg');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','MTPWneg');
            writematrix(All_MTP_Wn_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','MTPWneg');
            %All_Ankle_Wn
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','AnkleWneg');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','AnkleWneg');
            writematrix(All_A_Wn_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','AnkleWneg');
            %All_Knee_Wn
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','KneeWneg');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','KneeWneg');
            writematrix(All_K_Wn_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','KneeWneg');
            %All_Hip_Wn
            writecell(header,[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range','A1','Sheet','HipWneg');
            writematrix(subj(s),[mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',['A' num2str(s+1)],'Sheet','HipWneg');
            writematrix(All_H_Wn_avg(s,c,t), [mainDir,'Inverse Dynamics Outputs.xlsx'],'Range',[alphabet(outCol) num2str(s+1)],'Sheet','HipWneg');


        end %speed loop
    end %shoe loop
end %subject loop

function y = dynamicLPF(x, fs, fc_light, fc_heavy, order, blend_frac)
% Apply light filtering to initial 20% and heavier filtering to remaining stance.
if nargin < 6
    blend_frac = 0.05;
end
n = size(x,1);
if n < 6
    y = x;
    return;
end
idx20 = max(2, round(0.20*n));
blend_len = max(3, round(blend_frac*n));

y_light = filterdata(x, fs, 0, fc_light, order, 0);
y_heavy = filterdata(x, fs, 0, fc_heavy, order, 0);

y = y_heavy;
y(1:idx20,:) = y_light(1:idx20,:);

i1 = max(1, idx20 - floor(blend_len/2));
i2 = min(n, idx20 + floor(blend_len/2));
a = linspace(1,0,i2-i1+1)';
y(i1:i2,:) = a.*y_light(i1:i2,:) + (1-a).*y_heavy(i1:i2,:);
end


