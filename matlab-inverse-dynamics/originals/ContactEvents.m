%Written by:
%Brent Edwards (Winter 2015)

%you will need to change this for downhill running slightly
function [HS TO ind] = ContactEvents(Fvert,COPml, sf, slope);

%heavily smooth the vertical grf
smooth = filterdata(Fvert,sf,0,10,4,0);

%grab all smooth peaks above 100 Newtons
[peaks,locs] = pkpick(smooth,100,'');
%remove first and last peaks to avoid partial GRFs
peaks([1 end])=[];locs([1 end])=[];

%find heel strike off
for i = 1:length(peaks)
    k = locs(i);
    while k>1 & Fvert(k)>20
        k = k-1;
    end
    HS(i,:) = k;
end

%find toe off
for i = 1:length(peaks)
    k = locs(i);
    while k<length(Fvert) & Fvert(k)>20
        k = k+1;
    end
    TO(i,:) = k;
end

%determine right verses left HS/TO
COPloc = COPml(locs);
m1 = mean(COPloc(1:2:end-1));
m2 = mean(COPloc(2:2:end));

%the one that is further away is the right leg for flat and uphill trials
if slope>=0 & m1<m2;%m2 is the right foot remove HS and TO for left foot
    HS(1:2:end-1)=[];
    TO(1:2:end-1)=[];
elseif slope>=0 & m1>m2%m1 is the right foot
    HS(2:2:end)=[];
    TO(2:2:end)=[];
elseif slope<0 & m1<m2%m1 is the right foot
    HS(2:2:end)=[];
    TO(2:2:end)=[];
else%m2 is the right
    HS(1:2:end-1)=[];
    TO(1:2:end-1)=[];
end

%remove duplicates. This happens during walking trials sometimes
HS = unique(HS);TO = unique(TO);

%determine when right foot is off the ground
ind = [(1:HS(1))';(TO(end):length(Fvert))'];
for i = 1:length(TO)-1;
    ind = [ind;(TO(i):HS(i+1))'];
end
ind = sort(ind);

end

