%Written by:
%Brent Edwards (Winter 2015)

%you will need to change this for downhill running slightly
function [HS, TO, ind] = ContactEvents_Motek(Fvert,COPml,sf)

%heavily smooth the vertical grf
smoothH = filterdata(Fvert,sf,0,5,4,0);

%grab all smooth peaks above 100 Newtons
[peaks,locs] = findpeaks(smoothH,'MinPeakHeight',100);
%remove first and last peaks to avoid partial GRFs
peaks([1 end])=[];
locs([1 end])=[];

%lightly smooth the vertical grf for HS and TO times
smoothL = filterdata(Fvert,sf,0,20,4,0);

%find heel strike
for i = 1:length(peaks)
    k = locs(i);
    while k>1 && smoothL(k)>5
        k = k-1;
    end
    HS(i,:) = k;
end

%find toe off
for i = 1:length(peaks)
    k = locs(i);
    while k<length(smoothL) && smoothL(k)>5
        k = k+1;
    end
    TO(i,:) = k;
end

%determine right verses left HS/TO
COPloc = COPml(locs);
m1 = mean(COPloc(1:2:end-1));
m2 = mean(COPloc(2:2:end));

if m1<m2 %m2 is the right foot remove HS and TO for left foot
    HS(1:2:end-1)=[];
    TO(1:2:end-1)=[];
else
    HS(2:2:end)=[];
    TO(2:2:end)=[];
end

%determine when right foot is off the ground
ind = [(1:HS(1))';(TO(end):length(Fvert))'];
for i = 1:length(TO)-1
    ind = [ind;(TO(i):HS(i+1))'];
end
ind = sort(ind);

end

