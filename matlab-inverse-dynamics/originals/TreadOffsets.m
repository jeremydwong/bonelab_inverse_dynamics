%Written by:
%Brent Edwards (Winter 2015)

%This will remove any trend in the treadmill force data.
function newAnalogSignals = TreadOffsets(AnalogSignals);

indF1 = find(AnalogSignals(:,3)>-20);
indF2 = find(AnalogSignals(:,9)>-20);
    if length(indF1)<length(indF2); %subject is running on plate 1 using for offsets
        ind = indF1;
    else
        ind = indF2;%subject is runnign on plate 2
    end
%%get the mean values for each signal
offsets = mean(AnalogSignals(ind,:));
offsets = repmat(offsets,length(AnalogSignals),1);
newAnalogSignals = AnalogSignals-offsets;%subtract offsets

end