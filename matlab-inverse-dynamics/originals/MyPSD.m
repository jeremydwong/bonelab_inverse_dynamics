function [PSD, freq] = MyPSD(sig, SF)
% %This function calucations the power spectral density (PSD) and the 
% %frequencies (freq) at which the powers occur. Zoeros are padded so that
% %there are more data points than the sampling frequency (SF) and the 
% %number of data points is a power of 2. The power is adjusted for the 
% %padded zeros and the frequency bins are interpolated so that the 
% %width of each bin is 1 Hz.
% 
% 
% %Calculate PSD
%                     
%         ndp = length(sig);
%         z = 1;
%         while (2 ^ z) < ndp; z = z + 1; end ; %calculate next power of 2 > ndp
%         while (2 ^ z + ndp) < SF; z = z + 1; end;  %make sure more points than sf
%         tpnts = 2 ^ z;             %total number of points
%         nzp = tpnts - ndp;         %number of zero points
%                   
% 
%         sig = detrend(sig);
% %         sig = sig-mean(sig);
%         pads = ceil(nzp/2);
%         sig = padarray(sig,pads,0,'both');%pads front and back
%         [PSD, freq] = pwelch(sig,rectwin(length(sig)),[],[],SF);
%         PSD = PSD*((ndp+nzp)/ndp); %adjust power for padded zeros
%         finterp=0:(.5*SF-1);
%         PSD = interp1(freq,PSD,finterp,'linear')/freq(2); %interpolate to 1 Hz and adjust power for new bin width
%         freq=finterp;             
% %                
% end

% function [PSD, freq] = MyPSD(sig, SF)
%This function calucations the power spectral density (PSD) and the 
%frequencies (freq) at which the powers occur. The frequency bin width is 1 Hz.

 %[Pxx,F] = PWELCH(X,WINDOW,NOVERLAP,NFFT,Fs)
        
        oldtime=(0:1/SF:((length(sig)-1)/SF))';
        linetime=[0 ((length(sig)-1)/SF)]';
        lne=spline(linetime,[sig(1) sig(end)],oldtime);
        sig=sig-lne;
%        
%          sig=wextend('1D','asymw',sig,round(length(sig)),'b');
%          sig=wextend('1D','asymw',sig,round(.5*length(sig)),'b');
%          oldtime=(0:1/SF:((length(sig)-1)/SF))';

%         sig = detrend(sig);
        sig = sig-mean(sig);         
        NFFT=2^nextpow2(max(SF,length(sig)));
        newtime=(0:1/NFFT:((length(sig)-1)/SF))';
        sig=spline(oldtime,sig,newtime);
%          plot(sig);
%          pause;
       
        [PSD, freq] = pwelch(sig,rectwin(min(length(sig),NFFT)),[],NFFT,NFFT);
        PSD = PSD(1:round(0.5*SF)+1);
        freq = freq(1:round(0.5*SF)+1);
        

end



