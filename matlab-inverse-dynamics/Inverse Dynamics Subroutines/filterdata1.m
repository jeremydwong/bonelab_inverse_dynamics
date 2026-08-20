function [data_filt] = filterdata1(data_raw,sf,hpcutoff,lpcutoff,order,show)
  %This function bandpass filters discrete data in data_raw sampled at 
  %sf Hz. Set hpcutoff = 0 for a lowpass filter.
  %Inputs: data_raw = raw, unfiltered data
  %        sf = sampling frequency (Hz)
  %        hpcutoff = highpass cutoff frequency (if this is a lowpass filter, set hpcutoff=0
  %        lpcutoff = lowpass cutoff frequency
  %        order = filter order (4 = fourth order filter)
  %        show = whether to graph the filtered data (1 = yes, 0 = no)
  %Outputs: data_filt = filtered data

  [rows cols] = size(data_raw);

  fc1 = hpcutoff;
  fc2 = lpcutoff;
  if hpcutoff == 0 
      Wn = [2*fc2/sf];
  else
      Wn = [2*fc1/sf, 2*fc2/sf];
  end;
      
  [butter_b,butter_a] = butter(order/2,Wn);
  for p = 1:cols
       startdata=1;
       enddata=rows;
       while data_raw(startdata,p)==0 && startdata < rows;startdata=startdata+1;end;
       while data_raw(enddata,p)==0 && enddata >1;enddata=enddata-1;end;
      
       if enddata-startdata>(3*order)
         dr=data_raw(startdata:enddata,p);
         data_filt(startdata:enddata,p) = filtfilt(butter_b,butter_a,dr);
         
       else
          data_filt(:,p)=0;
       end
  end; %for
  
  %uncomment the following line to graph the filter response
%    figure(2);freqz(butter_b,butter_a,0:500,sf);
%    pause;
  
  %graph raw and filtered data if show = 1
  if show 
      [npoints nchan] = size(data_raw);
      t = 1/sf:1/sf:npoints/sf;
      figure;
      for ch = 1:nchan
          hold off;
          plot(t,data_raw(:,ch),'.');
          axis auto;          
          hold on;          
          plot(t,data_filt(:,ch),'r');
          title([num2str(ch) ' of ' num2str(nchan)]);
          drawnow;
          axis manual; % freezes the axis, so that as you go through the animation the axis doesn't change
          pause;
      end;
  end;
end


