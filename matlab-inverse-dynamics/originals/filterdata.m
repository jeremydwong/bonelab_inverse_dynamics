function [data_filt] = filterdata(data_raw,sf,hpcutoff,lpcutoff,order,show)
  %This function bandpass filters discrete data in data_raw sampled at 
  %sf Hz. Set hpcutoff = 0 for a lowpass filter. Set lpcutoff =0 for a high
  %pass filter

  [rows cols] = size(data_raw);

  fc1 = hpcutoff;
  fc2 = lpcutoff;
  if hpcutoff == 0 
      Wn = [2*fc2/sf];
  else
      Wn = [2*fc1/sf, 2*fc2/sf];
  end;
      
  [butter_b,butter_a] = butter(order/2,Wn);
%   for p = 1:cols
%       startdata=1;
%       enddata=rows;
      %while data_raw(startdata,p)==0;startdata=startdata+1;end;
      %while data_raw(enddata,p)==0;enddata=enddata-1;end;
      %dr=data_raw(startdata:enddata,p);
      data_filt = filtfilt(butter_b,butter_a,data_raw);
      
%   end; %for
  
  %uncomment the following line to graph the filter response
%   [h f]=freqz(butter_b,butter_a,0:100,sf);
  
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
%           pause
      end;
  end;
end



