function[angles] = CardanJointAngles(seg1, seg2, sequence)
%seg1 = proximal segment
%seg2 = distal segment
%sequence = sequence of rotations (i.e., sequence = 'zxy')
 X=1;Y=2;Z=3;
 rot(findstr('x',sequence))=X;
 rot(findstr('y',sequence))=Y;
 rot(findstr('z',sequence))=Z;
 
 if size(seg1,2)==3 %numrowsx3x3 matrices
     for i=1:size(seg1,1)
         LCS1=squeeze([seg1(i,1,1:3);seg1(i,2,1:3);seg1(i,3,1:3)]);
         LCS2=squeeze([seg2(i,1,1:3);seg2(i,2,1:3);seg2(i,3,1:3)]);
         if ~(isnan(LCS1)|isnan(LCS2));
             [angles(i,:)]=RTOCARDA((LCS1)*inv(LCS2),rot(1),rot(2),rot(3))*57.3;
         else
             angles(i,:)=[NaN NaN NaN];
            
         end

     end
 else %numrowsx9
     for i=1:size(seg1,1)
         LCS1=[seg1(i,1:3);seg1(i,4:6);seg1(i,7:9)];
         LCS2=[seg2(i,1:3);seg2(i,4:6);seg2(i,7:9)];
         if ~(isnan(LCS1)|isnan(LCS2));
             [angles(i,:)]=RTOCARDA((LCS1)*inv(LCS2),rot(1),rot(2),rot(3))*57.3;
         else
             angles(i,:)=[NaN NaN NaN];
             
         end
     end
 end