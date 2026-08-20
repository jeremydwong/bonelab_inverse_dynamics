function[angles,anglesb] = CardanSegAngles(seg, sequence)
%counterclockwise positive
 X=1;Y=2;Z=3;
 rot(findstr('x',sequence))=X;
 rot(findstr('y',sequence))=Y;
 rot(findstr('z',sequence))=Z;
 
 if size(seg,2)==3 %numrowsx3x3 matrices
     for i=1:size(seg,1)
         LCS=squeeze([seg(i,1,1:3);seg(i,2,1:3);seg(i,3,1:3)]);
         if ~isnan(LCS);
             [a b]=RTOCARDA(inv(LCS),rot(1),rot(2),rot(3));
             angles(i,:)=a*57.3;
%              anglesb(i,:)=b*57.3;
         else
             angles(i,:)=[NaN NaN NaN];
         end
     end
 else %numrowsx9 matrices

     for i=1:size(seg,1)
         LCS=[seg(i,1:3);seg(i,4:6);seg(i,7:9)];
         if ~isnan(LCS);
             [a b]=RTOCARDA(inv(LCS),rot(1),rot(2),rot(3));
             angles(i,:)=a*57.3;
         else
             angles(i,:)=[NaN NaN NaN];
         end;
%          anglesb(i,:)=b*57.3;
         %if mag rot(2) >90 use b (check this)
     end
 end