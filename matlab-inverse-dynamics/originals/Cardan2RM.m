
function RM=Cardan2RM(ang,sequence)

%Returns the rotation matrix from the angle and sequence of rotations

 X=1;Y=2;Z=3;
 rot(findstr('x',sequence))=X;
 rot(findstr('y',sequence))=Y; 
 rot(findstr('z',sequence))=Z;

RM=repmat2(eye(3),size(ang,1), [1 2], 2);
for d=1:3
    for i=1:size(ang,1)
        evalstr=['rot' sequence(d) '_global(ang(i,rot(d)),RM(i,:,:));'];
        %evalstr=['rot' sequence(d) '(ang(i,rot(d)),RM(i,:,:));'];
        RM(i,:,:)=squeeze(eval(evalstr));
    end;
end



end


