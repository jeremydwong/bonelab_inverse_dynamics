function plotCS(origin, RM,culr,lngth)

p0=origin;
daspect('manual');
set(gca,'CameraUpVector',  [0, 1, 0]);


hold on;
if size(RM,1) == 3
    if ~isnan(RM)
        px=p0+lngth*(RM(1,:));
        py=p0+lngth*(RM(2,:));
        pz=p0+lngth*(RM(3,:));
        arrow3(p0,px,'^r',.35);
        arrow3(p0,py,'^g',.35);
        arrow3(p0,pz,'^b',.35);
    end
else
    if ~isnan(RM)
        px=p0+lngth*(RM(1:3));
        py=p0+lngth*(RM(4:6));
        pz=p0+lngth*(RM(7:9));
        arrow3(p0,px,culr,.35);
        arrow3(p0,py,culr,.35);
        arrow3(p0,pz,culr,.35);
    end
end
hold off;