function StickFigure1(kindata, links,points,grf,cop,vu,ax)
%Plots a stick figure by connecting the paths in 'links'
%Uses GetMarkers3.

%kindata contains the kinematic data, each markers is 3 columns (x,y,z), rows represent frames of data.
%links contains the paths to connect, for instance [1:5,7,9;10,11,12].
%points contains the points that you want to display a marker at
%vu is the view, [0 1 1] for the y-z plane.
%ax sets the axis limits to include the range of motion:   [xmin xmax ymin ymax zmin zmax].
maxkz=max(ax);
if ~isempty(grf);maxaz=max(grf(:,2));else; maxaz=1;end
scale=maxkz/maxaz;
% Create movie file with required parameters
% h_fig = figure;
% winsize = get(h_fig,'Position');
% winsize(1:2) = [0 0];
% fps= 20;
% outFileName='c:\temp\trialvid.avi';
% outfile = sprintf('%s',outFileName)
% mov = avifile(outfile,'fps',fps,'quality',100,'compression','None');
% set(h_fig,'NextPlot','replacechildren');


close all;pause on;
[kindata_length,n]=size(kindata);
[numlinks] = length(links);
numpoints=size(points,2);
marks = GetMarkers3(kindata,links);%get markers in the stick figure
%points = GetMarkers3(kinpoints,[1:numpoints]);
points = GetMarkers3(kindata,points);


%make the 2D array a 3D array (frame, dimension, path)
kindata_stick = reshape(marks,[kindata_length 3 numlinks]);
points_stick = reshape(points,[kindata_length 3 numpoints]);
set(gcf,'DoubleBuffer','on'); %prevents flicker
for frame = 1 : 2:kindata_length
    
    hold off;
    if ~isempty(cop)
        x1 = [cop(frame,1), cop(frame,1)+scale*grf(frame,1)];
        y1 = [cop(frame,2), cop(frame,2)+scale*grf(frame,2)];
        z1 = [cop(frame,3), cop(frame,3)+scale*grf(frame,3)];
        if grf(frame,2)>10 
            plot3(x1,y1,z1,'b');
        else
            plot3([-100 -100 -100],[-100 -100 -100],[-100 -100 -100]); 
        end
        hold on;
    end
    %axis manual; % freezes the axis, so that as you go through the animation the axis doesn't change
    %axis equal; %set axes equal in every direction


    x(1:numlinks)=kindata_stick(frame,1,1:numlinks);
    y(1:numlinks)=kindata_stick(frame,2,1:numlinks);
    z(1:numlinks)=kindata_stick(frame,3,1:numlinks);
    plot3(x,y,z, 'r','LineWidth',3);
    axis(ax);   %[-2 3 -1 1 0 2] % set the axis limit to include the range of animation

    %plot  points
    hold on;
        axis manual; % freezes the axis, so that as you go through the animation the axis doesn't change

    xp(1:numpoints)=points_stick(frame,1,1:numpoints);
    yp(1:numpoints)=points_stick(frame,2,1:numpoints);
    zp(1:numpoints)=points_stick(frame,3,1:numpoints);
    plot3(xp,yp,zp, 'b.','MarkerSize',20);

    view(vu);     %view  plane
%     set(gca,'CameraUpVector',  [0, 1, 0]);


    hold off;

    %xlabel('X');ylabel('Y');zlabel('Z');
    drawnow;
    %saveas(gcf,['c:\My Data\emus\emu' num2str(frame) '.jpg'],'jpg');
%     h_fig=gcf;
%    % F = getframe(h_fig,winsize);
%     F = getframe;%For clean plot without title and axes
%     mov = addframe(mov,F);
  pause(0.05);
end %for frame
set(gcf,'DoubleBuffer','off'); %ends speedy graphics
%mov=close(mov);