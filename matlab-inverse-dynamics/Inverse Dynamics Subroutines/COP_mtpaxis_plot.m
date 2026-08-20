function COP_mtpaxis_plot(COP,Met5,Met1)

pause on
figure(100)
for i = 1:length(COP)
    plot(COP(:,3),COP(:,1))
    axis equal
    hold on
    plot(Met5(:,3),Met5(:,1))
    plot(Met1(:,3),Met1(:,1))
    plot(COP(i,3),COP(i,1),'o','MarkerFaceColor','red')
    % plot([Met5(i,3),Met1(i,3)],[Met5(i,1),Met1(i,1)])
    plot(Met5(i,3),Met5(i,1),'o','MarkerFaceColor','blue')
    plot(Met1(i,3),Met1(i,1),'o','MarkerFaceColor','green')
    ylabel('x axis')
    xlabel('z axis')
    legend('','','','COP','Met5','Met1')
    pause(0.5)
    clf
end

% for i = 1:length(COP)
%     plot([0 -0.6 -0.6 0 0], [0 0 0.9 0.9 0])
%     axis equal
%     hold on
%     plot(COP(:,3),COP(:,1))
%     plot(COP(i,3),COP(i,1),'o','MarkerFaceColor','red')
%     pause
%     clf
% end
