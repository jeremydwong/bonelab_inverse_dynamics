function COP_mtpaxis_plot_MotekTest(COP,Met5,Met1)

pause on
figure(100)
for i = 1:length(COP)
    plot(COP(:,1),COP(:,2))
    axis equal
    hold on
    plot(Met5(:,1),Met5(:,2))
    plot(Met1(:,1),Met1(:,2))
    plot(COP(i,1),COP(i,2),'o','MarkerFaceColor','red')
    plot([Met5(i,1),Met1(i,1)],[Met5(i,2),Met1(i,2)])
    plot(Met5(i,1),Met5(i,2),'o','MarkerFaceColor','blue')
    plot(Met1(i,1),Met1(i,2),'o','MarkerFaceColor','green')
    ylabel('x axis')
    xlabel('z axis')
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
