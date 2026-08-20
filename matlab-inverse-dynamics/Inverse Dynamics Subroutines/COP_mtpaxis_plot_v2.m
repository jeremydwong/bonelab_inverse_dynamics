function COP_mtpaxis_plot_v2(COP,Met5,Met1,Toe,Heel)

pause on
figure(100)

for i = 1:length(COP)
    plot(COP(:,3),COP(:,1))
    ylim([min(Heel(:,1))-0.1 max(Toe(:,1))+0.1]);
    xlim([min(min([Met5(:,3) Met1(:,3) Toe(:,3) Heel(:,3)]))-0.1 max(max([Met5(:,3) Met1(:,3) Toe(:,3) Heel(:,3)]))+0.1]);
    axis equal
    hold on
    plot(Met5(:,3),Met5(:,1))
    plot(Met1(:,3),Met1(:,1))
    plot(COP(i,3),COP(i,1),'o','MarkerFaceColor','red')
    % plot([Met5(i,3),Met1(i,3)],[Met5(i,1),Met1(i,1)])
    plot(Met5(i,3),Met5(i,1),'o','MarkerFaceColor','blue')
    plot(Met1(i,3),Met1(i,1),'o','MarkerFaceColor','green')
    plot(Toe(i,3),Toe(i,1),'o','MarkerFaceColor','cyan')
    plot(Heel(i,3),Heel(i,1),'o','MarkerFaceColor','black')
    ylabel('x axis')
    xlabel('z axis')
    legend('','','','COP','Met5','Met1','Toe','Heel')
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
