function save_fig_stress(file_name, ImgSeqNum, DICpara)
% Save figures or output images for solved stress fields
%
% Inputs:
%   file_name  - cell array of image file names
%   ImgSeqNum  - current image sequence number
%   DICpara    - parameter struct (.MethodToSaveFig, .OrigDICImgTransparency,
%                .winsize, .winstepsize)

%%
% Find img name
[~,imgname,imgext] = fileparts(file_name{1,ImgSeqNum});

%%
if DICpara.MethodToSaveFig == 1
    %% jpg
    figure(1); if DICpara.OrigDICImgTransparency == 0, colormap jet; caxis([-0.025,0.025]); end
    print([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_sxx'],'-djpeg','-r300')
    
    figure(2); if DICpara.OrigDICImgTransparency == 0, colormap jet; caxis([-0.025,0.025]); end
    print([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_sxy'],'-djpeg','-r300')
    
    figure(3); if DICpara.OrigDICImgTransparency == 0, colormap jet; caxis([-0.015,0.015]); end
    print([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_syy'],'-djpeg','-r300')
    
    figure(4); if DICpara.OrigDICImgTransparency == 0, colormap jet;  caxis([0,0.025]); end
    print([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_principal_max_xyplane'],'-djpeg','-r300')
    
    figure(5); if DICpara.OrigDICImgTransparency == 0, colormap jet;  caxis([-0.025,0]); end
    print([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_principal_min_xyplane'],'-djpeg','-r300')
    
    figure(6); if DICpara.OrigDICImgTransparency == 0, colormap jet;  caxis([0,0.07]); end
    print([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_maxshear_xyplane'],'-djpeg','-r300')
    
    figure(7); if DICpara.OrigDICImgTransparency == 0, colormap jet;  caxis([0,0.07]); end
    print([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_maxshear_xyz3d'],'-djpeg','-r300')
    
    figure(8); if DICpara.OrigDICImgTransparency == 0, colormap jet;  caxis([0,0.07]); end
    print([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_vonMises'],'-djpeg','-r300')
    
    
elseif DICpara.MethodToSaveFig == 2
    %% pdf
    filename = [imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_sxx'];
    figure(1); if DICpara.OrigDICImgTransparency == 0, colormap coolwarm(32); caxis([-0.025,0.025]); end
    export_fig( gcf , '-pdf' , '-r300' , '-painters' , filename);
    
    filename = [imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_sxy'];
    figure(2); if DICpara.OrigDICImgTransparency == 0, colormap coolwarm(32); caxis([-0.025,0.025]); end
    export_fig( gcf , '-pdf' , '-r300' , '-painters' , filename);
    
    filename = [imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_syy'];
    figure(3); if DICpara.OrigDICImgTransparency == 0, colormap coolwarm(32); caxis([-0.015,0.015]); end
    export_fig( gcf , '-pdf' , '-r300' , '-painters' , filename);
    
    filename = [imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_principal_max_xyplane'];
    figure(4); if DICpara.OrigDICImgTransparency == 0, colormap coolwarm(32); caxis([0,0.025]); end
    export_fig( gcf , '-pdf' , '-r300' , '-painters' , filename);
    
    filename = [imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_principal_min_xyplane'];
    figure(5); if DICpara.OrigDICImgTransparency == 0, colormap coolwarm(32); caxis([-0.025,0]); end
    export_fig( gcf , '-pdf' , '-r300' , '-painters' , filename);
    
    filename = [imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_maxshear_xyplane'];
    figure(6); if DICpara.OrigDICImgTransparency == 0, colormap coolwarm(32); caxis([0,0.07]); end
    export_fig( gcf , '-pdf' , '-r300' , '-painters' , filename);
    
    filename = [imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_maxshear_xyz3d'];
    figure(7); if DICpara.OrigDICImgTransparency == 0, colormap coolwarm(32); caxis([0,0.07]); end
    export_fig( gcf , '-pdf' , '-r300' , '-painters' , filename);
    
    filename = [imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_vonMises'];
    figure(8); if DICpara.OrigDICImgTransparency == 0, colormap coolwarm(32); caxis([0,0.07]); end
    export_fig( gcf , '-pdf' , '-r300' , '-painters' , filename);
    
    
else
    %% fig
    fprintf('Please modify codes manually in Section 8.');
    figure(1); colormap(coolwarm(128)); caxis([-0.05,0.1]); 
        savefig([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_sxx.fig']);
    figure(2); colormap(coolwarm(128)); caxis([-0.05,0.05]); 
        savefig([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_sxy.fig']);
    figure(3); colormap(coolwarm(128)); caxis([-0.1,0.05]); 
        savefig([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_syy.fig']);
    figure(4); colormap(coolwarm(128)); caxis([0,0.025]); 
        savefig([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_principal_max_xyplane.fig']);
    figure(5); colormap(coolwarm(128)); caxis([-0.025,0]); 
        savefig([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_principal_min_xyplane.fig']);
    figure(6); colormap(coolwarm(128)); caxis([0,0.07]); 
        savefig([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_maxshear_xyplane.fig']);
    figure(7); colormap(coolwarm(128)); caxis([0,0.07]); 
        savefig([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_maxshear_xyz3d.fig']);
    figure(8); colormap(coolwarm(128)); caxis([0,0.07]); 
        savefig([imgname,'_WS',num2str(DICpara.winsize),'_ST',num2str(DICpara.winstepsize),'_stress_vonMises.fig']);
    
    
end





