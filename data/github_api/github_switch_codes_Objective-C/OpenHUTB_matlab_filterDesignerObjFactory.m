// Repository: OpenHUTB/matlab
// File: toolbox/signal/signal/+signal/+task/+internal/+designfilt/filterDesignerObjFactory.m

function fdesObj=filterDesignerObjFactory(response)




    switch response
    case{'lowpassfir','lowpassiir',...
        'highpassfir','highpassiir',...
        'bandpassfir','bandpassiir',...
        'bandstopfir','bandstopiir',...
        'differentiatorfir','hilbertfir'}
        fdesObj=filterBuilder('FromFilterDesigner','Response',response,'DoNotRenderGUI');
    otherwise
        error(message('signal:task:designfiltTask:designfiltTask:DesignObjectNotAvailable'));
    end