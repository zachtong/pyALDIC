function [mark3,mark4] = mark_edge(coordinates,elements4,ImgRefMask,elementMinSize)
%mark_edge: Marks all quadrilateral elements which intersect the edge geometry
%
% Author: Jin Yang, aldicdvc@gmail.com; jyang526@wisc.edu
% Date: 2020.11

%%
elementSize  = zeros(size(elements4,1),1);
grayscaleRange = zeros(size(elements4,1),1);
mark3 = []; % Not coded yet.

if ~isempty(elements4)
    
    for eleInd = 1:size(elements4,1)
        
        xCoordMin = min(coordinates(elements4(eleInd,1:4),1));
        xCoordMax = max(coordinates(elements4(eleInd,1:4),1));
    
        yCoordMin = min(coordinates(elements4(eleInd,1:4),2));
        yCoordMax = max(coordinates(elements4(eleInd,1:4),2));
    
        grayscaleMin = min(min(ImgRefMask( xCoordMin:xCoordMax,  yCoordMin:yCoordMax )));
        grayscaleMax = max(max(ImgRefMask( xCoordMin:xCoordMax,  yCoordMin:yCoordMax )));
        
        grayscaleRange(eleInd) = (grayscaleMax-grayscaleMin);
        elementSize(eleInd) = min( [ abs(xCoordMax-xCoordMin), abs(yCoordMax-yCoordMin) ] );
        
    end
    
    mark4 = grayscaleRange>0 & elementSize>elementMinSize;
    
    
        
else
    mark4 = [];
     
end

end