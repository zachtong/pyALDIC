function [markInside4,markOutside4] = mark_inside(coordinates,elements4,ImgRefMask)
%FUNCTION mark_inside: to mark all the quadrilateral elements within or outside
%the center hole or other boundary edges
%
% Author: Jin Yang, aldicdvc@gmail.com; jyang526@wisc.edu
% Date: 2020.11
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

subsetInside4 = zeros(size(elements4,1),1);
if ~isempty(elements4)
    
    for eleInd = 1:size(elements4,1)
        
        
        xCoordMin = min(coordinates(elements4(eleInd,1:4),1));
        xCoordMax = max(coordinates(elements4(eleInd,1:4),1));
    
        yCoordMin = min(coordinates(elements4(eleInd,1:4),2));
        yCoordMax = max(coordinates(elements4(eleInd,1:4),2));
        
        % If more than 50% subset is the hole, it's inside the hole
        if sum(sum(ImgRefMask( xCoordMin:xCoordMax,  yCoordMin:yCoordMax ))) < 0.5*(xCoordMax-xCoordMin)*(yCoordMax-yCoordMin)
            subsetInside4(eleInd) = 1;
        end
        
    end
    
    markInside4 = find(subsetInside4 > 0);
    markOutside4 = setdiff([1:1:size(elements4,1)]', markInside4);
    
else
   markInside4 = [];
   markOutside4 = [];
   
end
 
