function h=show(elements3,elements4,coordinates,u,varargin)

switch nargin
    case 5
        edgeColorOrNot = varargin{1};
    otherwise
end
try
    temp = strcmp(edgeColorOrNot,'NoEdgeColor')==1;
catch
    edgeColorOrNot = 'EdgeColor';
end

% trisurf(elements3,coordinates(:,1),coordinates(:,2),u',...
% 'facecolor','interp' )
% hold on
Trix = zeros(3,size(elements3,1)); Triy = zeros(3,size(elements3,1)); Tric = zeros(3,size(elements3,1));
for j = 1:size(elements3,1)
    Trix(1:3,j) = coordinates(elements3(j,1:3),1);
    Triy(1:3,j) = coordinates(elements3(j,1:3),2);
    Tric(1:3,j) = u(elements3(j,1:3));
end
if size(elements3,1) > 2e4 || strcmp(edgeColorOrNot,'NoEdgeColor')==1
    h=patch(Trix,Triy,Tric,'facecolor','interp','edgecolor','none');
else
    h=patch(Trix,Triy,Tric,'facecolor','interp');
end

hold on;
% trisurf(elements4,coordinates(:,1),coordinates(:,2),u',...
% 'facecolor','interp' )


Sqx = zeros(4,size(elements4,1)); Sqy = zeros(4,size(elements4,1)); Sqc = zeros(4,size(elements4,1));
for j = 1:size(elements4,1)
    Sqx(1:4,j) = coordinates(elements4(j,1:4),1);
    Sqy(1:4,j) = coordinates(elements4(j,1:4),2);
    Sqc(1:4,j) = u(elements4(j,1:4));
end
if size(elements4,1) > 2e4 || strcmp(edgeColorOrNot,'NoEdgeColor')==1
    h=patch(Sqx,Sqy,Sqc,'facecolor','interp','edgecolor','none');
else
    h=patch(Sqx,Sqy,Sqc,'facecolor','interp');
end

hold off
view(10,40);
% title('Solution of the Problem')
box on
