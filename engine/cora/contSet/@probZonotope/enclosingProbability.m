function [eP] = enclosingProbability(pZ,m,dimensions)
% enclosingProbability - Computes the enclosing probability of a 
% probabilistic zonotope
%
% Syntax:  
%    [eP] = enclosingProbability(pZ,m)
%
% Inputs:
%    pZ - probabilistic zonotope object
%    m  - m of the mSigma operator
%
% Outputs:
%    eP - enclosing probability
%
% Example: 
%
% Other m-files required: vertices, polytope
% Subfunctions: none

% MAT-files required: none
%
% See also: interval,  vertices

% Author:       Matthias Althoff
% Written:      08-August-2007
% Last update:  24-August-2007
%               27-August-2007
%               20-March-2015
% Last revision:---

%------------- BEGIN CODE --------------

gridpoints=20;

%get Sigma 
origSigma=sigma(pZ);
%project sigma
Sigma(1,1)=origSigma(dimensions(1),dimensions(1));
Sigma(1,2)=origSigma(dimensions(1),dimensions(2));
Sigma(2,1)=origSigma(dimensions(2),dimensions(1));
Sigma(2,2)=origSigma(dimensions(2),dimensions(2));

%determine mesh size by n-sigma hyperbox
Z=zonotope(pZ,m);
IH=interval(Z);
inf=infimum(IH);
sup=supremum(IH);
x=linspace(inf(dimensions(1)),sup(dimensions(1)),gridpoints);
y=linspace(inf(dimensions(2)),sup(dimensions(2)),gridpoints);

%initialize x-,y- and prob-vector for the mesh
ind=full_fact(1:length(x),1:length(y));
xVector=x(ind(:,1));
yVector=y(ind(:,2)); 
prob=0*xVector;

%check if center of probabilistic zonotope is uncertain
c=pZ.Z(:,1);
G=pZ.Z(:,2:end);
if isempty(G)
    prob=gaussian([xVector-c(1);yVector-c(2)],Sigma); 
else
    %get uncertain mean
    Z=zonotope(pZ.Z);
    %Compute potential vertices
    V=vertices(Z);
    %Extract projected dimensions
    Vprojected=V(dimensions,:);

    %Plot convex hull of projected vertices
    xPotential=Vprojected(1,:);
    yPotential=Vprojected(2,:);

    %determine vertex indices for convex hull vertices from potential
    %vertices 
    vertexIndices=convhull(xPotential,yPotential);

    %Select convex hull vertices
    xCoordinates=xPotential(vertexIndices);
    yCoordinates=yPotential(vertexIndices);

    %points for mean values
    points=[];
    for i=2:length(xCoordinates)
        newX=linspace(xCoordinates(i-1),xCoordinates(i),10);
        newY=linspace(yCoordinates(i-1),yCoordinates(i),10);
        points=[points,[newX;newY]];
    end

    %find inside points
    insidePoint=[];
    %get polytope of Z
    P=polytope(Vprojected');
    for i=1:length(xVector)
        if isinside(P,[xVector(i);yVector(i)])
            insidePoint=[insidePoint,[xVector(i);yVector(i)]];
        end
    end
    points=[points,insidePoint];

    %for each mean point
    for j=1:length(points(1,:))
        %get gaussian distribution from the current mean
        c=points(:,j);
        tempP=gaussian([xVector-c(1);yVector-c(2)],Sigma);
        %save maximum values
        prob=max(prob,tempP);
    end
end

%Build probability matrix from probability vector and save x and y values
eP.P=reshape(prob,length(y),length(x));
eP.X=reshape(xVector,length(y),length(x));
eP.Y=reshape(yVector,length(y),length(x));

%------------- END OF CODE --------------