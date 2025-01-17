function [error] = linError(obj,options,R)
% linError - computes the linearization error
%
% Syntax:  
%    [obj] = linError(obj,options)
%
% Inputs:
%    obj - nonlinear system object
%    options - options struct
%    R - actual reachable set
%
% Outputs:
%    error - linearization error
%
% Example: 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: 
%
% References: 
%   [1] M. Althoff et al. "Reachability Analysis of Nonlinear Systems with 
%       Uncertain Parameters using Conservative Linearization"

% Author:       Matthias Althoff, Niklas Kochdumper
% Written:      29-October-2007 
% Last update:  22-January-2008
%               02-February-2010
%               25-July-2016 (intervalhull replaced by interval)
%               12-November-2018 (NK: changed method for remainder
%                                 over-approximation)
% Last revision: ---

%------------- BEGIN CODE --------------

% compute interval of reachable set
IH=interval(R);

% compute intervals of total reachable set
totalInt=interval(IH) + obj.linError.p.x;

% compute intervals of input
if isa(options.U,'interval')
    IHinput=options.U + options.uTrans;
else
    IHinput=interval(options.U) + options.uTrans;
end
inputInt=interval(IHinput);

% translate intervals by linearization point
IHinput=IHinput + (-obj.linError.p.u);

% obtain maximum absolute values within IH, IHinput
IHinf=abs(infimum(IH));
IHsup=abs(supremum(IH));
dx=max(IHinf,IHsup);

IHinputInf=abs(infimum(IHinput));
IHinputSup=abs(supremum(IHinput));
du=max(IHinputInf,IHinputSup);

% evaluate the hessian matrix with the selected range-bounding technique
if isfield(options,'lagrangeRem') && isfield(options.lagrangeRem,'method') && ...
   ~strcmp(options.lagrangeRem.method,'interval')

    % create taylor models or zoo-objects
    [objX,objU] = initRangeBoundingObjects(totalInt,inputInt,options);

    % evaluate the Lagrane remainder 
    H = obj.hessian(objX,objU,options.paramInt);
else
    H = obj.hessian(totalInt,inputInt,options.paramInt);
end

% compute an over-approximation of the Lagrange remainder according to
% Proposition 1 in [1]
error = zeros(length(H),1);
dz = [dx;du];

for i = 1:length(H)
    H_ = abs(H{i});
    H_ = max(infimum(H_),supremum(H_));
    error(i) = 0.5 * dz' * H_ * dz;
end

%------------- END OF CODE --------------