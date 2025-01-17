function completed = example_hybrid_reach_01_bouncingBall_zonoGirard
% example_hybrid_reach_01_bouncingBall_zonoGirard - example for hybrid
% dynamics. This example is also a unit test function. Girard's method is
% used to calculate the guard intersections.
%
% Checks the solution of the hybrid system class for the classical bouncing 
% ball example.
%
% Syntax:  
%    example_hybrid_reach_01_bouncingBall_zonoGirard
%
% Inputs:
%    no
%
% Outputs:
%    res - boolean 
%
% Example: 
%
% 
% Author:       Matthias Althoff
% Written:      27-July-2016
% Last update:  ---
% Last revision:---


%------------- BEGIN CODE --------------


%set options---------------------------------------------------------------
options.x0 = [1; 0]; %initial state for simulation
options.R0 = zonotope([options.x0, diag([0.05, 0.05])]); %initial state for reachability analysis
options.startLoc = 1; %initial location
options.finalLoc = 0; %0: no final location
options.tStart = 0; %start time
options.tFinal = 1.7; %final time
options.timeStepLoc{1} = 0.05; %time step size for reachable set computation in location 1
options.taylorTerms = 10;
options.zonotopeOrder = 20;
options.polytopeOrder = 10;
options.errorOrder=2;
options.reductionTechnique = 'girard';
options.isHyperplaneMap = 0;
options.guardIntersect = 'zonoGirard';
options.enclosureEnables = [1,2]; %choose enclosure method(s)
options.originContained = 0;
%--------------------------------------------------------------------------


%specify hybrid automaton--------------------------------------------------
%specify linear system of bouncing ball
A = [0 1; 0 0];
B = eye(2); % no loss of generality to specify B as the identity matrix
linSys = linearSys('linearSys',A,B);

%define large and small distance
dist = 1e3;
eps = 1e-6;
alpha = -0.75; %rebound factor

%invariant
inv = interval([0; -dist], [dist; dist]);
%guard sets
temp = halfspace([1,0],0); 
guard = constrainedHyperplane(temp,[0 1],0);
%resets
reset.A = [0, 0; 0, alpha]; reset.b = zeros(2,1);
%transitions
trans{1} = transition(guard,reset,1,'a','b'); %--> next loc: 1; 'a', 'b' are dummies
%specify location
loc{1} = location('loc1',1,inv,trans,linSys); 
%specify hybrid automata
HA = hybridAutomaton(loc); % for "geometric intersection"
%--------------------------------------------------------------------------

%set input:
options.uLoc{1} = [0; -9.81]; %input for simulation
options.uLocTrans{1} = options.uLoc{1}; %input center for reachability analysis
options.Uloc{1} = zonotope(zeros(2,1)); %input deviation for reachability analysis

%simulate hybrid automaton
HA = simulate(HA,options); 

%compute reachable set
[HA] = reach(HA,options);

%choose projection and plot------------------------------------------------
figure 
hold on
options.projectedDimensions = [1 2];
options.plotType = 'b';
plot(HA,'reachableSet',options); %plot reachable set
plotFilled(options.R0,options.projectedDimensions,'w','EdgeColor','k'); %plot initial set
plot(HA,'simulation',options); %plot simulation
axis([0,1.2,-6,4]);
%--------------------------------------------------------------------------

%example completed
completed = 1;

%------------- END OF CODE --------------
