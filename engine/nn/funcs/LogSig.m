classdef LogSig
    % LOGSIG Class contains methods for reachability analysis of layer with
    % Sigmoid activation function.
    % Reference: https://www.mathworks.com/help/deeplearning/ref/logsig.html
    % Author: Dung Tran
    % Date: 28/2/2019
    
    properties
        
    end
    
    methods(Static)  % evaluate method and over-approximate reachability analysis with stars
        
        % evaluation
        function y = evaluate(x)
            y = logsig(x);
        end
        
        
        % main method
        function S = reach_star_approx(varargin)
            % author: Dung Tran
            % date: 3/19/2020
            
            switch nargin
                case 1
                    I = varargin{1};
                    method = 'approx-star-no-split';
                    reachOption = [];
                    relaxFactor = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = [];
                    relaxFactor = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = varargin{3};
                    relaxFactor = 0;
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 4
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 5
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = varargin{5};
                    lp_solver = 'glpk';
                case 6
                    I = varargin{1};
                    method = varargin{2};
                    reachOption = varargin{3};
                    relaxFactor = varargin{4};
                    dis_opt = varargin{5};
                    lp_solver = varargin{6};
                otherwise
                    error('Invalid number of input arguments, should be 1, 2, 3, 4, 5 or 6');
            end
            
            if ~isa(I, 'Star')
                error('Input set is not a star set');
            end
           
            if strcmp(method, 'approx-star-no-split') || strcmp(method, 'approx-star')
                if relaxFactor == 0
                    S = LogSig.reach_star_approx_no_split(I, dis_opt, lp_solver);
                else
                    S = LogSig.relaxedMultiStepLogSig_NoSplit(I, relaxFactor, dis_opt, lp_solver);
                end
            elseif strcmp(method, 'approx-star-split')
                S = LogSig.reach_star_approx_split(I);
            else
                error('Unknown reachability method');
            end
            
        end
        
        
        % reachability method with star
        function S = reach_star_approx_no_split(varargin)
            % @I: the input star set
            % @S: a star set output
            
            % author: Dung Tran
            % date: 3/19/2020
            % update:4/2/2020, 6/12/2020
            % update: 7/15/2020: add display option
            %         7/16/2020: add lp_solver option
            
            switch nargin
                case 1
                    I = varargin{1};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    I = varargin{1};
                    dis_opt = varargin{2};
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    dis_opt = varargin{2};
                    lp_solver = varargin{3};
                otherwise
                    error('Invalid number of input arguments, should be 1 or 2');
            end
            S = LogSig.multiStepLogSig_NoSplit(I, dis_opt, lp_solver);      % more efficient method compared with stepLogSig_NoSplit       
%             n = I.dim;
%             S = I;
%             for i=1:n
%                 S = LogSig.stepLogSig_NoSplit(S, i); 
%             end
                        
        end
        
        % reachability method with star
        function S = reach_star_approx_split(I)
            % @I: the input star set
            % @S: an array of star set output
            
            % author: Dung Tran
            % date: 3/19/2020
            % update:4/2/2020
           
            n = I.dim;
            S = I;
            for i=1:n
                m = length(S);
                O = [];
                for j=1:m
                    O = [O LogSig.stepLogSig_Split(S(j), i)];
                end
                S = O;
            end
       
        end
        
        % stepLogSig
        function S = stepLogSig_Split(I, index)
            % @I: input star set
            % @index: index of the neuron
            
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @y_l: = logsig(l); output of logsig at lower bound
            % @y_u: = logsig(u); output of logsig at upper bound
            % @dy_l: derivative of LogSig at the lower bound
            % @dy_u: derivative of LogSig at the upper bound
            
            % @S: output star set
            
            % author: Dung Tran
            % date: 3/19/2020
            % update: 4/2/2020
            
            %[l, u] = I.Z.getRange(index);
            % Sung Woo change for temp check
            [l, u] = I.getRange(index);
            y_l = logsig(l);
            y_u = logsig(u);
            dy_l = logsig('dn', l);
            dy_u = logsig('dn', u);
            
            if l == u
               
                new_V = I.V;
                new_V(index, 1:I.nVar+1) = 0;
                new_V(index, 1) = y_l;
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;
                    c(index) = y_l;
                    V(index, :) = 0;
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z); 
               
            elseif l > 0
                % y is convex when x >= 0
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                % constarint 2: y <= y'(u) * (x - u) + y(u) 
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);


                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                C1 = [-dy_l*I.V(index, 2:n) 1];
                d1 = dy_l * I.V(index, 1) - dy_l*l + y_l; 
                % constraint 2: y <= y'(u) * (x - u) + y(u)
                C2 = [-dy_u*I.V(index, 2:n) 1];
                d2 = dy_u * I.V(index, 1) - dy_u*u + y_u;
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                a = (y_u - y_l)/(u - l);
                C3 = [a*I.V(index, 2:n) -1];
                d3 = a*l - y_l - a*I.V(index, 1);

                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];

                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_l, dy_u);
                    mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                    mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end

                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);

            elseif u <= 0
                % y is concave when x <= 0
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                % constraint 2: y >= y'(u) * (x - u) + y(u)
                % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);

                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C1 = [dy_l*I.V(index, 2:n) -1];
                d1 = -dy_l * I.V(index, 1) + dy_l*l - y_l; 
                % constraint 2: y >= y'(u) * (x - u) + y(u)
                C2 = [dy_u*I.V(index, 2:n) -1];
                d2 = -dy_u * I.V(index, 1) + dy_u*u - y_u;
                % constraint 3: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                a = (y_u - y_l)/(u - l);
                C3 = [-a*I.V(index, 2:n) 1];
                d3 = -a*l + y_l + a*I.V(index, 1);

                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_l, dy_u);
                    mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                    mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end

                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);

            elseif l <= 0 && u >0
                % y is concave for x in [l, 0] and convex for x
                % in [0, u]
                % split can be done here 

                % case 1: x in [l, 0]
                % y'(0) = 0.25
                % y is concave when x <= 0
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                % constraint 2: y >= y'(0) * (x) + y(0)
                % constraint 3: y <= (y(0) - y(l)) * (x -l) / (0 - l) + y(l);

                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C1 = [dy_l*I.V(index, 2:n) -1];
                d1 = -dy_l * I.V(index, 1) + dy_l*l - y_l; 
                % constraint 2: y >= y'(0) * (x - 0) + y(0) = 0.25*x + 0.5
                C2 = [0.25*I.V(index, 2:n) -1];
                d2 = -0.25 * I.V(index, 1) - 0.5;
                % constraint 3: y <= (y(0) - y(l)) * (x - l) / (0 - l) + y(l);
                a = (0.5 - y_l)/(0 - l);
                C3 = [-a*I.V(index, 2:n) 1];
                d3 = -a*l + y_l + a*I.V(index, 1);

                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; 0.5];
                
                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_l, 0.25);
                    mu1 = 0.5*(0.5 + y_l - lamda *(0 + l));
                    mu2 = 0.5*(0.5 - y_l - lamda *(0 - l));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                
                S1 = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);

                % case 2: x in [0, u] 
                % y is convex when x >= 0
                % constraint 1: y <= y'(0) * (x - 0) + y(0) = 0.25*x + 0.5
                % constarint 2: y <= y'(u) * (x - u) + y(u) 
                % constraint 3: y >= (y(u) - y(0)) * (x - 0) / (u - 0) + y(0);

                % over-approximation constraints 
                % constraint 1: y <= y'(0) * (x - 0) + y(0) = 0.25*x + 0.5
                C1 = [-0.25*I.V(index, 2:n) 1];
                d1 = 0.25 * I.V(index, 1) + 0.5; 
                % constraint 2: y <= y'(u) * (x - u) + y(u)
                C2 = [-dy_u*I.V(index, 2:n) 1];
                d2 = dy_u * I.V(index, 1) - dy_u*u + y_u;
                % constraint 3: y >= (y(u) - y(0)) * (x - 0) / (u - 0) + y(0);
                a = (y_u - 0.5)/u;
                C3 = [a*I.V(index, 2:n) -1];
                d3 = -0.5 - a*I.V(index, 1);

                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 

                % update predicate bound
                new_predicate_lb = [I.predicate_lb; 0.5]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                
                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_u, 0.25);
                    mu1 = 0.5*(y_u + 0.5 - lamda *(u + 0));
                    mu2 = 0.5*(y_u - 0.5 - lamda *(u - 0));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                
                S2 = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);

                S = [S1 S2];
            end
            
               

        end
        
        % stepLogSig
        function S = stepLogSig_NoSplit(I, index)
            % @I: input star set
            % @index: index of the neuron
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @y_l: = logsig(l); output of logsig at lower bound
            % @y_u: = logsig(u); output of logsig at upper bound
            % @dy_l: derivative of LogSig at the lower bound
            % @dy_u: derivative of LogSig at the upper bound
            
            % @S: output star set
            
            % author: Dung Tran
            % date: 3/19/2020
            
            
%             
%             if ~isempty(I.Z) && I.nVar > 200
%                 [l, u] = I.Z.getRange(index);
%             else
%                 [l, u] = I.getRange(index);
%             end
%           
            fprintf('\nStepLogSig(Sigmoid)_NoSplit using approx-star at index %d', index);
            [l, u] = I.estimateRange(index);
            fprintf('\nEstimated Range at x[%d]:l = %.5f, u = %.5f', index, l, u);
            [l, u] = I.getRange(index);
            fprintf('\nComputed Range at x[%d]: l = %.5f, u = %.5f', index, l, u);           
            y_l = logsig(l);
            y_u = logsig(u);
            dy_l = logsig('dn', l);
            dy_u = logsig('dn', u);
            fprintf('\ny_l = %.5f, y_u = %.5f, dy_l = %.5f, dy_u = %.5f', y_l, y_u, dy_l, dy_u);
            
            if l == u
                
                new_V = I.V;
                new_V(index, 1:I.nVar+1) = 0;
                new_V(index, 1) = y_l;
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;
                    c(index) = y_l;
                    V(index, :) = 0;
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z);
                
            elseif l > 0
                % y is convex when x >= 0
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                % constarint 2: y <= y'(u) * (x - u) + y(u) 
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                
                
                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                C1 = [-dy_l*I.V(index, 2:n) 1];
                d1 = dy_l * I.V(index, 1) - dy_l*l + y_l; 
                % constraint 2: y <= y'(u) * (x - u) + y(u)
                C2 = [-dy_u*I.V(index, 2:n) 1];
                d2 = dy_u * I.V(index, 1) - dy_u*u + y_u;
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                a = (y_u - y_l)/(u - l);
                C3 = [a*I.V(index, 2:n) -1];
                d3 = a*l - y_l - a*I.V(index, 1);
                
                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 
                
                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                
                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_l, dy_u);
                    mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                    mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                
                
                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);             
                
            elseif u <= 0
                % y is concave when x <= 0
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                % constraint 2: y >= y'(u) * (x - u) + y(u)
                % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
                
                n = I.nVar + 1;
                % over-approximation constraints 
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C1 = [dy_l*I.V(index, 2:n) -1];
                d1 = -dy_l * I.V(index, 1) + dy_l*l - y_l; 
                % constraint 2: y >= y'(u) * (x - u) + y(u)
                C2 = [dy_u*I.V(index, 2:n) -1];
                d2 = -dy_u * I.V(index, 1) + dy_u*u - y_u;
                % constraint 3: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                a = (y_u - y_l)/(u - l);
                C3 = [-a*I.V(index, 2:n) 1];
                d3 = -a*l + y_l + a*I.V(index, 1);
                
                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3];
                new_d = [d0; d1; d2; d3];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 
                
                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                
                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_l, dy_u);
                    mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                    mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end
                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);

            elseif l <= 0 && u >0
                % y is concave for x in [l, 0] and convex for x
                % in [0, u]
                % split can be done here 
                
                % combine two stars into one star               
                x1 = (0.5 + dy_u * u - y_u)/(dy_u - 0.25);
                y1 = 0.25 * x1 + 0.5;
                x2 = (0.5 + dy_l * l - y_l)/(dy_l - 0.25);
                y2 = 0.25 * x2 + 0.5; 
                
                % over-approximation constraints 
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                % constraint 2: y <= y'(u) * (x - u) + y(u)
                % constraint 3: y <= (y(x1) - y(l))*(x - l)/(x1 - l) + y(l);
                % constraint 4: y >= (y(x2) - y(u)) * (x - u)/(x2 - u) + y(u)
                
                               
                n = I.nVar + 1;
                
                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C1 = [dy_l*I.V(index, 2:n) -1];
                d1 = -dy_l * I.V(index, 1) + dy_l*l - y_l; 
                % constraint 2: y <= y'(u) * (x - u) + y(u)
                C2 = [-dy_u*I.V(index, 2:n) 1];
                d2 = dy_u * I.V(index, 1) - dy_u*u + y_u;
                % constraint 3: y <= (y(x1) - y(l))*(x - l)/(x1 - l) + y(l);
                a = (y1 - y_l)/(x1 - l);
                C3 = [-a*I.V(index, 2:n) 1];
                d3 = -a*l + y_l + a*I.V(index, 1);
                                
                % constraint 4: y >= (y(x2) - y(u)) * (x - u)/(x2 - u) + y(u);
                a = (y2 - y_u)/(x2 - u);
                C4 = [a*I.V(index, 2:n) -1];
                d4 = a*u - y_u - a*I.V(index, 1);
                
                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_C = [C0; C1; C2; C3; C4];
                new_d = [d0; d1; d2; d3; d4];
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1; 
                
                % update predicate bound
                new_predicate_lb = [I.predicate_lb; y_l]; 
                new_predicate_ub = [I.predicate_ub; y_u];
                
                % update outer-zonotope
                if ~isempty(I.Z)
                    c = I.Z.c;
                    V = I.Z.V;                
                    lamda = min(dy_l, dy_u);
                    mu1 = 0.5*(y_u + y_l - lamda *(u + l));
                    mu2 = 0.5*(y_u - y_l - lamda *(u - l));
                    c(index) = lamda * c(index) + mu1;
                    V(index, :) = lamda * V(index, :); 
                    I1 = zeros(I.dim, 1);
                    I1(index) = mu2;
                    V = [V I1];
                    new_Z = Zono(c, V);
                else
                    new_Z = [];
                end

                S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);
                                
            end

        end
        
        
        % multistepLogSig
        function S = multiStepLogSig_NoSplit(varargin)
            % @I: input star set
            
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @yl: = logsig(l); output of logsig at lower bound
            % @yu: = logsig(u); output of logsig at upper bound
            % @dyl: derivative of LogSig at the lower bound
            % @dyu: derivative of LogSig at the upper bound
            
            % @S: output star set
            
            % author: Dung Tran
            % date: 6/12/2020
            % update: 7/15/2020: add display option 
            %         7/16/2020: add lp_solver option
            switch nargin
                case 1
                    I = varargin{1};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 2
                    I = varargin{1};
                    dis_opt = varargin{2};
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    dis_opt = varargin{2};
                    lp_solver = varargin{3};
                otherwise
                    error('Invalid number of input arguments, should be 1, 2 or 3');
            end

            
           %[l, u] = I.estimateRanges;
            N = I.dim;
            inds = 1:N;
            if strcmp(dis_opt, 'display')
                fprintf('\nComputing lower-bounds: ');
            end
            l = I.getMins(inds, [], dis_opt, lp_solver);
            if strcmp(dis_opt, 'display')
                fprintf('\nComputing upper-bounds: ');  
            end
            u = I.getMaxs(inds, [], dis_opt, lp_solver);
            yl = logsig(l);
            yu = logsig(u);
            dyl = logsig('dn', l);
            dyu = logsig('dn', u);

           

            % l ~= u
            map2 = find(l ~= u);
            m = length(map2);
            V2 = zeros(N, m);
            for i=1:m
                V2(map2(i), i) = 1;
            end

            % new basis matrix
            new_V = [zeros(N, I.nVar+1) V2];
            
             % l == u
            map1 = find(l == u);
            yl1 = yl(map1(:));         
            new_V(map1, 1) = yl1;
            new_V(map1, 2:I.nVar+1+m) = 0;

            % add new constraints

            % C0, d0
            n = size(I.C, 1);
            C0 = [I.C zeros(n, m)];
            d0 = I.d;

            nv = I.nVar+1;

            % C1, d1, x >= 0
            % y is convex when x >= 0
            % constraint 1: y <= y'(l) * (x - l) + y(l)
            % constarint 2: y <= y'(u) * (x - u) + y(u) 
            % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            map1 = find(l >= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                C11 = [-da.*I.V(map1, 2:nv) V2(map1, :)];
                d11 = da.*(I.V(map1, 1)-l(map1)) + a;
                % constraint 2: y <= y'(u) * (x - u) + y(u) 
                C12 = [-db.*I.V(map1, 2:nv) V2(map1, :)];
                d12 = db.*(I.V(map1, 1) - u(map1)) + b;
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C13 = [gamma.*I.V(map1, 2:nv) -V2(map1, :)];
                d13 = -gamma.*(I.V(map1, 1)-l(map1)) - a;

                C1 = [C11; C12; C13]; 
                d1 = [d11; d12; d13];
            else
                C1 = [];
                d1 = [];                
            end
            

            % C2, d2, x <= 0 
            % y is concave when x <= 0
            % constraint 1: y >= y'(l) * (x - l) + y(l)
            % constraint 2: y >= y'(u) * (x - u) + y(u)
            % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);

            map1 = find(u <= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);

                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C21 = [da.*I.V(map1, 2:nv) -V2(map1, :)];
                d21 = -da.*(I.V(map1, 1)-l(map1)) - a;
                % constraint 2: y >= y'(u) * (x - u) + y(u) 
                C22 = [db.*I.V(map1, 2:nv) -V2(map1, :)];
                d22 = -db.*(I.V(map1, 1) - u(map1)) - b;
                % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C23 = [-gamma.*I.V(map1, 2:nv) V2(map1, :)];
                d23 = gamma.*(I.V(map1, 1)-l(map1)) + a;

                C2 = [C21; C22; C23]; 
                d2 = [d21; d22; d23];
            else
                C2 = [];
                d2 = [];
            end
            
            % C3, d3, l< 0 & u > 0, x >0 or x < 0
            %y is concave for x in [l, 0] and convex for x
            % in [0, u]
            % split can be done here            

            map1 = find(l < 0 & u > 0);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);

                dmin = (min(da', db'))';
                % over-approximation constraints 
                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u)

                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                C31 = [dmin.*I.V(map1, 2:nv) -V2(map1, :)];
                d31 = -dmin.*(I.V(map1, 1)-l(map1)) - a;
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
                C32 = [-dmin.*I.V(map1, 2:nv) V2(map1, :)];
                d32 = dmin.*(I.V(map1, 1) - u(map1)) + b;


                y1 = dmin.*(-l(map1)) + a;
                y2 = dmin.*(-u(map1)) + b;
                g2 = (y2 - a)./(-l(map1));
                g1 = (y1 - b)./(-u(map1));

                % constraint 3: y <= g2 * x + y2
                C33 = [-g2.*I.V(map1, 2:nv) V2(map1, :)];
                d33 = g2.*I.V(map1, 1) + y2;

                % constraint 4: y >= g1 * x + y1
                C34 = [g1.*I.V(map1, 2:nv) -V2(map1, :)];
                d34 = -g1.*I.V(map1, 1) - y1;

                C3 = [C31; C32; C33; C34]; 
                d3 = [d31; d32; d33; d34];
            else
                C3 = [];
                d3 = [];
            end
            
            new_C = [C0; C1; C2; C3];
            new_d = [d0; d1; d2; d3]; 

            new_pred_lb = [I.predicate_lb; yl(map2)];
            new_pred_ub = [I.predicate_ub; yu(map2)];

            S = Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub);
        end
        
        
        % multistepLogSig
        function S = relaxedMultiStepLogSig_NoSplit(varargin)
            % @I: input star set
            % @relaxFactor: percentage of optimized ranges used for
            % constructing reachable set
            
            % @l: l = min(x[index]), lower bound at neuron x[index] 
            % @u: u = min(x[index]), upper bound at neuron x[index]
            % @yl: = logsig(l); output of logsig at lower bound
            % @yu: = logsig(u); output of logsig at upper bound
            % @dyl: derivative of LogSig at the lower bound
            % @dyu: derivative of LogSig at the upper bound
            
            % @S: output star set
            
            % author: Dung Tran
            % date: 6/12/2020
            % update: 7/15/2020 add display option
            
            switch nargin
                case 2 
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    dis_opt = [];
                    lp_solver = 'glpk';
                case 3
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = 'glpk';
                case 4
                    I = varargin{1};
                    relaxFactor = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = varargin{4};
                otherwise
                    error('Invalid number of input arguments, should be 2 or 3');
            end
            
            if ~isa(I, 'Star')
                error('Input is not a star');
            end
            if (relaxFactor < 0) || (relaxFactor > 1)
                error('Invalid relax factor');
            end
            
            [l, u] = I.estimateRanges;
            n1 = round((1-relaxFactor)*length(l));
            [~, midx] = sort(u - l, 'descend');
            
            N = I.dim;
            if strcmp(dis_opt, 'display')
                fprintf('\nComputing (1-%.3f) x %d = %d lower-bounds, i.e. relaxing %2.2f%%: ' , relaxFactor, length(l), n1, 100*relaxFactor);
            end
            l2 = I.getMins(midx(1:n1), [], dis_opt, lp_solver);
            if strcmp(dis_opt, 'display')
                fprintf('\nComputing (1-%.3f) x %d = %d upper-bounds, i.e. relaxing %2.2f%%: ' , relaxFactor, length(l), n1, 100*relaxFactor);
            end
            u2 = I.getMaxs(midx(1:n1), [], dis_opt, lp_solver);
            l(midx(1:n1)) = l2;
            u(midx(1:n1)) = u2;
            
            yl = logsig(l);
            yu = logsig(u);
            dyl = logsig('dn', l);
            dyu = logsig('dn', u);

            % l ~= u
            map2 = find(l ~= u);
            m = length(map2);
            V2 = zeros(N, m);
            for i=1:m
                V2(map2(i), i) = 1;
            end

            % new basis matrix
            new_V = [zeros(N, I.nVar+1) V2];
            
             % l == u
            map1 = find(l == u);
            yl1 = yl(map1(:));         
            new_V(map1, 1) = yl1;
            new_V(map1, 2:I.nVar+1+m) = 0;

            % add new constraints

            % C0, d0
            n = size(I.C, 1);
            C0 = [I.C zeros(n, m)];
            d0 = I.d;
            
            nv = I.nVar+1;

            % C1, d1, x >= 0
            % y is convex when x >= 0
            % constraint 1: y <= y'(l) * (x - l) + y(l)
            % constarint 2: y <= y'(u) * (x - u) + y(u) 
            % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            map1 = find(l >= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);
                % constraint 1: y <= y'(l) * (x - l) + y(l)
                C11 = [-da.*I.V(map1, 2:nv) V2(map1, :)];
                d11 = da.*(I.V(map1, 1)-l(map1)) + a;
                % constraint 2: y <= y'(u) * (x - u) + y(u) 
                C12 = [-db.*I.V(map1, 2:nv) V2(map1, :)];
                d12 = db.*(I.V(map1, 1) - u(map1)) + b;
                % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C13 = [gamma.*I.V(map1, 2:nv) -V2(map1, :)];
                d13 = -gamma.*(I.V(map1, 1)-l(map1)) - a;

                C1 = [C11; C12; C13]; 
                d1 = [d11; d12; d13];
            else
                C1 = [];
                d1 = [];                
            end
            

            % C2, d2, x <= 0 
            % y is concave when x <= 0
            % constraint 1: y >= y'(l) * (x - l) + y(l)
            % constraint 2: y >= y'(u) * (x - u) + y(u)
            % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);

            map1 = find(u <= 0 & l~=u);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);

                % constraint 1: y >= y'(l) * (x - l) + y(l)
                C21 = [da.*I.V(map1, 2:nv) -V2(map1, :)];
                d21 = -da.*(I.V(map1, 1)-l(map1)) - a;
                % constraint 2: y >= y'(u) * (x - u) + y(u) 
                C22 = [db.*I.V(map1, 2:nv) -V2(map1, :)];
                d22 = -db.*(I.V(map1, 1) - u(map1)) - b;
                % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);
                gamma = (b-a)./(u(map1)-l(map1));
                C23 = [-gamma.*I.V(map1, 2:nv) V2(map1, :)];
                d23 = gamma.*(I.V(map1, 1)-l(map1)) + a;

                C2 = [C21; C22; C23]; 
                d2 = [d21; d22; d23];
            else
                C2 = [];
                d2 = [];
            end
            
            % C3, d3, l< 0 & u > 0, x >0 or x < 0
            %y is concave for x in [l, 0] and convex for x
            % in [0, u]
            % split can be done here            

            map1 = find(l < 0 & u > 0);
            if ~isempty(map1)
                a = yl(map1);
                b = yu(map1);
                da = dyl(map1);
                db = dyu(map1);

                dmin = (min(da', db'))';
                % over-approximation constraints 
                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u)

                % constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
                C31 = [dmin.*I.V(map1, 2:nv) -V2(map1, :)];
                d31 = -dmin.*(I.V(map1, 1)-l(map1)) - a;
                % constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u) 
                C32 = [-dmin.*I.V(map1, 2:nv) V2(map1, :)];
                d32 = dmin.*(I.V(map1, 1) - u(map1)) + b;


                y1 = dmin.*(-l(map1)) + a;
                y2 = dmin.*(-u(map1)) + b;
                g2 = (y2 - a)./(-l(map1));
                g1 = (y1 - b)./(-u(map1));

                % constraint 3: y <= g2 * x + y2
                C33 = [-g2.*I.V(map1, 2:nv) V2(map1, :)];
                d33 = g2.*I.V(map1, 1) + y2;

                % constraint 4: y >= g1 * x + y1
                C34 = [g1.*I.V(map1, 2:nv) -V2(map1, :)];
                d34 = -g1.*I.V(map1, 1) - y1;

                C3 = [C31; C32; C33; C34]; 
                d3 = [d31; d32; d33; d34];
            else
                C3 = [];
                d3 = [];
            end
            
            new_C = [C0; C1; C2; C3];
            new_d = [d0; d1; d2; d3]; 

            new_pred_lb = [I.predicate_lb; yl(map2)];
            new_pred_ub = [I.predicate_ub; yu(map2)];

            S = Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub);
            
        end


          
    end
    
    
    methods(Static) % over-approximate reachability analysis using Zonotope
        
        function Z = reach_zono_approx(I)
            % @I: zonotope input set
            % @Z: zonotope output set
            
            % author: Dung Tran
            % date: 5/3/2019
            
            % reference: Fast and Effective Robustness Certification,
            % Gagandeep Singh, NIPS, 2018
            
            if ~isa(I, 'Zono')
                error('Input set is not a Zonotope');
            end
            
            B = I.getBox;
            
            lb = B.lb;
            ub = B.ub;
            G = [logsig('dn', lb) logsig('dn', ub)];
            gamma_opt = min(G, [], 2);
            gamma_mat = diag(gamma_opt);
            mu1 = 0.5 * (logsig(ub) + logsig(lb) - gamma_mat * (ub + lb));
            mu2 = 0.5 * (logsig(ub) - logsig(lb) - gamma_mat * (ub - lb));
            Z1 = I.affineMap(gamma_mat, mu1);
            new_V = diag(mu2);
            V = [Z1.V new_V];
            Z = Zono(Z1.c, V);
            
        end
        
        % dealing with multiple inputs in parallel
        function S = reach_zono_approx_multipleInputs(varargin)
            % author: Dung Tran
            % date: 3/27/2020
            
            switch nargin
                case 1
                    I = varargin{1};
                    parallel = []; % no parallel computation
                case 2
                    I = varargin{1};
                    parallel = varargin{2};
                otherwise
                    error('Invalid number of input arguments, should be 1 or 2');
            end
            
            p = length(I);
            S = [];
            if isempty(parallel)
                
                for i=1:p
                    S =[S LogSig.reach_zono_approx(I(i))];
                end
                
            elseif strcmp(parallel, 'parallel')
                
                parfor i=1:p
                    S =[S, LogSig.reach_zono_approx(I(i))];
                end
                
            else
                error('Unknown parallel computation option');
            end

        end
        
    end
    
methods(Static) % over-approximate reachability analysis using abstract domain
    
    
    function S = stepLogSig_abstract_domain(I, index, l, u, y_l, y_u, dy_l, dy_u)
        % @I: input star set
        % @index: index of the neuron
        % @l: l = min(x[index]), lower bound at neuron x[index] 
        % @u: u = min(x[index]), upper bound at neuron x[index]
        % @y_l: = logsig(l); output of logsig at lower bound
        % @y_u: = logsig(u); output of logsig at upper bound
        % @dy_l: derivative of LogSig at the lower bound
        % @dy_u: derivative of LogSig at the upper bound

        % @S: output star set

        % author: Dung Tran
        % date: 3/27/2020
                
        if l == u
            new_V = I.V;
            new_V(index, 1:I.nVar+1) = 0;
            new_V(index, 1) = y_l;
            S = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub);                
        elseif l >= 0
            % y is convex when x >= 0
            % constraint 2: y <= y'(u) * (x - u) + y(u) 
            % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);


            n = I.nVar + 1;
            % over-approximation constraints 
            % constraint 2: y <= y'(u) * (x - u) + y(u)
            C2 = [-dy_u*I.V(index, 2:n) 1];
            d2 = dy_u * I.V(index, 1) - dy_u*u + y_u;
            % constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            a = (y_u - y_l)/(u - l);
            C3 = [a*I.V(index, 2:n) -1];
            d3 = a*l - y_l - a*I.V(index, 1);

            m = size(I.C, 1);
            C0 = [I.C zeros(m, 1)];
            d0 = I.d;
            new_C = [C0; C2; C3];
            new_d = [d0; d2; d3];
            new_V = [I.V zeros(I.dim, 1)];
            new_V(index, :) = zeros(1, n+1);
            new_V(index, n+1) = 1; 

            % update predicate bound
            new_predicate_lb = [I.predicate_lb; y_l]; 
            new_predicate_ub = [I.predicate_ub; y_u];
            S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);


        elseif u <= 0
            % y is concave when x <= 0
            % constraint 1: y >= y'(l) * (x - l) + y(l)
            % constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l);

            n = I.nVar + 1;
            % over-approximation constraints 
            % constraint 1: y >= y'(l) * (x - l) + y(l)
            C1 = [dy_l*I.V(index, 2:n) -1];
            d1 = -dy_l * I.V(index, 1) + dy_l*l - y_l; 
            % constraint 3: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l);
            a = (y_u - y_l)/(u - l);
            C3 = [-a*I.V(index, 2:n) 1];
            d3 = -a*l + y_l + a*I.V(index, 1);

            m = size(I.C, 1);
            C0 = [I.C zeros(m, 1)];
            d0 = I.d;
            new_C = [C0; C1; C3];
            new_d = [d0; d1; d3];
            new_V = [I.V zeros(I.dim, 1)];
            new_V(index, :) = zeros(1, n+1);
            new_V(index, n+1) = 1; 

            % update predicate bound
            new_predicate_lb = [I.predicate_lb; y_l]; 
            new_predicate_ub = [I.predicate_ub; y_u];
            S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);


        elseif l <0 && u >0
            % y is concave for x in [l, 0] and convex for x
            % in [0, u]
            % split can be done here 

            
            % over-approximation constraints 
            % constraint 1: y >= y'(l) * (x - l) + y(l)
            % constraint 2: y <= y'(u) * (x - u) + y(u)
           
            n = I.nVar + 1;

            dy_min = min(dy_l, dy_u);
            % constraint 1: y >= y'_min * (x - l) + y(l)
            C1 = [dy_min*I.V(index, 2:n) -1];
            d1 = -dy_min * I.V(index, 1) + dy_min*l - y_l; 
            % constraint 2: y <= y'_min * (x - u) + y(u)
            C2 = [-dy_min*I.V(index, 2:n) 1];
            d2 = dy_min * I.V(index, 1) - dy_min*u + y_u;
            
            m = size(I.C, 1);
            C0 = [I.C zeros(m, 1)];
            d0 = I.d;
            new_C = [C0; C1; C2];
            new_d = [d0; d1; d2];
            new_V = [I.V zeros(I.dim, 1)];
            new_V(index, :) = zeros(1, n+1);
            new_V(index, n+1) = 1; 

            % update predicate bound
            new_predicate_lb = [I.predicate_lb; y_l]; 
            new_predicate_ub = [I.predicate_ub; y_u];
            S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub);

        end

        
        
    end
    
    function S = reach_abstract_domain_approx(I)
        % @I: star input set
        % @Z: Star output set

        % author: Dung Tran
        % date: 3/27/2020

        % reference: An abstract domain for certifying neural networks. Proceedings of the ACM on Programming Languages,
        % Gagandeep Singh, POPL, 2019

        if ~isa(I, 'Star')
            error('Input set is not a Star');
        end

        %[l, u] = I.estimateRanges;  
        [l, u] = I.getRanges;
        
        y_l = logsig(l);
        y_u = logsig(u);
        dy_l = logsig('dn', l);
        dy_u = logsig('dn', u);

        n = I.dim;
        S = I;
        for i=1:n
            S = LogSig.stepLogSig_abstract_domain(S, i, l(i), u(i), y_l(i), y_u(i), dy_l(i), dy_u(i)); 
        end

    end
    
    % dealing with multiple inputs in parallel
    function S = reach_abstract_domain_approx_multipleInputs(varargin)
        % author: Dung Tran
        % date: 3/27/2020

        switch nargin
            case 1
                I = varargin{1};
                parallel = []; % no parallel computation
            case 2
                I = varargin{1};
                parallel = varargin{2};
            otherwise
                error('Invalid number of input arguments, should be 1 or 2');
        end

        p = length(I);
        S = [];
        if isempty(parallel)

            for i=1:p
                S =[S LogSig.reach_abstract_domain_approx(I(i))];
            end

        elseif strcmp(parallel, 'parallel')

            parfor i=1:p
                S =[S, LogSig.reach_abstract_domain_approx(I(i))];
            end

        else
            error('Unknown parallel computation option');
        end

    end

end

methods(Static) % over-approximate reachability analysis using abstract-domain (absdom) based on eran
    
    % step over-approximate reachability analysis using abstract-domain
    % we use absdom set to represent abstract-domain
    function A = stepLogSig_absdom(I, index, l, u, y_l, y_u, dy_l, dy_u)
        % @I: absdom-input set
        % @index: index of neuron performing stepReach
        % @l: l = min(x[index]), lower bound at neuron x[index] 
        % @u: u = min(x[index]), upper bound at neuron x[index]
        % @y_l: = logsig(l); output of logsig at lower bound
        % @y_u: = logsig(u); output of logsig at upper bound
        % @dy_l: derivative of LogSig at the lower bound
        % @dy_u: derivative of LogSig at the upper bound
        
        % @A: absdom output set

        % author: Sung Woo Choi
        % date: 01/29/2021

        % reference: An Abstract Domain for Certifying Neural Networks,
        % Gagandeep Singh, POPL 2019
        
        if ~isa(I, 'AbsDom')
                error('Input is not a AbsDom');
        end
            
        lower_a = I.lower_a;
        upper_a = I.upper_a;
        lb = I.lb;
        ub = I.ub;
        len = length(lower_a);
        if l == u
            L = zeros(1, I.dim + 1);
            L(index+1) = y_l;
            lower_a{len}(index,:) = L;

            U = zeros(1, I.dim + 1);
            U(index+1) = y_l;
            upper_a{len}(index,:) = U;

            lb{len}(index) = y_l;
            ub{len}(index) = y_u;

            A = AbsDom(lower_a, upper_a, lb, ub, I.iter);
        else
            lamda = (y_u - y_l)/(u - l);
            dlamda = min(dy_l , dy_u);
            if l > 0
                % constraint 1: y[index] >= y(l) + lamda * (x[index] - l)
                L = zeros(1, I.dim + 1);
                L(1) = y_l - lamda*l;
                L(index+1) = lamda;
                lower_a{len}(index,:) = L;
            else
                % constraint 1: y[index] >= y(l) + lamda' * (x[index] - l)
                L = zeros(1, I.dim + 1);
                L(1) = y_l - dlamda*l;
                L(index+1) = dlamda;
                lower_a{len}(index,:) = L;
            end
            
            if u <=0
                % constraint 2: y[index] <= y(u) + lamda * (x[index] - u)
                U = zeros(1, I.dim + 1);
                U(1) = y_u - lamda*u;
                U(index+1) = lamda;
                upper_a{len}(index,:) = U;
            else
                % constraint 2: y[index] <= y(u) + lamda' * (x[index] - u)
                U = zeros(1, I.dim + 1);
                U(1) = y_u - dlamda*u;
                U(index+1) = dlamda;
                upper_a{len}(index,:) = U;
            end
            
            lb{len}(index) = y_l;
            ub{len}(index) = y_u;
            
            A = AbsDom(lower_a, upper_a, lb, ub, I.iter);
        end
    end
    
    function A = reach_absdom_approx(I)
        % @I: absdom imput set
        % %A: absdom output set
        
        % author: Sung Woo Choi
        % date: 01/29/2021
        
        % reference: An abstract domain for certifying neural networks. Proceedings of the ACM on Programming Languages,
        % Gagandeep Singh, POPL, 2019
        
        if ~isa(I, 'AbsDom')
            error('Input is not a AbsDom');
        end
            
        if isempty(I)
            A = [];
        else
            lower_a = I.lower_a;
            upper_a = I.upper_a;
            lb = I.lb;
            ub = I.ub;
            len = length(lower_a);

            l = lb{len};
            u = ub{len};
            y_l = logsig(l);
            y_u = logsig(u);
            dy_l = logsig('dn', l);
            dy_u = logsig('dn', u);

            % create new matrices for lower and upper constraints and bounds.
            lower_a{len+1} = zeros(I.dim, I.dim + 1);
            upper_a{len+1} = zeros(I.dim, I.dim + 1);
            lb{len+1} = zeros(I.dim, 1);
            ub{len+1} = zeros(I.dim, 1);
            A = AbsDom(lower_a, upper_a, lb, ub, I.iter);

            for i=1:I.dim
                A = LogSig.stepLogSig_absdom(A, i, l(i), u(i), y_l(i), y_u(i), dy_l(i), dy_u(i)); 
            end
        end
    end
end

methods(Static) % over-approximate reachability analysis using abstract-domain (absdom) based on eran
    
    % step over-approximate reachability analysis using abstract-domain
    % we use absdom set to represent abstract-domain
    function R = stepLogSig_absdom_twoConstraints_rstar(I, index, l, u, y_l, y_u, dy_l, dy_u)
        % @I: rstar-input set
        % @index: index of neuron performing stepReach
        % @l: l = min(x[index]), lower bound at neuron x[index] 
        % @u: u = min(x[index]), upper bound at neuron x[index]
        % @y_l: = logsig(l); output of logsig at lower bound
        % @y_u: = logsig(u); output of logsig at upper bound
        % @dy_l: derivative of LogSig at the lower bound
        % @dy_u: derivative of LogSig at the upper bound
        
        % @A: rstar output set

        % author: Sung Woo Choi
        % date: 01/29/2021

        % reference: An Abstract Domain for Certifying Neural Networks,
        % Gagandeep Singh, POPL 2019
        
        if ~isa(I, 'RStar')
            error('Input is not a RStar');
        end
            
        lower_a = I.lower_a;
        upper_a = I.upper_a;
        lb = I.lb;
        ub = I.ub;
        len = length(lower_a);
        if l == u
            new_V = I.V;
            new_V(index,:) = 0;
            new_V(index,1) = y_l;
            
            L = zeros(1, I.dim + 1);
            L(index+1) = y_l;
            lower_a{len}(index,:) = L;

            U = zeros(1, I.dim + 1);
            U(index+1) = y_l;
            upper_a{len}(index,:) = U;

            lb{len}(index) = y_l;
            ub{len}(index) = y_u;

            R = RStar(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, lower_a, upper_a, lb, ub, I.iter);
        else
            new_V = [I.V zeros(I.dim, 1)];
            new_V(index, :) = 0;
            new_V(index, end) = 1;
            
            C0 = [I.C zeros(size(I.C,1),1)];
            d0 = I.d;
                
            if l >= 0
                lambda = (y_u - y_l)/(u - l);
                
                % constraint 1: y[index] >= y(l) + lambda * (x[index] - l)
                L = zeros(1, I.dim + 1);
                L(1) = y_l - lambda*l;
                L(index+1) = lambda;
                lower_a{len}(index,:) = L;
                
                C1 = [lambda*I.X(index,:) -1];
                d1 = -y_l - lambda*(I.c(index) - l);
                
                % constraint 2: y[index] <= y(u) + y'(u) * (x[index] - u)                
                U = zeros(1, I.dim + 1);
                U(1) = y_u - dy_u*u;
                U(index+1) = dy_u;
                upper_a{len}(index,:) = U;
                
                C2 = [-dy_u*I.X(index,:) 1];
                d2 = y_u + dy_u*(I.c(index) - u);
            elseif u <= 0
                lambda = (y_u - y_l)/(u - l);
                
                % constraint 1: y[index] >= y(l) + y'(l) * (x[index] - l)               
                L = zeros(1, I.dim + 1);
                L(1) = y_l - dy_l*l;
                L(index+1) = dy_l;
                lower_a{len}(index,:) = L;
                
                C1 = [dy_l*I.X(index,:) -1];
                d1 = -y_l - dy_l*(I.c(index) - l);
                
                % constraint 2: y[index] <= y(l) + lambda * (x[index] - l)
                U = zeros(1, I.dim + 1);
                U(1) = y_l - lambda*l;
                U(index+1) = lambda;
                upper_a{len}(index,:) = U;
                
                C2 = [-lambda*I.X(index,:) 1];
                d2 = y_l + lambda*(I.c(index) - l);
                
                
%                 % constraint 2: y[index] <= y(u) + lambda * (x[index] - u)
%                 U = zeros(1, I.dim + 1);
%                 U(1) = y_u - lamda*u;
%                 U(index+1) = lambda;
%                 upper_a{len}(index,:) = U;
%                 
%                 C2 = [-lambda*I.X(index,:) 1];
%                 d2 = y_u + lambda*(I.c(index) - u);
            else
                dlambda = min(dy_l , dy_u);
                % constraint 1: y[index] >= y(l) + lambda' * (x[index] - l)
                L = zeros(1, I.dim + 1);
                L(1) = y_l - dlambda*l;
                L(index+1) = dlambda;
                lower_a{len}(index,:) = L;
                
                C1 = [dlambda*I.X(index,:) -1];
                d1 = -y_l - dlambda*(I.c(index) - l);
                
                % constraint 2: y[index] <= y(u) + lamda' * (x[index] - u)               
                U = zeros(1, I.dim + 1);
                U(1) = y_u - dlambda*u;
                U(index+1) = dlambda;
                upper_a{len}(index,:) = U;
                
                C2 = [-dlambda*I.X(index,:) 1];
                d2 = y_u + dlambda*(I.c(index) - u);
            end
          
%             dlamda = min(dy_l , dy_u);
%             if l > 0
%                 % constraint 1: y[index] >= y(l) + lamda * (x[index] - l)
%                 L = zeros(1, I.dim + 1);
%                 L(1) = y_l - lamda*l;
%                 L(index+1) = lamda;
%                 lower_a{len}(index,:) = L;
%                 
%                 C1 = [lamda*I.X(index,:) -1];
%                 d1 = -y_l - lamda*(I.c(index) - l);
%             else
%                 % constraint 1: y[index] >= y(l) + lamda' * (x[index] - l)
%                 L = zeros(1, I.dim + 1);
%                 L(1) = y_l - dlamda*l;
%                 L(index+1) = dlamda;
%                 lower_a{len}(index,:) = L;
%                 
%                 C1 = [dy_l*I.X(index,:) -1];
%                 d1 = -y_l - dy_l*(I.c(index) - l);
%             end
%             
%             if u <=0
%                 % constraint 2: y[index] <= y(u) + lamda * (x[index] - u)
%                 U = zeros(1, I.dim + 1);
%                 U(1) = y_u - lamda*u;
%                 U(index+1) = lamda;
%                 upper_a{len}(index,:) = U;
%                 
%                 C2 = [-lamda*I.X(index,:) 1];
%                 d2 = y_u + lamda*(I.c(index) - u);
%             else
%                 % constraint 2: y[index] <= y(u) + lamda' * (x[index] - u)
%                 U = zeros(1, I.dim + 1);
%                 U(1) = y_u - dlamda*u;
%                 U(index+1) = dlamda;
%                 upper_a{len}(index,:) = U;
%                 
%                 C2 = [-dlamda*I.X(index,:) 1];
%                 d2 = y_u + dlamda*(I.c(index) - u);
%             end

            new_pred_lb = [I.predicate_lb; y_l];
            new_pred_ub = [I.predicate_ub; y_u];
                    
            lb{len}(index) = y_l;
            ub{len}(index) = y_u;
            
            new_C = [C0; C1; C2];
            new_d = [d0; d1; d2];
            
            R = RStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub, lower_a, upper_a, lb, ub, I.iter);
        end
    end
    
    function R = reach_rstar_absdom_with_two_pred_const(I)
        % @I: RStar imput set
        % %A: RStar output set
        
        % author: Sung Woo Choi
        % date: 01/29/2021
        
        % reference: An abstract domain for certifying neural networks. Proceedings of the ACM on Programming Languages,
        % Gagandeep Singh, POPL, 2019
        
        if ~isa(I, 'RStar')
            error('Input is not a RStar');
        end
            
        if isempty(I)
            R = [];
        else
            lower_a = I.lower_a;
            upper_a = I.upper_a;
            lb = I.lb;
            ub = I.ub;
            len = length(lower_a);

%             [l_e, u_e] = I.getExactRanges;
            l = lb{len};
            u = ub{len};
%             if(l_e ~= l)
%                 fprintf('different lower bound %e \n', l_e-l);
%             end
%             if(u_e ~= u)
%                 fprintf('different upper bound %e \n', u_e-u);
%             end
            
            y_l = logsig(l);
            y_u = logsig(u);
            dy_l = logsig('dn', l);
            dy_u = logsig('dn', u);

            % create new matrices for lower and upper constraints and bounds.
            lower_a{len+1} = zeros(I.dim, I.dim + 1);
            upper_a{len+1} = zeros(I.dim, I.dim + 1);
            lb{len+1} = zeros(I.dim, 1);
            ub{len+1} = zeros(I.dim, 1);
            R = RStar(I.V, I.C, I.d, I.predicate_lb, I.predicate_ub, lower_a, upper_a, lb, ub, I.iter);

            for i=1:I.dim
                R = LogSig.stepLogSig_absdom_twoConstraints_rstar(R, i, l(i), u(i), y_l(i), y_u(i), dy_l(i), dy_u(i)); 
            end
        end
    end
end

methods(Static) % over-approximate reachability analysis using abstract-domain (absdom) based on eran
    
    % step over-approximate reachability analysis using abstract-domain
    % we use absdom set to represent abstract-domain
    function R = stepLogSig_absdom_fourConstraints_rstar(I, index, l, u, y_l, y_u, dy_l, dy_u)
        % @I: rstar-input set
        % @index: index of neuron performing stepReach
        % @l: l = min(x[index]), lower bound at neuron x[index] 
        % @u: u = min(x[index]), upper bound at neuron x[index]
        % @y_l: = logsig(l); output of logsig at lower bound
        % @y_u: = logsig(u); output of logsig at upper bound
        % @dy_l: derivative of LogSig at the lower bound
        % @dy_u: derivative of LogSig at the upper bound
        
        % @A: rstar output set

        % author: Sung Woo Choi
        % date: 01/29/2021

        % reference: An Abstract Domain for Certifying Neural Networks,
        % Gagandeep Singh, POPL 2019
        
        if ~isa(I, 'RStar')
            error('Input is not a RStar');
        end
            
        lower_a = I.lower_a;
        upper_a = I.upper_a;
        lb = I.lb;
        ub = I.ub;
        len = length(lower_a);
        if l == u
            new_V = I.V;
            new_V(index,:) = 0;
            new_V(index,1) = y_l;
            
            L = zeros(1, I.dim + 1);
            L(index+1) = y_l;
            lower_a{len}(index,:) = L;

            U = zeros(1, I.dim + 1);
            U(index+1) = y_l;
            upper_a{len}(index,:) = U;

            lb{len}(index) = y_l;
            ub{len}(index) = y_u;

            R = RStar(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, lower_a, upper_a, lb, ub, I.iter);
        else
            new_V = [I.V zeros(I.dim, 1)];
            new_V(index, :) = 0;
            new_V(index, end) = 1;
            
            C0 = [I.C zeros(size(I.C,1),1)];
            d0 = I.d;
                
            lamda = (y_u - y_l)/(u - l);
            if l > 0
                % constraint 1: y[index] >= y(l) + lamda * (x[index] - l)
                L = zeros(1, I.dim + 1);
                L(1) = y_l - lamda*l;
                L(index+1) = lamda;
                lower_a{len}(index,:) = L;
                
                C1 = [lamda*I.X(index,:) -1];
                d1 = -y_l - lamda*(I.c(index) - l);
                
                % constraint 2: y[index] <= y(u) + lamda' * (x[index] - u)
                U = zeros(1, I.dim + 1);
                U(1) = y_u - dy_u*u;
                U(index+1) = dy_u;
                upper_a{len}(index,:) = U;
                
                C2 = [-dy_u*I.X(index,:) 1];
                d2 = y_u + dy_u*(I.c(index) - u);
            elseif u <= 0
                % constraint 1: y[index] >= y(l) + lamda' * (x[index] - l)
                L = zeros(1, I.dim + 1);
                L(1) = y_l - dy_l*l;
                L(index+1) = dy_l;
                lower_a{len}(index,:) = L;
                
                C1 = [dy_l*I.X(index,:) -1];
                d1 = -y_l - dy_l*(I.c(index) - l);
                
                % constraint 2: y[index] <= y(u) + lamda * (x[index] - u)
                U = zeros(1, I.dim + 1);
                U(1) = y_u - lamda*u;
                U(index+1) = lamda;
                upper_a{len}(index,:) = U;
                
                C2 = [-lamda*I.X(index,:) 1];
                d2 = y_u + lamda*(I.c(index) - u);
            else
                dlamda = min(dy_l , dy_u);
                % constraint 1: y[index] >= y(l) + lamda' * (x[index] - l)
                L = zeros(1, I.dim + 1);
                L(1) = y_l - dlamda*l;
                L(index+1) = dlamda;
                lower_a{len}(index,:) = L;
                
                C1 = [dlamda*I.X(index,:) -1];
                d1 = -y_l - dlamda*(I.c(index) - l);
                
                % constraint 2: y[index] <= y(u) + lamda' * (x[index] - u)
                U = zeros(1, I.dim + 1);
                U(1) = y_u - dlamda*u;
                U(index+1) = dlamda;
                upper_a{len}(index,:) = U;
                
                C2 = [-dlamda*I.X(index,:) 1];
                d2 = y_u + dlamda*(I.c(index) - u);
            end

            new_pred_lb = [I.predicate_lb; y_l];
            new_pred_ub = [I.predicate_ub; y_u];
                    
            lb{len}(index) = y_l;
            ub{len}(index) = y_u;
            
            new_C = [C0; C1; C2];
            new_d = [d0; d1; d2];
            
            R = RStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub, lower_a, upper_a, lb, ub, I.iter);
        end
    end
    
    function R = reach_rstar_absdom_with_four_pred_const(I)
        % @I: RStar imput set
        % %A: RStar output set
        
        % author: Sung Woo Choi
        % date: 01/29/2021
        
        % reference: An abstract domain for certifying neural networks. Proceedings of the ACM on Programming Languages,
        % Gagandeep Singh, POPL, 2019
        
        if ~isa(I, 'RStar')
            error('Input is not a RStar');
        end
            
        if isempty(I)
            R = [];
        else
            lower_a = I.lower_a;
            upper_a = I.upper_a;
            lb = I.lb;
            ub = I.ub;
            len = length(lower_a);

            l = lb{len};
            u = ub{len};
            y_l = logsig(l);
            y_u = logsig(u);
            dy_l = logsig('dn', l);
            dy_u = logsig('dn', u);

            % create new matrices for lower and upper constraints and bounds.
            lower_a{len+1} = zeros(I.dim, I.dim + 1);
            upper_a{len+1} = zeros(I.dim, I.dim + 1);
            lb{len+1} = zeros(I.dim, 1);
            ub{len+1} = zeros(I.dim, 1);
            R = RStar(I.V, I.C, I.d, I.predicate_lb, I.predicate_ub, lower_a, upper_a, lb, ub, I.iter);

            for i=1:I.dim
                R = LogSig.stepLogSig_absdom_fourConstraints_rstar(R, i, l(i), u(i), y_l(i), y_u(i), dy_l(i), dy_u(i)); 
            end
        end
    end
end

methods(Static) % main reach method
    
    % main function for reachability analysis
    function R = reach(varargin)
        % @I: an array of star input sets
        % @method: 'approx-star' or 'approx-zono' or 'abs-dom' 
        % @option: = 'parallel' or [] using parallel computation or not

        % author: Dung Tran
        % date: 3/27/2019
        % update: 4/2/2020
        % update: 6/26/2020: add relaxed approx-star method
        % update 7/15/2020: add display option
        %        7/16/2020: add lp_solver option
        % update: 01/29/2020: add absdom & rstar (Sung)
            
        switch nargin
            
            case 6
                I = varargin{1};
                method = varargin{2};
                reachOption = varargin{3};
                relaxFactor = varargin{4}; % used for aprox-star only
                dis_opt = varargin{5}; % display option
                lp_solver = varargin{6};
                
            case 5
                I = varargin{1};
                method = varargin{2};
                reachOption = varargin{3};
                relaxFactor = varargin{4}; % used for aprox-star only
                dis_opt = varargin{5}; % display option
                lp_solver = 'glpk';
            case 4
                I = varargin{1};
                method = varargin{2};
                reachOption = varargin{3};
                relaxFactor = varargin{4}; % used for aprox-star only
                dis_opt = []; % display option
                lp_solver = 'glpk';

            case 3
                I = varargin{1};
                method = varargin{2};
                reachOption = varargin{3};
                relaxFactor = 0; % used for aprox-star only
                dis_opt = []; % display option
                lp_solver = 'glpk';
            case 2
                I = varargin{1};
                method = varargin{2};
                reachOption = [];
                relaxFactor = 0; % used for aprox-star only
                dis_opt = []; % display option
                lp_solver = 'glpk';
            case 1
                I = varargin{1};
                method = 'approx-star';
                reachOption = [];
                relaxFactor = 0; % for relaxed approx-star method
                dis_opt = [];
                lp_solver = 'glpk';
            otherwise
                error('Invalid number of input arguments (should be 1, 2, 3, 4, 5, or 6)');
        end

        if strcmp(method, 'approx-star') || strcmp(method, 'approx-star-no-split') || strcmp(method, 'approx-star-split')
            R = LogSig.reach_star_approx(I, method, reachOption, relaxFactor, dis_opt, lp_solver);
        elseif strcmp(method, 'approx-zono')  % over-approximate analysis using zonotope
            R = LogSig.reach_zono_approx(I);
        elseif strcmp(method, 'abs-dom')  % over-approximate analysis using abstract-domain
            R = LogSig.reach_abstract_domain_approx(I);
        elseif strcmp(method, 'absdom') % over-approximate analysis using abstract-domain based on eran
            R = LogSig.reach_absdom_approx(I);
        elseif strcmp(method, 'rstar-absdom-two') % over-approximate analysis using abstract-domain with 2 star constraints
            R = LogSig.reach_rstar_absdom_with_two_pred_const(I);
        elseif strcmp(method, 'rstar-absdom-four') % over-approximate analysis using abstract-domain with 4 star constraints
%             R = LogSig.reach_rstar_absdom_with_four_pred_const(I);
        else
            error('Unknown or unsupported reachability method for layer with LogSig activation function');
        end
            


    end
    
end

    
end

