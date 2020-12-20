classdef RStar
    % Relaxed Star class
    % Sung Woo Choi: 12/18/2020
    
    properties
        V = []; % basic matrix that contains c and X
        C = []; % constraint matrix
        d = []; % constraint vector
        c = []; % center vector
        X = []; % basic matrix
        
        lower_a = {[]}; % a set of matrix for lower constraint for bound (a[<=]) ((1 a[1] a[2] ... a[n])'
        upper_a = {[]}; % a set of matrix for upper constraint for bound (a[>=])
        lb = []; % a set of matrix for lower bound
        ub = []; % a set of matrix for upper bound

        iter = inf; % number of iterations for back substitution
        dim = 0; % dimension of current relaxed star set
    end
    
    methods
        
        % constructor
        function obj = RStar(varargin)
         % @V: bassic matrix
            % @C: constraint matrix
            % @d: constraint vector
            % @lower_a: a set of matrix for lower polyhedral constraint
            % @upper_a: a set of matrix for upper polyhedral constraint
                
            switch nargin
                
                case 8
                    V = varargin{1};
                    C = varargin{2};
                    d = varargin{3};
                    lower_a = varargin{4};
                    upper_a = varargin{5};
                    lb = varargin{6};
                    ub = varargin{7};
                    iter = varargin{8};
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % add code for checking properties
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    obj.V = V;
                    obj.C = C;
                    obj.d = d;
                    
                    obj.lower_a = lower_a;
                    obj.upper_a = upper_a;
                    obj.lb = lb;
                    obj.ub = ub;
                    
                    obj.iter = iter;
                    len = length(lower_a);
                    obj.dim = size(lower_a{len}, 1);
                    
                case 7
                    V = varargin{1};
                    C = varargin{2};
                    d = varargin{3};
                    lower_a = varargin{4};
                    upper_a = varargin{5};
                    lb = varargin{6};
                    ub = varargin{7};
                    iter = inf;
                    
                    obj = RStar(V, C, d, lower_a, upper_a, lb, ub, iter);
                    
                case 2
                    I = varargin{1};
                    iter = varargin{2};
                    if isa(I, 'Polyhedron')
                        dim = I.Dim;
                        c = zeros(dim,1);
                        Ve = eye(dim);
                        V = [c Ve];
                        if ~isempty(I.Ae)
                            C = [I.A;I.Ae;-I.Ae];
                            d = [I.b;I.be;-I.be];
                        else
                            C = I.A;
                            d = I.b;
                        end
                        I.outerApprox;
                        l = I.Internal.lb;
                        u = I.Internal.ub;
                    elseif isa(I, 'Star')
                        dim = I.dim;
                        V = I.V;
                        C = I.C;
                        d = I.d;
                        [l, u] = I.getRanges();
                    elseif isa(I, 'Zono')
                        dim = I.dim;
                        [l, u] = I.getRanges();
                        V = [I.c I.V];
                        C = [eye(dim); -eye(dim)];
                        d = [u; -l];
                    else
                        error('Unkown imput set');
                    end
                    
                    if iter <= 0
                        error('Iteration must be greater than zero');
                    end
                    
                    lower_a{1} = [zeros(dim, 1) eye(dim)];
                    upper_a{1} = [zeros(dim, 1) eye(dim)];
                    lb{1} = l;
                    ub{1} = u;
                    obj = RStar(V, C, d, lower_a, upper_a, lb, ub, iter);
                    
                case 1
                    I = varargin{1};
                    obj = RStar(I, inf);
                    
                case 0
                    obj.V = [];
                    obj.C = [];
                    obj.d = [];
                    obj.lower_a = {[]};
                    obj.upper_a = {[]};
                    obj.lb = [];
                    obj.ub = [];
                    obj.iter = inf;
                    obj.dim = 0;
                    
                otherwise
                    error('Invalid number of input arguments (should be 0, 1, 2, 7, 8)');
            end
        end
        
        % affine abstract mapping of RStar set
        function R = affineMap(varargin)
            switch nargin
                case 3
                    obj = varargin{1};
                    W = varargin{2};
                    b = varargin{3};
                case 2
                    obj = varargin{1};
                    W = varargin{2};
                    b = zeros(size(W,1),1);
            end
            
            [nW, mW] = size(W);
            [nb, mb] = size(b);
            
            if mW ~= obj.dim
                error('Inconsistency between the affine mapping matrix and dimension of the RStar set');
            end
            
            if mb > 1
                error('bias vector must be one column');
            end
            
            if nW ~= nb
                error('Inconsistency between the affine mapping matrix and the bias vector');
            end

            % affine mapping of basic matrix
            V = W * obj.V;
            if mb ~= 0
                V(:, 1) = V(:, 1) + b;
            end

            % new lower and upper polyhedral contraints
            lower_a = obj.lower_a;
            upper_a = obj.upper_a;
            len = length(lower_a);
            
            lower_a{len+1} = [b W];
            upper_a{len+1} = [b W];
            
            % new lower and uppper bounds
            lb = obj.lb;
            ub = obj.ub;
            lb{len+1} = obj.lb_backSub(lower_a, upper_a);
            ub{len+1} = obj.ub_backSub(lower_a, upper_a);

            R = RStar(V, obj.C, obj.d, lower_a, upper_a, lb, ub, obj.iter); 
        end

        % lower bound back-substitution
        function lb = lb_backSub(obj, lower_a, upper_a)
            maxIter = obj.iter;
            len = length(upper_a);
            [nL, mL] = size(upper_a{len});
            alpha = upper_a{len}(:,2:end);
            lower_v = zeros(nL, 1);
            upper_v = upper_a{len}(:,1);
            
            % b[s+1] = v' + sum( max(0,w[j]')*lower_a[j] + min(w[j]',0)*upper_a[j}] ) for j is element of k and for k < i
            % iteration until lb' = b[s'] = v''
            len = len - 1;
            iter = 0;
            while (len > 1 && iter < maxIter)
                [nL, mL] = size(upper_a{len});
                dim = nL;
                
                max_a = max(0, alpha);
                min_a = min(alpha, 0);

                lower_v = max_a * lower_a{len}(:,1) + lower_v;
                upper_v = min_a * upper_a{len}(:,1) + upper_v;
                
                alpha = max_a * lower_a{len}(:,2:end) + ...
                        min_a * upper_a{len}(:,2:end);

                len = len - 1;
                iter = iter + 1;
            end
            
            max_a = max(0, alpha);
            min_a = min(alpha, 0);
            
            [lb1,ub1] = getRanges_L(obj,len);
            lb = max_a * lb1 + lower_v + ...
                 min_a * ub1 + upper_v;
        end
        
        % upper bound back-substituion
        function ub = ub_backSub(obj, lower_a, upper_a)
            maxIter = obj.iter;
            len = length(upper_a);
            [nL, mL] = size(upper_a{len});
            alpha = upper_a{len}(:,2:end);
            lower_v = zeros(nL, 1);
            upper_v = upper_a{len}(:,1);
            
            % c[t+1] = v' + sum( max(0,w[j]')*upper_a[j] + min(w[j]',0)*lower_a[j}] )  for j is element of k and for k < i
            % iteration until ub' = c[t'] = v''
            len = len - 1;
            iter = 0;
            while (len > 1 && iter < maxIter)
                dim = size(lower_a{len}, 1);
                
                max_a = max(0, alpha);
                min_a = min(alpha, 0);
                
                lower_v = min_a * lower_a{len}(:,1) + lower_v;
                upper_v = max_a * upper_a{len}(:,1) + upper_v;
                
                alpha = min_a * lower_a{len}(:,2:end) + ...
                        max_a * upper_a{len}(:,2:end);
                    
                len = len - 1;
                iter = iter + 1;
            end

            max_a = max(0, alpha);
            min_a = min(alpha, 0);
            
            [lb1,ub1] = getRanges_L(obj,len);
            ub = min_a * lb1 + lower_v + ...
                 max_a * ub1 + upper_v;
        end

        % get the lower and upper bound of a current layer at specific
        % position
        function [lb,ub] = getRange(obj, i)
            if i > obj.dim
                error('i should not exceed dimnesion');
            end
            
            len = length(obj.lb);
            lb = obj.lb{len}(i);
            ub = obj.ub{len}(i);
        end
        
        % get lower and upper bounds of a current layer
        function [lb,ub] = getRanges(obj)
            len = length(obj.lb);
            lb = obj.lb{len};
            ub = obj.ub{len};
        end
        
        % get lower and upper bound of a specific layer
        function [lb,ub] = getRanges_L(obj, len)
            numL = length(obj.lower_a);
            if len > numL
                error('range request should be layers within iteration');
            end
            lb = obj.lb{len};
            ub = obj.ub{len};
        end
        
        % convert to Polyhedron
        function P = toPolyhedron(obj)
            P1 = Polyhedron('A', [obj.C], 'b', [obj.d]);
            X = obj.X;
            c = obj.c;
            P = X*P1 + c;
        end
        
        % convert to Zonotope
        function Z = toZono(obj)
            Z = Zono(obj.c, obj.X);
        end
        
        % convert to Star
        function S = toStar(obj)
            S = Star(obj.V, obj.C, obj.d);
        end
        
        % get basic matrix
        function X = get.X(obj)
            X = obj.V(:, 2:end);
        end
        
        % get center vector
        function c = get.c(obj)
            c = obj.V(:,1);
        end
        
        % plot RStar set
        function plot (varargin)
            color = 'red';
                   
            switch nargin
                case 1
                    obj = varargin{1};
                case 2
                    obj = varargin{1};
                    color = varargin{2};
            end
          
            n = length(obj);
            if ~strcmp(color, 'rand')
                c_rand = color;
            end
            hold on;
            for i=1:n
                I = obj(i);
                P = I.toPolyhedron;
                if strcmp(color, 'rand')
                    c_rand = rand(1,3);
                end
            
                plot(P, 'color', c_rand);
            end
            hold off
        end
        
        % intersection with other star set (half space)
        function S = Intersect(obj1, obj2)
            C1 = obj2.C * obj1.get_V;
            d1 = obj2.d - obj2.C * obj1.get_c;

            new_C = [obj1.C; C1];
            new_d = [obj1.d; d1];
            S = Star(obj1.V, new_C, new_d);     
            if isEmptySet(S)
                S = [];
            end
        end
    end
end