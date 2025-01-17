function [dx]=pancreas_St28_FlowEq(t,x,u)

dx(1,1) = 0.50631862*x(7) - 0.0278*x(1) - 2.78695;
dx(2,1) = 0.2 - 0.0078*x(3) - 0.0142*x(2);
dx(3,1) = 0.0152*x(2) - 0.0078*x(3);
dx(4,1) = 0.0581*x(5) - 0.0871*x(4) - x(4)*(0.00012207*x(1) + 0.01258413)*(0.0000025097*x(4)^2 - 0.0026*x(4) + 1);
dx(5,1) = 0.0871*x(4) - 0.0628*x(5) - 0.0121*x(9) + u(1)*(0.00000001113*x(11)^2 - 0.00001482*x(11) + 0.0049) + 3.7314;
dx(6,1) = 0.225*x(7) - 0.4219*x(6);
dx(7,1) = 0.0019*x(2) + 0.0078*x(3) + 0.1545*x(6) - 0.315*x(7);
dx(8,1) = 0.08377934*x(7) - 0.0046*x(8);
dx(9,1) = 0.0046*x(8) - 0.0046*x(9);
dx(10,1) = 0.05221*x(5) - 0.1*x(10);
dx(11,1) = 1;