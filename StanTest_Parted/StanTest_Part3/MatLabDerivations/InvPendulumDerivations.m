%[text] # Inv Pend Math
%%
%[text] ## Laplace Solution
clc
clear
syms Tau b theta(t) I_s I_w s g m t l
theta_dot = diff(theta,1,t);
theta_ddot = diff(theta_dot,1,t);
Is = I_s; Iw = I_w;
T = Tau;th = theta; thd = theta_dot;thdd = theta_ddot;
eqn1 = T - b*thd + g*m*l*th == Is*thdd %[output:919e7c6b]
eqn2 = laplace(eqn1,t,s) %[output:73b56516]
Y = solve(eqn2,th(t)) %[output:042f0a06]
% Assume theta(0) = 0
Y = subs(Y,th(0),0) %[output:497007ad]
TF = Y/T %[output:25e2fd19]
fprintf('The final transfer function utilized in the calculations is') %[output:3b7464fb]
display(TF) %[output:3cdd601c]
%%
%[text] ## State Space System
% Based off my calculations, Controllable Cannonical Form
A = [0 1;-g*m*l -b/Is] %[output:4d2c69cf]
C = [-1 0] %[output:0fc8c2dc]
B = [0;1] %[output:96c3dc0b]
D = 0 %[output:6d603c09]
A = double(subs(A,[m g l b Is], [ 1 2 3 4 5])) % insert example components %[output:5182fdea]
rank(ctrb(A,B)) %[output:9d0edf1e]
rank(obsv(A,C)) %[output:8a7244ef]
fprintf('The controlability and observailty matricies are full rank') %[output:2aef1fa0]

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":58.2}
%---
%[output:919e7c6b]
%   data: {"dataType":"symbolic","outputData":{"name":"eqn1(t)","value":"-b\\,\\frac{\\partial }{\\partial t}\\;\\theta \\left(t\\right)+T+g\\,l\\,m\\,\\theta \\left(t\\right)=I_s \\,\\frac{\\partial^2 }{\\partial t^2 }\\;\\theta \\left(t\\right)"}}
%---
%[output:73b56516]
%   data: {"dataType":"symbolic","outputData":{"name":"eqn2","value":"\\frac{T}{s}+b\\,{\\left(\\theta \\left(0\\right)-s\\,\\mathrm{laplace}\\left(\\theta \\left(t\\right),t,s\\right)\\right)}+g\\,l\\,m\\,\\mathrm{laplace}\\left(\\theta \\left(t\\right),t,s\\right)=-I_s \\,{\\left(s\\,\\theta \\left(0\\right)+{\\left({{\\left(\\frac{\\partial }{\\partial t}\\;\\theta \\left(t\\right)\\right)}\\left|\\right.}_{t=0} \\right)}-s^2 \\,\\mathrm{laplace}\\left(\\theta \\left(t\\right),t,s\\right)\\right)}"}}
%---
%[output:042f0a06]
%   data: {"dataType":"symbolic","outputData":{"name":"Y","value":"\\frac{I_s \\,\\theta \\left(0\\right)\\,s^2 +b\\,\\theta \\left(0\\right)\\,s+T}{I_s \\,s^2 +b\\,s-g\\,l\\,m}"}}
%---
%[output:497007ad]
%   data: {"dataType":"symbolic","outputData":{"name":"Y","value":"\\frac{T}{I_s \\,s^2 +b\\,s-g\\,l\\,m}"}}
%---
%[output:25e2fd19]
%   data: {"dataType":"symbolic","outputData":{"name":"TF","value":"\\frac{1}{I_s \\,s^2 +b\\,s-g\\,l\\,m}"}}
%---
%[output:3b7464fb]
%   data: {"dataType":"text","outputData":{"text":"The final transfer function utilized in the calculations is","truncated":false}}
%---
%[output:3cdd601c]
%   data: {"dataType":"symbolic","outputData":{"name":"TF","value":"\\frac{1}{I_s \\,s^2 +b\\,s-g\\,l\\,m}"}}
%---
%[output:4d2c69cf]
%   data: {"dataType":"symbolic","outputData":{"name":"A","value":"\\left(\\begin{array}{cc}\n0 & 1\\\\\n-g\\,l\\,m & -\\frac{b}{I_s }\n\\end{array}\\right)"}}
%---
%[output:0fc8c2dc]
%   data: {"dataType":"matrix","outputData":{"columns":2,"name":"C","rows":1,"type":"double","value":[["-1","0"]]}}
%---
%[output:96c3dc0b]
%   data: {"dataType":"matrix","outputData":{"columns":1,"name":"B","rows":2,"type":"double","value":[["0"],["1"]]}}
%---
%[output:6d603c09]
%   data: {"dataType":"textualVariable","outputData":{"name":"D","value":"0"}}
%---
%[output:5182fdea]
%   data: {"dataType":"matrix","outputData":{"columns":2,"name":"A","rows":2,"type":"double","value":[["0","1.0000"],["-6.0000","-0.8000"]]}}
%---
%[output:9d0edf1e]
%   data: {"dataType":"textualVariable","outputData":{"name":"ans","value":"2"}}
%---
%[output:8a7244ef]
%   data: {"dataType":"textualVariable","outputData":{"name":"ans","value":"2"}}
%---
%[output:2aef1fa0]
%   data: {"dataType":"text","outputData":{"text":"The controlability and observailty matricies are full rank","truncated":false}}
%---
