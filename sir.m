clc; clear; close all; 


beta = 0.01;
gamma = 0.1;
tmax = 50;
tspan = [0 tmax];
N = 150;


%% model1
S0 = [N-1 1 0];
[t,S] = ode45(@(t,S) SIR1(t,S,beta,gamma), tspan, S0);
figure;
plot(t,S(:,1),'g-',t,S(:,2),'r-',t,S(:,3),'k-');
%legend('S(t)','I(t)','R(t)');
title('SIR 1');

%% discrete model 1
dt = 0.1;
S0 = [N-1; 1; 0];
St = S0;

i = 1;
St_hist = zeros(tmax/dt,3);
St_hist(i, :) = S0;
T = 0:dt:50;
for t = T
    St = D_SIR1_step(St,beta,gamma,dt);   
    St_hist(i, :) = St;
    i = i+1;
end
%figure;
hold on;
plot(T,St_hist(:,1), 'g--'); 
plot(T,St_hist(:,2), 'r--'); 
plot(T,St_hist(:,3), 'k--');
legend('$S(t)$','$I(t)$','$R(t)$', '$S_k$', '$I_k$', '$R_k$', 'Interpreter','latex');
title('SIR', 'Interpreter','latex');
xlabel('t, k', 'Interpreter','latex');


function St = D_SIR1_step(S,beta,gamma,dt)
    St = zeros(3,1);
    St(1) = S(1) - beta*S(1)*S(2)*dt;
    St(2) = S(2) + beta*S(1)*S(2)*dt - gamma*S(2)*dt;
    St(3) = S(3) + gamma*S(2)*dt; 
end

function dSdt = SIR1(t,S,beta,gamma)
    ds = -beta*S(1)*S(2);
    di = beta*S(1)*S(2) - gamma*S(2);
    dr = gamma*S(2);
    dSdt = [ds;di;dr];
end