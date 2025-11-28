clear; clc; close all;

% Model Matrices
%A=[1 -0.8351];
%B=[0 0 0 0.2731];
%Process Matrices:
%Ap=[1 -0.8];
%Bp=[0 0 0 0.28];

%Process 
z = tf('z');
Gz =  0.2713 * z^(-3) / (1 - 0.8351 * z^(-1)); 
[Ad, Bd, Cd, Dd] = ssdata(Gz);  % for state-space iteration
xk = zeros(size(Ad,1),1);  % plant states

%% 1) Step-response coefficients g_i (N=30)
g = [ 0.000  0.000  0.271  0.498  0.687  0.845  0.977  1.087  1.179  1.256 ...
    1.320  1.374  1.419  1.456  1.487  1.513  1.535  1.553  1.565  1.581 ...
    1.592  1.600  1.608  1.614  1.619  1.623  1.627  1.630  1.633  1.635 ]';

N = 30;     % total # of known step-response coeffs
Np = 10;    % prediction horizon
Nc = 5;     % control horizon

%% 2) Simulation length = 120 steps
nSim = 120;

% We'll log output y, input u, alpha, lambda, reference
yData   = zeros(nSim,1);
uData   = zeros(nSim,1);
alphaData = kron([0 0.7 0 0.7], ones(1,30));
lambdaData = kron([1 0.1], ones(1,60));
refData = kron([1 0 1 0], ones(1,30));

% Keep track of last N=30 increments:
DeltaU_history = zeros(N,1);
U_history = zeros(N,1);

% Initialize
yk = 0;
uk = 0;

%% 3) Main Loop
for k = 1:nSim

    % 3.1) store current output & input
    yData(k) = yk;
    uData(k) = uk;

    % 3.2) Build the reference trajectory rVec for the next Np steps
    %      (1st term uses yk as the "current" output, plus alpha as a 
    %       smoothing factor toward refData(k))
    rVec = zeros(Np,1);
    rVec(1) = alphaData(k)*yk + (1 - alphaData(k))*refData(k);
    for j = 2:Np
        rVec(j) = alphaData(k)*rVec(j-1) + (1 - alphaData(k))*refData(k);
    end

    % 3.3) Compute free response fVec using "calculateF"
    %      f(t+k) = y(t) + sum_{i=1..N}[ (g(k+i)-g(i)) * DeltaU(t-i) ]
    fVec = calculateF(Np, yk, N, g, DeltaU_history);

    % 3.4) Build the dynamic matrix G and compute L row for this lambdaData(k)
    [G, Lrow] = calculateGL(g, Np, Nc, lambdaData(k));

    % 3.5) Compute Delta_u(t) = Lrow*( rVec - fVec )
    Delta_u = Lrow * (rVec - fVec);

    % 3.6) Update total control
    uk = uk + Delta_u;

    % 3.7) Shift increments
    DeltaU_history = [DeltaU_history(2:end); Delta_u];

    % 3.8) Advance
    yk = Cd*xk + Dd*uk;
    xk = Ad*xk + Bd*uk;
    
    
end

%% 4) Plots
figure('Name','Water-Heater: 120-step Combined Scenario','Color','w',...
       'Position',[100 100 900 600]);

subplot(2,1,1)
plot(1:nSim, yData, 'b-o','LineWidth',1.1); hold on;
plot(1:nSim, refData, 'r--','LineWidth',1.2);
xlabel('k (samples)'); ylabel('y(k)');
title('Output vs. Reference');
legend('y','r_{sp}','Location','Best');
grid on;

subplot(2,1,2)
plot(1:nSim, uData, 'k-o','LineWidth',1.1);
xlabel('k (samples)'); ylabel('u(k)');
title('Control Input');
grid on;

%% ---------------- LOCAL FUNCTIONS (included below in the same .m script)
function [G, L] = calculateGL(g, Np, Nc, lambda)
    G = g(1:Np, 1);
    for i = 1:Nc-1
        g_aux = [zeros(i,1); g(1:Np-i)];
        G  = [G, g_aux];
    end
    M = inv(G'*G + lambda * eye(Nc))*(G');
    L = M(1,:);
end

function f = calculateF(Np, ym, N, g, deltauhist)
    f = ym*ones(Np, 1); 
    for k = 1:Np
        for i = 1:N
             f(k) = f(k) + (g(min(k+i,N)) - g(i)) * deltauhist(end-i+1);
        end
    end
end