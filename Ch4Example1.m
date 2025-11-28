clear; clc; close all;

%% 1) System Definition (Discrete-Time)
% The continuous-time model from the example has already been discretized 
% at Ts=0.1 s, yielding (A,B) plus output matrix C. Disturbances (u_w, w_w)
% are omitted in this nominal simulation for clarity.

A = [ 0.9996,   0.0383,  0.0131, -0.0322;
     -0.0056,   0.9647,  0.7446,  0.0001;
      0.0020,  -0.0097,  0.9543,  0.0000;
      0.0001,  -0.0005,  0.0978,  1.0000];

B = [  0.0001,  0.1002;
      -0.0615,  0.0183;
      -0.1133,  0.0586;
      -0.0057,  0.0029];

C = [  1,      0,     0,    0;    % y1 = airspeed = (u - u_w) approx here
       0,     -1,     0,  7.74]; % y2 = climb rate = -w + 774 * theta

% Number of states and inputs
nx = size(A,1); 
nu = size(B,2); 
ny = size(C,1);

% We assume no direct feedthrough (D=0)
D = zeros(ny, nu);

%% 2) MPC Parameters
Np = 30;          % Prediction horizon
Nc = 10;          % Control horizon (number of moves)
Qy = diag([1, 1]);% Output error weighting for (y1, y2)
Ru = diag([5, 5]);% Input effort weighting for (elevator, throttle)

%  (Optional) Later we can try Ru = diag([10,1]) to emphasize y1 more.

%% 3) Build the augmented matrices for the standard SS-MPC formulation
% We want J = sum ( y - r )^T Q ( y - r ) + u^T R u  over Np steps
% Using the provided function: [Gu, fx, boldQ, boldR] = SSMPCmatrices(...)
[Gu, fx, bigQ, bigR] = SSMPCmatrices(A, B, C, Qy, Ru, Np);

% The dimension of the final optimization vector is Np*nu for unconstrained MPC.
% We will implement a “move blocking” style or simply apply the first Nc moves
% and keep the rest zero (one approach). For simplicity, we show an unconstrained 
% approach but only the first move is applied in closed loop (receding horizon).

%% 4) Simulation Settings
Tf = 50;     % total simulation time (in steps)
x  = zeros(nx, 1); % initial states
u  = zeros(nu, 1); % elevator & throttle commands
y  = zeros(ny, 1); % outputs

% Store data for plotting
Xlog = zeros(nx, Tf+1); 
Ylog = zeros(ny, Tf+1);
Ulog = zeros(nu, Tf);

Xlog(:,1) = x;  % log initial state
Ylog(:,1) = C*x; 

% Define setpoints for each output
% For t < 10 steps, r=[10;0], then at t=10 -> r=[10;5], or similar
% as in Figure 4.1 
rVec = zeros(ny, Tf+1); 
for k=1:1:Tf+1
   if k <= 10
       rVec(:,k) = [10; 0];   % airspeed=10 ft/s, climb=0
   else
       rVec(:,k) = [10; 5];   % climb changes to 5 ft/s at t=10
   end
end

%% 5) Main Closed-Loop Simulation
for k=1:1:Tf
    
    % Build the "reference" vector for the next Np steps (stacking)
    rStack = [];
    for j=1:Np
        if (k+j) <= (Tf+1)
            rStack = [rStack; rVec(:,k+j)]; 
        else
            % hold last reference if we go beyond horizon
            rStack = [rStack; rVec(:,Tf+1)];
        end
    end
    
    % Calculate the predicted output offset: fx*x
    fx_x = zeros(ny*Np, 1);
    for j=1:Np
       fx_x( (j-1)*ny+1 : j*ny ) = C*(A^j)*x;
    end
    
    % Solve unconstrained MPC problem for the next Np moves
    % The cost function in stacked form is: J = (rStack - fx_x - Gu*uVec)' Q (rStack - fx_x - Gu*uVec) + uVec' R uVec
    % So: uOpt = (Gu'Q Gu + R)^-1 Gu' Q ( rStack - fx_x )
    % bigQ, bigR are repeated forms of Qy, Ru
    Hu = Gu;  % rename for clarity
    Qbar = bigQ;
    Rbar = bigR;
    
    % since we only apply the "first" input in the horizon, dimension is Np*nu
    % We do not incorporate Nc < Np in a fancy way here, but you could 
    % do move-blocking or other advanced schemes if desired.
    
    H = (Hu' * Qbar * Hu + Rbar); 
    f = (Hu' * Qbar) * (rStack - fx_x);
    
    % unconstrained solution
    uVec = H \ f;  % size is Np*nu by 1
    
    % the actual input to apply is the first "nu" entries of uVec
    %  for MIMO with nu=2, we take uVec(1:2)
    uMPC = uVec(1:nu);
    
    % apply the input and simulate the system one step
    u = uMPC;   % elevator & throttle
    x = A*(x) + B*(u); 
    y = C*(x);
    
    % log data
    Xlog(:,k+1) = x;
    Ylog(:,k+1) = y;
    Ulog(:,k)   = u;
end

%% 6) Plot Results
time = 0:1:Tf;
time2= 0:1:Tf;  % for outputs

figure('Name','Aircraft MPC Simulation','Color',[1 1 1]);

subplot(2,1,1)
plot(time2, Ylog(1,:), 'b-', 'LineWidth',1.5); hold on;
plot(time2, Ylog(2,:), 'r-', 'LineWidth',1.5);
plot(time2, rVec(1,1:Tf+1), 'b--','LineWidth',1); 
plot(time2, rVec(2,1:Tf+1), 'r--','LineWidth',1);
grid on; xlabel('Time (samples)'); 
ylabel('Outputs'); 
legend('Airspeed (ft/s)','Climb Rate (ft/s)','r1','r2','Location','Best');
title('Outputs vs. setpoints');

subplot(2,1,2)
plot(Ulog(1,:), 'm-', 'LineWidth',1.5); hold on;
plot(Ulog(2,:), 'k-', 'LineWidth',1.5);
grid on; xlabel('Time (samples)'); 
ylabel('Control Inputs');
legend('Elevator','Throttle','Location','Best');
title('Control actions (elevator & throttle)');

sgtitle('Boeing 747 Flight Control - State-Space MPC Example');

