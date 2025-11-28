clear all; 
close all; 
clc;

% Simulation parameters
g = 9.81;             % Gravitational acceleration (m/s^2)
m = 0.05;              % Mass of the object (kg)
k = 1.0;              % Magnetic force constant
Ts = 0.005;            % Sampling time
Tf = 3;               % Final simulation time (seconds)
numSteps = Tf / Ts;   % Number of simulation steps

% NMPC setup
Np = 30;              % Prediction horizon
Q = 100;              % Position error weight
R = 1;                % Control effort weight
setpoint = 0.1;       % Target position (m)
uMin = 0;             % Minimum input 
uMax = 5;             % Maximum input 
options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');
parameters = [ Np, Ts, Q, R, setpoint, g, m, k];

% Initialize state and control history
x = [0.15; 0];        % Initial state
xHistory = x;         % To store state evolution over time
uHistory = [];        % To store control inputs over time

% Simulation loop
u_0 = zeros(Np, 1); % Initial guess for control inputs over the horizon
for l = 1:numSteps
    clc
    error = setpoint - x(1);
    errorrel = 100*error/setpoint;
    fprintf('%d/%d %f %f',l,numSteps,error, errorrel);

    % Set up optimization problem for current step

    % Optimize control sequence with fmincon
    u_opt = fmincon(@(U) costFunction(U, x, parameters), ...
                         u_0, [], [], [], [],...
                         uMin * ones(Np, 1), uMax * ones(Np, 1), ...
                         [], options);
    %u_0(1:Np-1) = u_opt(2:end);
    %u_0(end) = u_opt(end);
    u_0 = [u_opt(2:end); u_opt(end)];

    % Update system state using nonlinear dynamics (Euler integration)
    % x = x + Ts * stateFcn(x, u_opt(1), parameters);
    % Update system state using nonlinear dynamics
    [~, xout] = ode45(@(t, xk) stateFcn(xk, u_opt(1), parameters), [0 Ts], x);
    x = xout(end, :)';
    
    % Store history for plotting
    xHistory = [xHistory, x];
    uHistory = [uHistory, u_opt(1)];
end

% Time vector
time = 0:Ts:Tf;
time_control = 0:Ts:Tf-Ts; % For control input plot (one less step)

%% Create figure for position and control input plots
figure;

% Position plot
subplot(2, 1, 1);
plot(time, xHistory(1, :), 'k-', 'LineWidth', 0.8); % Position trajectory
hold on;
yline(setpoint, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8); % Target position as dashed line
%title('Object Position');
%ylim([0.05 0.15])
xlabel('Time (s)');
ylabel('Position (m)');
%legend('Position', 'Setpoint');
grid on;
set(gca, 'GridLineStyle', '--', 'LineWidth', 0.5); % Neat dashed grid lines

% Control input plot
subplot(2, 1, 2);
plot(time_control, uHistory, 'k-', 'LineWidth', 0.8); % Control input trajectory
%title('Current');
xlabel('Time (s)');
ylabel('Current (A)');
%legend('Control Input');
grid on;
set(gca, 'GridLineStyle', '--', 'LineWidth', 0.5); % Neat dashed grid lines

% Adjust layout
set(gcf, 'Position', [100, 100, 800, 600]); % Optional: set figure size

% Cost function
function J = costFunction(U, x, parameters)
    J = 0; % Initialize cost
    xk = x; % Initialize predicted state

    Np = parameters(1);
    Ts = parameters(2);
    Q = parameters(3);
    R = parameters(4);
    targetPos = parameters(5);
    
    for i = 1:Np
        u = U(i);
        xk1 = xk + Ts * stateFcn(xk, u, parameters);  % Predict next state
        posError = (xk1(1) - targetPos)^2;
        controlEffort = u^2;
        J = J + Q * posError + R * controlEffort;
        xk = xk1; % Update state for next step
    end
end

% Dynamics function for maglev system
function f = stateFcn(x, u, parameters)
    % Unpack parameters
    g = parameters(6);
    m = parameters(7);
    k = parameters(8);

    pos = x(1);    % Position
    vel = x(2);    % Velocity
    f = [vel; g - (k * u^2) / (m * pos^2)];
end