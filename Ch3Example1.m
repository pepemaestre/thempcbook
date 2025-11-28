clear; clc; close all;
A=[1 -0.8];
B=[0.4 0.6];
Np = 3;
Nc = 3;
Nsim = 12; 
lambda = 0.8;
G = buildG(A, B, Np, Nc);
H = inv(G'*G + lambda*eye(Nc))*G';
yhist = zeros(size(A,2),1);
uhist = [zeros(size(B,2),1)];
deltauhist = [zeros(size(B,2),1)];
r = ones(Np,1);
u = 0;

for k = 1:Nsim
    f = buildResponse(A, B, Np, yhist, [deltauhist;0]);
    U = H*(r-f);
    deltau = U(1);
    u = u + deltau;
    y = plant(A, B, u);
    deltauhist(end+1) = deltau;
    uhist(end+1) = u;
    yhist(end+1) = y;
end


%% Plot
% Define x-axis values starting from 0
x1 = 0:(length(yhist) - size(A,2));
x2 = 0:(length(deltauhist) - size(B,2) - 1);

% Plot with adjusted x-axis
plot(x1, yhist(size(A,2):end), 'DisplayName', 'yhist');
hold on;
plot(x2, deltauhist(size(B,2)+1:end), 'DisplayName', 'deltauhist');

% Add labels and legend
xlabel('Time Steps');
ylabel('Values');
legend('Location', 'Best');
grid on;


function yk = plant(A, B, uk)
    persistent xk Ad Bd Cd Dd
    if isempty(xk)
        xk = zeros(size(A, 1), 1);  % Initialize to zero the first time
        % Convert the transfer function to state-space
        Gz = tf(B, A, -1);
        [Ad, Bd, Cd, Dd] = ssdata(Gz);
    end
    
    % Update state and compute output
    yk = Cd * xk + Dd * uk;
    xk = Ad * xk + Bd * uk;
end


function G = buildG(A,B,Np,Nc)
    yhist = zeros(size(A,2),1);
    uhist = [zeros(size(B,2)-1,1); 1];
    g = buildResponse(A, B, Np, yhist, uhist);
    G = g;
    for i = 1:Nc-1
        g_aux = [zeros(i,1); g(1:Np-i)];
        G  = [G, g_aux];
    end
end

function ypred = buildResponse(A, B, Np, yhist, uhist)
    % ypred = [y(k+1); y(k+2);...; y(k+Np)]
    % A, B are polyn. describing the system
    % Np is pred. horizon
    % yhist = [...; y(k-1); y(k)]
    % uhist = [...; deltau(k-1); deltau(k)]
    ypred = zeros(Np,1);
    Avir = [A 0]-[0 A]; % (1-z^-1)A
    Na = size(A,2);  % past output needed
    Nb = size(B,2);  % past inputs needed 
    yhist = yhist(end:-1:end-Na+1); 
    uhist = uhist(end:-1:end-Nb+1);
    for j = 1 : Np 
        ypred(j)= -Avir(2:Na+1)*yhist + B(1:Nb)*uhist;
        yhist = [ypred(j); yhist(1:Na-1)];
        uhist = [ 0 ; uhist(1:Nb-1)];
    end
end


%% Alternatives for programming the computation of G
function G = buildGcoefbased(A, B, Np, Nc)
    g = zeros(Np,1);
    g(1) = B(1); % g(1) = b0

    % g(2) = -a1*g(1) + b0 + b1, etc.
    for j = 2 : Np
        for i = 2 : min(length(A), j)  
            g(j) = g(j) - A(i)*g(j - (i-1));
        end
        for i = 1 : min(length(B), j)
            g(j) = g(j) + B(i);
        end
    end

    G = g;
    for i = 1:Nc-1
        g_aux = [zeros(i,1); g(1:Np-i)];
        G  = [G, g_aux];
    end
end

function G=buildGcodeduardo(A,B,Np,Nc)
   % Building the matrix G of GPC
    G = zeros(Np,Nc);
    Avir = [A 0]-[0 A]; % (1-z^-1)A
    Na = size(A,2);  % past output needed for calculations
    yaux = zeros(Na,1);
    IncU = [1; zeros(Nc-1,1)]; % input is an impulse
    B    = [B zeros(1,Nc-size(B,2))];
    for j = 1 : Np
        if (j>1)*(Nc>1)  
          G(j,2:Nc) = G (j-1,1:Nc-1); 
        end 
        G(j,1)= -Avir(2:Na+1)*yaux + B(1:Nc)*IncU;
        yaux = [G(j,1) ; yaux(1:Na-1)];
        IncU = [0 ; IncU(1:Nc-1)];
    end
end

