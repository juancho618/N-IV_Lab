function [data, x_exact] = generate_data_2D(Q, R)


% This function generates the motion data for a 2D motion along a
% track that is divided in 5 segments, the used model is constant v:
%  - (200,-100)  to (100,100)
%  - (100,100)   to (100,300)
%  - (100,300)   to (-200,300)
%  - (-200,300)  to (-200,-200)
%  - (-200,-200) to (0,0)
% Everything + (300,300) to assure operation in first quadrant

plot_grafs = true;
points = [200, -100; 100, 100; 100, 300; -200, 300; -200, -200; 0,0];
dist = (diff(points)).^2; dist = round(sqrt(dist(:,1) + dist(:,2)));

% How many data points need to be generated?
length_segment = dist;
nSegments = 5;

NumberOfDataPoints = sum(length_segment);

T = 0.5; %[s]

v_set = 2 * [-0.4472, 0.8944; 0, 1; -1, 0; 0, -1; sqrt(2)/2, sqrt(2)/2]; % 2 m/s

v = [];
for ctr = 1:nSegments
    v = [v; (repmat(v_set(ctr,:), length_segment(ctr), 1))];
end


% ==motion generation====================================================

%x(:,1) = [200; 0; -100; 0];
x(:,1) = [500; 0; 200; 0];

A = [1 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 0];
B = [T 0; 1 0; 0 T;0 1];

G = [T.^2/2 0; T 0; 0 T.^2/2; 0 T];


if (length(Q) == 1), Q = [Q; Q]; end
w_x = randn(1, NumberOfDataPoints) * sqrt(Q(1)); %acceleration noise in x-direction
w_y = randn(1, NumberOfDataPoints) * sqrt(Q(2)); %acceleration noise in y-direction
w = [w_x; w_y];

for idx = 2:NumberOfDataPoints
    x(:,idx) = A * x(:, idx-1) + B * v(idx,:)' + G * w(:,idx);
end

if plot_grafs
    figure;
    plot(x(1,:), x(3,:), 'x');
    title('trajectory');
    xlabel('x-axis [m]'); ylabel('y-axis [m]');
    
end

x_exact = x;


% ==measurement generation===============================================
position = x([1,3],:);

% Noiseless measurement
data = zeros(NumberOfDataPoints,2);
for idx = 1:NumberOfDataPoints
    data(idx,1) = sqrt(position(:,idx)' * position(:,idx));
    data(idx,2) = atan2(position(2,idx), position(1,idx));
end

% to avoid problems
data(:,2) = unwrap(data(:,2));

if (length(R) == 1), R = [R; R]; end
v_r = randn(1, NumberOfDataPoints) * sqrt(R(1));
v_a = randn(1, NumberOfDataPoints) * sqrt(R(2));
v_meas = [v_r; v_a]';

data_exact = data';
data = data_exact + v_meas';

if plot_grafs
    figure; xlab ={'Radius [m]', 'Angle [rad]'};
    for idx = 1:2
        subplot(2,1,idx); hold on;
        plot(data_exact(idx,:), 'b'); plot(data(idx,:), 'xr');
        legend('Exact', 'Measurement (noisy)');
        xlabel('Time step []'); ylabel(xlab{idx});
    end
end

% save data.mat data x_exact;