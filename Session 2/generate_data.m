function [data x_exact] = generate_data(R)

% generate_data.m

% This function generates the data for a 1D motion along a track that is
% divided in 5 segments: 
%  - standstill
%  - constant acceleration
%  - constant velocity
%  - constant deceleration
%  - standstill

% How many data points need to be generated?
length_segment = 100;
nSegments = 1;%5

NumberOfDataPoints = nSegments * length_segment;

T = 1; %[s]

a_set = [0 1 0 -1 0]; %[m/s^2]  [0 1 0 -1 0]; %[m/s^2]
a = repmat (a_set, length_segment, 1);
a = a(:);

x = zeros(3,NumberOfDataPoints); % als initialisatie
x(:,1) = [0; 0; a(1)];

A = [1 T T^2/2; 0 1 T; 0 0 1];

v = randn(1, NumberOfDataPoints) * sqrt(R);

for idx = 2:NumberOfDataPoints
    x(:,idx) = A * [x(1:2, idx-1); a(idx)];
end

x_exact = x/10;  % according to first example in task assignment
data = x(1,:)/10 + v;

% save data.mat data x_exact;