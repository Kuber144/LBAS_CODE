clear;
clc;
close all;

% Network Parameters
numSensors = 200; % Number of sensors
xMax = 100; % Area dimensions
yMax = 100;
commRange = 20; % Communication range
numClusters = round(numSensors * 0.10); % Approximately 10% of nodes as CHs
dataSize = 10; % Size of data packets (units)
energyPerDataUnit = 0.5; % Energy consumed per unit of data transmission
p = 0.1; % Desired percentage of CHs (e.g., 10% of nodes are CHs)
rounds = 100; % Total number of rounds for simulation

% Exponential distribution parameters
lambda_x = 1; % Rate parameter for x-coordinates
lambda_y = 1; % Rate parameter for y-coordinates

% Generate random positions for sensors using exponential distribution
sensorPositions_x = exprnd(1/lambda_x, numSensors, 1) * xMax; % Scale to fit xMax
sensorPositions_y = exprnd(1/lambda_y, numSensors, 1) * yMax; % Scale to fit yMax
sensorPositions = [sensorPositions_x, sensorPositions_y];

% Clip positions to remain within the area boundaries
sensorPositions(:, 1) = min(max(sensorPositions(:, 1), 0), xMax);
sensorPositions(:, 2) = min(max(sensorPositions(:, 2), 0), yMax);

% Assign random attributes to sensors (e.g., energy levels and capabilities)
sensorEnergy = rand(numSensors, 1) * 100; % Energy levels between 0 and 100
initialEnergy = sensorEnergy; % Initial energy
sensorCapabilities = randi([1, 5], numSensors, 1); % Capabilities between 1 and 5
% Plot the distribution of nodes
figure;
scatter(sensorPositions(:, 1), sensorPositions(:, 2), 'filled');
xlabel('X Coordinate');
ylabel('Y Coordinate');
title('Distribution of Sensor Nodes');
xlim([0 xMax]);
ylim([0 yMax]);
grid on;
% Compute distance to Base Station (BS)
BS_position = [xMax / 2, yMax / 2]; % BS position at center
distToBS = sqrt((sensorPositions(:, 1) - BS_position(1)).^2 + (sensorPositions(:, 2) - BS_position(2)).^2);
maxDistToBS = max(distToBS);

% Compute the number of neighbors for each node
neighborCounts = zeros(numSensors, 1);
for i = 1:numSensors
    distances = sqrt((sensorPositions(:, 1) - sensorPositions(i, 1)).^2 + (sensorPositions(:, 2) - sensorPositions(i, 2)).^2);
    neighborCounts(i) = sum(distances <= commRange) - 1; % Exclude itself
end

% Function to calculate impact values
function weights = calculateWeights(distToBS, maxDistToBS, neighborCounts, sensorEnergy, initialEnergy, p, round)
    numSensors = length(distToBS);
    % Distance Factor (D_f)
    D_f = distToBS / maxDistToBS;

    % Node Density Factor (N_f)
    N_f = neighborCounts / numSensors;

    % Energy Factor (E_f)
    E_f = sensorEnergy ./ initialEnergy;

    % Probability Factor (T_f)
    T_f = (p * rand(numSensors, 1)) ./ (1 - p * mod(round, 1 / p));

    % Weights Calculation
    % Coefficients (You can adjust these according to importance)
    alpha = 0.25; % Weight for Distance Factor
    beta = 0.25;  % Weight for Node Density Factor
    gamma = 0.25; % Weight for Energy Factor
    delta = 0.25; % Weight for Probability Factor

    weights = alpha * (1 - D_f) + beta * N_f + gamma * E_f + delta * T_f; % Note (1 - D_f) to give higher weight for closer nodes
end

% Function to select cluster heads based on probabilistic approach
function [clusterHeads, clusterAssignments] = selectClusterHeads(sensorPositions, weights, numClusters, round, p)
    numSensors = size(sensorPositions, 1);
    clusterHeads = NaN(numClusters, 2);
    clusterAssignments = zeros(numSensors, 1);

    % Probability Factor
    probabilities = (p * rand(numSensors, 1)) ./ (1 - p * mod(round, 1 / p));
    
    % Sort nodes by their impact values
    [~, sortedIndices] = sort(weights, 'descend');
    
    % Select top nodes based on sorted impact values
    topIndices = sortedIndices(1:numSensors);
    
    % Initialize the number of CHs selected
    numSelected = 0;
    
    % Try to select cluster heads
    for i = 1:numSensors
        idx = topIndices(i);
        if rand() < probabilities(idx) % Use probability to select CH
            if numSelected < numClusters
                clusterHeads(numSelected + 1, :) = sensorPositions(idx, :);
                numSelected = numSelected + 1;
            else
                break;
            end
        end
    end

    % Ensure at least numClusters CHs are selected
    if numSelected < numClusters
        % If not enough CHs are selected, randomly select from remaining nodes
        remainingIndices = setdiff(1:numSensors, topIndices(1:numSelected));
        additionalHeads = remainingIndices(1:(numClusters - numSelected));
        clusterHeads(numSelected + 1:numClusters, :) = sensorPositions(additionalHeads, :);
    end

    % Update cluster assignments based on selected cluster heads
    for i = 1:numSensors
        if any(~isnan(clusterHeads(:,1))) % Ensure that newClusterHeads is not empty
            distances = arrayfun(@(c) distance(sensorPositions(i, :), clusterHeads(c, :)), 1:numClusters);
            [~, clusterAssignments(i)] = min(distances);
        end
    end
end

% Function to calculate average load
function [lambda, loadMatrix] = calculateLoadMatrix(clusterAssignments, weights, numClusters)
    numSensors = length(clusterAssignments);
    loadMatrix = zeros(numSensors, numClusters);

    % Calculate the average load per cluster
    avgLoad = mean(weights);
    
    % Calculate load difference b = l - avgLoad
    loadDifference = weights - avgLoad;

    % Define matrix A for least-squares solution
    A = eye(numSensors); % Simplistic matrix for illustration; adjust based on problem specifics

    % Define vector b for least-squares solution
    b = loadDifference;

    % Solve for λ using least-squares
    lambda = A \ b;

    % Calculate the load transfer matrix δ
    loadMatrix = loadDifference - lambda;
end

% Function to reconfigure cluster heads based on load balancing
function [newClusterHeads, clusterAssignments] = reconfigureClusterHeads(sensorPositions, weights, clusterAssignments, numClusters, numSensors)
    % Calculate average load
    averageLoad = mean(weights); % Assuming weights reflect load
    loadDifference = weights - averageLoad;

    % Initialize new cluster heads
    newClusterHeads = NaN(numClusters, 2);

    % Initialize load transfer matrix
    loadMatrix = zeros(numSensors, numClusters);
    
    % Update loadMatrix with load differences
    for i = 1:numSensors
        clusterIdx = clusterAssignments(i);
        if clusterIdx > 0
            loadMatrix(i, clusterIdx) = loadDifference(i);
        end
    end

    % Solve for λ using least-squares or other method
    % Construct A and b for solving lambda
    A = eye(numSensors);
    b = loadDifference;

    % Solve the least-squares problem
    lambda = A \ b;

    % Calculate the load transfer matrix δ
    delta = loadMatrix - lambda;

    % Reconfigure cluster heads based on load
    for c = 1:numClusters
        % Get the nodes in the current cluster
        clusterNodes = find(clusterAssignments == c);

        % Skip if no nodes are assigned to this cluster
        if isempty(clusterNodes)
            continue;
        end

        % Determine nodes with high load
        highLoadNodes = clusterNodes(delta(clusterNodes, c) > 0);

        % Determine nodes with moderate or low load
        moderateOrLowLoadNodes = clusterNodes(delta(clusterNodes, c) <= 0);

        % Reconfigure if the current CH is a high-load node
        currentCH = find(clusterAssignments == c & ~isnan(clusterAssignments), 1);
        if ~isempty(currentCH)
            % Reconfigure if high load or moderate load with a better candidate
            if any(delta(currentCH, c) > 0) || ...
               (any(delta(moderateOrLowLoadNodes, c) > 0) && ...
               any(weights(moderateOrLowLoadNodes) > weights(currentCH)))
                
                % Find the best candidate for new CH
                if ~isempty(highLoadNodes)
                    bestCandidate = highLoadNodes(1); % Simplify for demonstration
                else
                    bestCandidate = moderateOrLowLoadNodes(1); % Simplify for demonstration
                end

                % Replace current CH
                if ~isempty(bestCandidate)
                    newClusterHeads(c, :) = sensorPositions(bestCandidate, :);
                end
            else
                % Maintain current CH
                newClusterHeads(c, :) = sensorPositions(currentCH, :);
            end
        end
    end
end

% Simulation over rounds
totalEnergy = zeros(rounds, 1);
fnd = zeros(rounds, 1);
hnd = zeros(rounds, 1);
lnd = zeros(rounds, 1);
% Initialize arrays to store time taken for each round
fndTime = zeros(rounds, 1);

for round = 1:rounds
    % Calculate weights based on current state
    weights = calculateWeights(distToBS, maxDistToBS, neighborCounts, sensorEnergy, initialEnergy, p, round);
    
    % Select cluster heads based on weights
    [clusterHeads, clusterAssignments] = selectClusterHeads(sensorPositions, weights, numClusters, round, p);
    
    % Calculate load matrix and balance loads
    [lambda, loadMatrix] = calculateLoadMatrix(clusterAssignments, weights, numClusters);
    
    % Reconfigure cluster heads based on load
    [newClusterHeads, clusterAssignments] = reconfigureClusterHeads(sensorPositions, weights, clusterAssignments, numClusters, numSensors);
    
    % Update energy levels based on transmission (simplified for demonstration)
    energyConsumption = dataSize * energyPerDataUnit; % Energy consumption per data transmission
    sensorEnergy = sensorEnergy - energyConsumption * (clusterAssignments > 0); % Reduce energy for nodes that transmitted
    
    % Calculate and time FND
    tic; % Start timing for FND
    fnd(round) = sum(sensorEnergy <= 0 & initialEnergy > 0); % FND
    fndTime(round) = toc; % Store time taken for FND calculation
    
    % Store other statistics
    totalEnergy(round) = sum(sensorEnergy);
    hnd(round) = sum(sensorEnergy < 0 & initialEnergy > 0); % HND
    lnd(round) = sum(sensorEnergy < 0); % LND
end

% Visualization of FND time
figure;
plot(1:rounds, fndTime, '-o');
xlabel('Rounds');
ylabel('Time for FND Calculation (seconds)');
title('Time Taken to Calculate FND over Rounds');
grid on;

% Visualization for FND, HND, and LND
figure;
bar3([fnd, hnd, lnd]);
set(gca, 'XTickLabel', {'FND', 'HND', 'LND'}, 'YTickLabel', num2cell(1:rounds), 'ZTickLabel', []);
xlabel('Metrics');
ylabel('Rounds');
zlabel('Count');
title('FND, HND, and LND over Rounds');
grid on;


% Visualization for FND, HND, and LND
figure;
bar3([fnd, hnd, lnd]);
set(gca, 'XTickLabel', {'FND', 'HND', 'LND'}, 'YTickLabel', num2cell(1:rounds), 'ZTickLabel', []);
xlabel('Metrics');
ylabel('Rounds');
zlabel('Count');
title('FND, HND, and LND over Rounds');
grid on;

% Visualization of HND over rounds
figure;
plot(1:rounds, hnd, '-s', 'MarkerFaceColor', 'b');
xlabel('Rounds');
ylabel('Half Node Deaths (HND)');
title('Half Node Deaths (HND) over Rounds');
grid on;

% Visualization of LND over rounds
figure;
plot(1:rounds, lnd, '-d', 'MarkerFaceColor', 'r');
xlabel('Rounds');
ylabel('Last Node Deaths (LND)');
title('Last Node Deaths (LND) over Rounds');
grid on;
