clear;
clc;
close all;

% Network Parameters
numSensors = 100; % Number of sensors
xMax = 100; % Area dimensions
yMax = 100;
commRange = 20; % Communication range
numClusters = round(numSensors * 0.10); % Approximately 10% of nodes as CHs
dataSize = 10; % Size of data packets (units)
energyPerDataUnit = 0.5; % Energy consumed per unit of data transmission
p = 0.1; % Desired percentage of CHs (e.g., 10% of nodes are CHs)
rounds = 100; % Total number of rounds for simulation

% Generate random positions for sensors
sensorPositions = [xMax * rand(numSensors, 1), yMax * rand(numSensors, 1)];

% Assign random attributes to sensors (e.g., energy levels and capabilities)
sensorEnergy = rand(numSensors, 1) * 100; % Energy levels between 0 and 100
initialEnergy = sensorEnergy; % Initial energy
sensorCapabilities = randi([1, 5], numSensors, 1); % Capabilities between 1 and 5

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

% Function to check Cluster Head energy and re-elect if necessary
function [newClusterHeads, clusterAssignments] = checkAndReelectCHs(sensorPositions, weights, clusterHeads, clusterAssignments, sensorEnergy, numClusters)
    numSensors = size(sensorPositions, 1);
    reelectFlag = false(numClusters, 1);  % Track which clusters need re-election
    
    % Initialize newClusterHeads to current cluster heads
    newClusterHeads = clusterHeads; 
    
    % Check each CH's energy
    for c = 1:numClusters
        if ~isnan(clusterHeads(c,1))
            % Get the CH index
            currentCH = find(ismember(sensorPositions, clusterHeads(c, :), 'rows'));

            % Check if this CH has depleted energy
            if sensorEnergy(currentCH) <= 0
                reelectFlag(c) = true;  % Mark this cluster for re-election
            end
        end
    end
    
    % Re-elect Cluster Heads if necessary
    if any(reelectFlag)
        for c = find(reelectFlag)'  % Loop through clusters that need re-election
            % Get all nodes in this cluster
            clusterMembers = find(clusterAssignments == c);
            
            % Select a new CH based on weights
            if ~isempty(clusterMembers)
                [~, bestNode] = max(weights(clusterMembers));
                newCH = clusterMembers(bestNode);
                newClusterHeads(c, :) = sensorPositions(newCH, :);
            else
                newClusterHeads(c, :) = NaN; % If no valid members, CH is NaN
            end
        end
        
        % Update cluster assignments with new CHs
        for i = 1:numSensors
            validClusters = find(~isnan(newClusterHeads(:,1))); % Only consider non-NaN cluster heads
            if ~isempty(validClusters) % Ensure that newClusterHeads is not empty
                distances = arrayfun(@(c) distance(sensorPositions(i, :), newClusterHeads(c, :)), validClusters);
                [~, closestClusterIdx] = min(distances);
                clusterAssignments(i) = validClusters(closestClusterIdx); % Assign to the closest valid CH
            end
        end
    end
end

% Function to reconfigure cluster heads based on load balancing
function [newClusterHeads, clusterAssignments] = reconfigureClusterHeads(sensorPositions, weights, clusterAssignments, numClusters, numSensors)
    % Calculate average load
    averageLoad = mean(weights); % Assuming weights reflect load
    loadDifference = weights - averageLoad;

    % Initialize new cluster heads
    newClusterHeads = NaN(numClusters, 2);

    % Update loadMatrix with load differences
    for c = 1:numClusters
        clusterMembers = find(clusterAssignments == c);
        if ~isempty(clusterMembers)
            [~, bestNode] = max(weights(clusterMembers));
            newClusterHeads(c, :) = sensorPositions(clusterMembers(bestNode), :);
        end
    end

    % Update cluster assignments based on new cluster heads
    for i = 1:numSensors
        if any(~isnan(newClusterHeads(:,1))) % Ensure that newClusterHeads is not empty
            distances = arrayfun(@(c) distance(sensorPositions(i, :), newClusterHeads(c, :)), 1:numClusters);
            [~, clusterAssignments(i)] = min(distances);
        end
    end
end

% Main Simulation Loop
totalEnergy = zeros(rounds, 1);
avgSensorEnergy = zeros(rounds, 1);

for r = 1:rounds
    % Calculate weights (using existing method)
    weights = calculateWeights(distToBS, maxDistToBS, neighborCounts, sensorEnergy, initialEnergy, p, r);

    % Select initial cluster heads (your existing method)
    if r == 1
        [clusterHeads, clusterAssignments] = selectClusterHeads(sensorPositions, weights, numClusters, r, p);
    end

    % Simulate data transmission and energy consumption
    energyConsumed = dataSize * energyPerDataUnit;
    
    % Increase energy consumption for CHs
    for i = 1:numClusters
        if ~isnan(clusterHeads(i,1))
            currentCH = find(ismember(sensorPositions, clusterHeads(i, :), 'rows'));
            sensorEnergy(currentCH) = sensorEnergy(currentCH) - 2 * energyConsumed;  % CHs use more energy
        end
    end
    
    % Regular nodes consume energy
    sensorEnergy = sensorEnergy - (energyConsumed * (clusterAssignments > 0));

    % Reconfigure cluster heads based on load balancing (your existing logic)
    [clusterHeads, clusterAssignments] = reconfigureClusterHeads(sensorPositions, weights, clusterAssignments, numClusters, numSensors);

    % Check for CH failure and re-election
    [clusterHeads, clusterAssignments] = checkAndReelectCHs(sensorPositions, weights, clusterHeads, clusterAssignments, sensorEnergy, numClusters);

    % Calculate total and average energy consumption
    totalEnergy(r) = sum(initialEnergy - sensorEnergy);
    avgSensorEnergy(r) = mean(sensorEnergy);

    % Visualize the results at certain rounds
    if r == 1 || r == rounds || mod(r,10)==0
        figure;
        hold on;
        colors = lines(numClusters); % Generate distinct colors for clusters
        for c = 1:numClusters
            % Cluster members
            clusterMembers = find(clusterAssignments == c);
            scatter(sensorPositions(clusterMembers, 1), sensorPositions(clusterMembers, 2), 50, colors(c, :), 'o', 'filled', 'DisplayName', sprintf('Cluster %d', c));

            % Cluster heads
            scatter(clusterHeads(c, 1), clusterHeads(c, 2), 100, 'k', 'x', 'LineWidth', 1.5, 'DisplayName', sprintf('CH %d', c));
        end
        title(sprintf('Round %d', r));
        xlabel('X Position');
        ylabel('Y Position');
        hold off;

        figure;
        scatter(sensorPositions(:, 1), sensorPositions(:, 2), 50, 'filled');
        hold on;
        scatter(clusterHeads(:, 1), clusterHeads(:, 2), 100, 'r', 'filled');
        title(sprintf('Cluster Heads and Assignments after Round %d', r));
        xlabel('X Position');
        ylabel('Y Position');
        legend('Sensors', 'Cluster Heads');
        hold off;
    end
end

% Final Results
fprintf('Total Energy Consumption: %.2f\n', totalEnergy(rounds));
fprintf('Average Sensor Energy: %.2f\n', abs(avgSensorEnergy(rounds)));

