function epsilon = estimateEpsilon(vectors, kValues, nvpair)
% Estimates epsilon for DBSCAN by constructing a kNN distance plot and
% searching for the epsilon value corresponding to the knee. This is
% accomplished here by connecting the endpoints of the graph and projecting
% all points onto this line. The distance corresponding to the point with
% the greatest projection error is the estimate for epsilon. This
% capability is offered via the estimateEpsilon object function associated
% with the clusterDBSCAN object through the Phased Array System Toolbox
% (exact method for epsilon esimation unclear, but documentation suggests
% it also relies on maximizing distance from points to a line connecting
% the endpoints). However, it seems that Euclidean distance is the only
% distance metric available through that object function. This function
% here aims to allow other distance metrics in the estimation of epsilon,
% in anticipation of performing DBSCAN with these other distance metrics.
%
% PARAMETERS
% ----------
% vectors -- Data for which epsilon is to be estimated. m x n matrix
%          where rows are observations and columns are variables.
% kValues -- Vector of scalar values for k to be tested.
% Name-Value Pairs (nvpair)
%   ;distanceMetric' -- Distance metric for determing kNN. Options
%                       correspond to the 'Distance' Name-Value Pair
%                       Arguments for knnsearch.m. Default is
%                       'euclidean'.
%   'angleOption'    -- Has effect only if 'cosine' was specified as the
%                       distance metric. Default is 'versine', which
%                       causes epsilon to be estimated from cosine
%                       distance to nearest neighbors. 'angularDistance'
%                       will cause cosine distance to be converted to
%                       angular distance. 'cosineSimilarity' will cause
%                       cosine distance to be converted to cosine
%                       similarity.
%   'normalize'      -- Default is 0. If set to 1, vectors will be normed
%                       to unit length. 
%
% RETURNS
% -------
% epsilon -- Vector of estimates for epsilon. One estimate for each value
%            of k in kValues.
% plot    -- Plot of distance vs point index for each value of k.
%
% Author: Jonathan Chien 12/24/20.

arguments
    vectors 
    kValues {mustBeInteger}
    nvpair.distanceMetric string = 'euclidean'
    nvpair.angleOption string = 'versine'
    nvpair.normalize = false
end
    
% Get number of observations (m). 
nObs = size(vectors, 1);

% Get number of values for k to be tested.
n_kValues = length(kValues);

% Option to norm vectors to unit length. 
if nvpair.normalize 
    vectors = vectors./vecnorm(vectors,2,2);
end

% % Only applicable for angle based distance: 
% % Calculate cosine similarities and remove diagonal (similarity of a
% % given observation with itself), then sort. Assumes vectors are unit
% % length.
% unsortedNeighbors = vectors*vectors' - 2*speye(nObs); % Multiply by 2 as dot product between some observations may be negative.
% sortedNeighbors = sort(unsortedNeighbors, 2, 'descend'); % Large cosine similarity indicates small angular distance.
% % sortedNeighbors = acos(sortedNeighbors); % If conversion from cosine similarity to angular distance desired.

% Find distance to all nObs-1 neighbors for each observation according to
% specified distance metric and sort neighbors by distance.
[~,sortedNeighbors] = knnsearch(vectors, vectors, 'K', nObs, 'Distance', nvpair.distanceMetric);
sortedNeighbors = sortedNeighbors(:,2:end); % Each observation's "nearest neighbor" (i.e. k = 1) is itself.

% If nvpair.distanceMetric is 'cosine' and 'angularDistance' is requested
% for 'angleOption', convert cosine distance (versine) to angular distance;
% if 'cosineSimilarity' is requested for 'angleOption', convert cosine
% distance to cosine similarity. 
if strcmp(nvpair.distanceMetric, 'cosine') && strcmp(nvpair.angleOption, 'angularDistance')
    sortedNeighbors = acos(1 - sortedNeighbors);
    distanceMetricLabel = 'angular'; % A bit janky, but for purposes of labeling y axis in plot.
elseif strcmp(nvpair.distanceMetric, 'cosine') && strcmp(nvpair.angleOption, 'cosineSimilarity')
    sortedNeighbors = 1 - sortedNeighbors;
    distanceMetricLabel = 'cosine similarity'; 
else
    distanceMetricLabel = nvpair.distanceMetric; 
end

% Preallocate.
epsilon = NaN(n_kValues,1);
graphLabels = cell(n_kValues,1);
graphLabelX = NaN(n_kValues,1);
graphLabelY = NaN(n_kValues,1);

% Estimate epsilon for each k and plot.
kIdx = 0;
for k = kValues
    
    kIdx = kIdx + 1;
    
    % Retain only k nearest neighbors.
    kNN = sortedNeighbors(:, 1:k);

    % Place all columns of kNN matrix into a single column vector (will
    % contain nObs*k points) and sort.
    sortedDists = sort(kNN(:), 'ascend');
    pointIdx = (1:nObs*k)';

    % Center 1st point at origin and calculate projection errors. Distance
    % corresponding to point with maximum projection error will be estimate
    % for epsilon for current value of k.
    b = cat(2, pointIdx - 1, sortedDists - sortedDists(1));
    a = [pointIdx(end); b(end,end)];
    x_hat = b*a/(a'*a);
    p = x_hat*a';
    e = b - p;
    eNorms = vecnorm(e,2,2);
    [~,maxIdx] = max(eNorms);
    epsilon(kIdx) = sortedDists(maxIdx); % maxIdx corresponds to eNorms, which is unaffected by subtraction of 1; thus no need to add 1 back.
    
    % Plot epsilon vs point index.
    hold on
    plot(pointIdx, sortedDists, '-o', ...
         'Color', [0 0.4470 0.7410], 'MarkerSize', 1.5, 'MarkerFaceColor', [0 0.4470 0.7410]);
    h = plot(maxIdx, epsilon(kIdx), 'o', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 1.5);
    grid on
    title('kNN Distance Plot')
    xlabel('Point Index')
    if strcmp(distanceMetricLabel, 'cosine similarity') % Handle different cases for y label. Janky.
        ylabel('epsilon (cosine similarity)')
    else
        ylabel(sprintf('epsilon (%s distance)', distanceMetricLabel))
    end
    graphLabels{kIdx} = [num2str(k) '-NN'];
    graphLabelX(kIdx) = pointIdx(end);
    graphLabelY(kIdx) = sortedDists(end) + 0.05*range(sortedDists);
    legend(h, 'Estimated epsilon for k', 'Location', 'northwest')
    
end

% Label each plot corresponding to each value of k.
text(graphLabelX, graphLabelY, graphLabels)

end
