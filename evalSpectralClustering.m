function [meanMetric,optimalK] = evalSpectralClustering(vectors,kVals,nBootstraps,nvp)
% Uses either Adjusted Rand Index (ARI) or Adjusted Mutual Information
% (AMI) to estimate optimal value for k (number of clusters) by evaluating
% stability across clusterings of repeated jacknifed/bootstrapped datasets
% for each value of k.
%
% PARAMETERS
% ----------
% vectors     -- m x n matrix dataset where the m rows correspond to
%                observations the n columns to variables or features.
% kVals       -- 1D array of integer values of k (number of clusters) to be
%                tested. Default is 2:10.
% nBootstraps -- Number of bootstrap (or jacknife) iterations used to
%                perturb dataset.
% Name-Value Pairs (nvp)
%   'subsample' -- Fraction of original data to randomly sample to
%                  generate each jacknife/bootstrap dataset. Default is
%                  0.85.
%   'replace'   -- Logical true or false (default). Specify whether to
%                  subsample with replacement or without (usually
%                  'bootstrapping' refers to sampling with replacement, but
%                  default is without). 
%   'stability' -- Metric used to evaluate clustering stability. Specify
%                  'ARI' (default) for Adjusted Rand Index and 'AMI' for
%                  Adjusted Mutual Information.
%
% RETURNS
% -------
% meanMetric -- k x 1 array, where each element is the mean value of the
%               specified stability metric for that value of k.
% optimalK   -- Optimal estimated value for k, designated as the value of
%               k corresponding to the maximum mean metric value.
% 2D plot    -- kVals vs meanMetric, with optimalK circled.
%
% Author: Jonathan Chien 3/22/21 Version 1 Last edit: 3/23/21

arguments
    vectors
    kVals {mustBeInteger} = [2:10]
    nBootstraps {mustBeInteger} = 100
    nvp.subsample (1,1) = 0.85 
    nvp.replace = false
    nvp.stability string = 'ARI'  
end

% Get parameters, preallocate, initialize.
nObs = size(vectors, 1);
sampleSize = ceil(nObs*nvp.subsample);
clusterings = NaN(length(kVals), nBootstraps, sampleSize);
bootVecsIdx = NaN(length(kVals), nBootstraps, sampleSize);
kIdx = 0;
metric = NaN(length(kVals), nBootstraps, nBootstraps);
w = waitbar(0, '');

% Iterate over all values of k.
for k = kVals
    
    % Update waitbar. Note kIdx is also used elsewhere below.
    kIdx = kIdx + 1;
    waitbar(kIdx./length(kVals), w, sprintf('Testing k = %d...', k));               
    
    % Iterate over bootstraps.
    for b = 1:nBootstraps  
        
        % Sample data and cluster bootstrapped data.
        [bootVecs, bootVecsIdx(kIdx,b,:)]...
            = datasample(vectors, sampleSize, 1, 'Replace', nvp.replace);
        
        clusterings(kIdx,b,:) = spectralcluster(bootVecs, k);
                        
        % Evaluate clusterings as they are produced.
        if b > 1
            for bb = 1:b-1
                % Get index of points in intersection of two samples.
                [~,ib,ibb] = intersect(bootVecsIdx(kIdx,b,:),...
                                       bootVecsIdx(kIdx,bb,:));
                % Evaluate agreement of two clusterings.
                switch nvp.stability
                    case 'ARI'
                        metric(kIdx,b,bb) = rand_index(clusterings(kIdx,b,ib),...
                                                       clusterings(kIdx,bb,ibb),...
                                                       'adjusted');
                    case 'AMI'
                        metric(kIdx,b,bb) = AMI(clusterings(kIdx,b,ib),...
                                                clusterings(kIdx,bb,ibb));
                end
            end
        end
    end
end
close(w)

% Determine optimal k.
meanMetric = mean(metric, [2 3], 'omitnan');
[maxMetric, iOptimalK] = max(meanMetric);
optimalK = kVals(iOptimalK);

% Plot mean metric value for each k value.
figure
hold on
plot(kVals, meanMetric, '-o', ...
     'Color', [0 0.4470 0.7410], 'MarkerSize', 1.5,...
     'MarkerFaceColor', [0 0.4470 0.7410]);
h = plot(optimalK, maxMetric, 'o', 'Color', [0.8500, 0.3250, 0.0980],...
         'LineWidth', 1.5);
xlabel('k (number of clusters)')
ylabel(sprintf('Mean %s', nvp.stability))
title(sprintf('Mean %s across values for k', nvp.stability))
legend(h, 'Estimated optimal k', 'Location', 'northeast')

end
