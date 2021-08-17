function clusterIdentity = dbscan_jmc(vectors, epsilon, minPtsValues, nvp)
% Takes in a value for epsilon and a range of values for minPts, performing
% DBSCAN over that range of minPts values with distance metric as
% specified. Returns clusterIdentity, an m x p matrix, where m is the
% number of observations, and p is the number of minPts values tested; each
% column of clusterIdentity is a clustering corresponding to one value for
% minPts. This is not a grid search, or by itself a rigorous way to search
% across values for even minPts alone; it is mainly a visualization tool
% over a small set of minPts.
%
% PARAMETERS
% ----------
% vectors      -- Data to be analyzed. m x n matrix where m is the
%                 number of observations and n the number of variables.
% epsilon      -- Parameter for DBSCAN controlling radius of a
%                 neighborhood.
% minPtsValues -- Range of minPts values to test with DBSCAN clustering.
% Name-Value Pairs (nvp)
%   'distanceMetric' -- Distance metric to be used in DBSCAN. Default
%                       is 'cosine'.
%   'subplotDims'    -- Dimensions for the grid of subplots. 
%   'plotDims'       -- Dimensions of clustered data to plot in each
%                       subplot. Must be 2 or 3 (default).
%
% RETURNS
% -------
% clusterIdentity   -- m x p matrix. Columns correspond to a given value
%                      for minPts, and entries of each column contain the
%                      cluster assignment for each observation within that
%                      column (i.e. for that value for minPts).
% figure (subplots) -- Grid of subplots containing for each value of minPts
%                      one plot of the cluster assignments in the first two
%                      dimensions of the data.
%                    
% Author: Jonathan Chien 12/27/20

arguments
    vectors
    epsilon (1,1) {mustBeNumeric}
    minPtsValues {mustBeInteger}
    nvp.distanceMetric string = 'cosine'
    nvp.subplotDims {mustBeInteger}
    nvp.plotDims {mustBeMember(nvp.plotDims, [2 3])} = 3
end

if nvp.plotDims == 3 && size(vectors,2) == 2
    warning('3D plot requested, but data is 2D. Setting plot dimensionality to 2.')
    nvp.plotDims = 2;
elseif size(vectors,2) == 1
    warning('Dimensionality of data is 1. Will not produce plot.')
end

% Get number of observations and number of values for minPts to test.
nObs = size(vectors,1);
nMinPts = length(minPtsValues);

% Preallocate.
clusterIdentity = NaN(nObs, nMinPts);

for iMinPts = 1:nMinPts
    
    % Perform DBSCAN clustering.
    clusterIdentity(:,iMinPts) = dbscan(vectors, epsilon, minPtsValues(iMinPts),...
                                        'Distance', nvp.distanceMetric);
    
    % Plot if data is at least 2D.
    if size(vectors, 2) > 1
        
        clusterIndices = unique(clusterIdentity(:,iMinPts))'; % Indexing in loop below won't work if column vector; thus transpose to row vector.
        
        subplot(nvp.subplotDims(1), nvp.subplotDims(2), iMinPts)
        for clusterIdx = clusterIndices
            
            switch nvp.plotDims
                case 2
                    hold on
                    plot(vectors(clusterIdentity(:,iMinPts)==clusterIdx, 1),...
                         vectors(clusterIdentity(:,iMinPts)==clusterIdx, 2),...
                         'o')
                    grid on
                case 3
                    plot3(vectors(clusterIdentity(:,iMinPts)==clusterIdx, 1),...
                          vectors(clusterIdentity(:,iMinPts)==clusterIdx, 2),...
                          vectors(clusterIdentity(:,iMinPts)==clusterIdx, 3),...
                         'o')
                    hold on % must come after axes creation or will default to 2D
                    grid on
                    set(gca,'xminorgrid','on','yminorgrid','on')
            end
            
        end
        
        % Plot tingz.
        sgtitle(sprintf('DBSCAN Cluster Assignments for %s = %s (%s distance)',...
                        char(949), num2str(epsilon), nvp.distanceMetric)) % Using %d for epsilon directly seems to give less tidy results for nonintegers. 
        title(sprintf('minPts = %d', minPtsValues(iMinPts)))
        xlabel('Axis 1') 
        ylabel('Axis 2')
        if nvp.plotDims == 3
            zlabel('PA 3')
        end    
    end  
end

end
    