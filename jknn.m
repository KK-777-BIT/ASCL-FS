function pred = jknn(xTrain, yTrain, xQuery, k)
%JKNN Euclidean k-NN with deterministic majority voting.
% xTrain: nTrain-by-d features; yTrain: nTrain class labels.
% xQuery: nQuery-by-d features; pred: nQuery-by-1 predicted labels.
% k defaults to 1 and is capped at the number of training samples.
% This function returns labels, not cross-validation accuracy.
% If several classes receive the same number of votes, the class of the
% nearest neighbor among the tied classes is returned.

if isempty(xTrain) || size(xTrain, 2) == 0
    error('asclfs:EmptyFeatureSet', 'At least one feature is required.');
end
if nargin < 4 || isempty(k)
    k = 1;
end
if ~isscalar(k) || ~isfinite(k) || k < 1 || k ~= round(k)
    error('asclfs:InvalidK', 'k must be a positive integer.');
end
k = min(k, size(xTrain, 1));
yTrain = yTrain(:);

queryNorm = sum(xQuery .^ 2, 2);
trainNorm = sum(xTrain .^ 2, 2)';
distance2 = queryNorm + trainNorm - 2 * (xQuery * xTrain');
distance2 = max(distance2, 0);

if k == 1
    [~, nearest] = min(distance2, [], 2);
    pred = yTrain(nearest);
    return;
end

[~, neighborIndex] = mink(distance2, k, 2);
% Preserve one row per query, including the single-query case.
neighborLabel = reshape(yTrain(neighborIndex), size(neighborIndex));
classes = unique(yTrain, 'stable');
pred = repmat(classes(1), size(xQuery, 1), 1);

for queryIndex = 1:size(xQuery, 1)
    votes = zeros(numel(classes), 1);
    for classIndex = 1:numel(classes)
        votes(classIndex) = nnz( ...
            neighborLabel(queryIndex, :) == classes(classIndex));
    end
    tiedClasses = classes(votes == max(votes));
    if numel(tiedClasses) == 1
        pred(queryIndex) = tiedClasses;
    else
        % neighborLabel is ordered by increasing distance.
        for neighbor = 1:k
            currentLabel = neighborLabel(queryIndex, neighbor);
            if any(currentLabel == tiedClasses)
                pred(queryIndex) = currentLabel;
                break;
            end
        end
    end
end
end
