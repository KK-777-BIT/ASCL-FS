function [cost, classError, featureRatio, foldError] = jFitnessFunction(feat, label, mask, opts)
%JFITNESSFUNCTION Accuracy-aligned single-objective fitness.
% feat: n-by-d features; label: n labels; mask: d feature-selection flags.
% opts.cvTrain/cvVal are cells of logical fold masks; opts.k sets kNN size.
% Returns scalar cost, pooled classError, featureRatio, and a fold-error column.
% The classification term is the pooled inner-CV misclassification rate,
% matching the ordinary accuracy reported on the outer test folds.

mask = logical(mask(:)');
numSelected = nnz(mask);
featureRatio = numSelected / numel(mask);
if numSelected == 0
    classError = 1;
    cost = 1;
    foldError = ones(numel(opts.cvTrain), 1);
    return;
end

numFolds = numel(opts.cvTrain);
foldError = zeros(numFolds, 1);
foldSize = zeros(numFolds, 1);
for f = 1:numFolds
    tr = opts.cvTrain{f};
    va = opts.cvVal{f};
    pred = jknn( ...
        feat(tr, mask), label(tr), feat(va, mask), opts.k);
    yValid = label(va);
    foldSize(f) = numel(yValid);
    foldError(f) = mean(pred ~= yValid);
end
% Fixed Case A objective weights, applied to the original feature dimension.
alpha = 0.90;
beta = 0.10;
classError = sum(foldSize .* foldError) / sum(foldSize);
cost = alpha * classError + beta * featureRatio;
end
