% ASCL-FS: edit the settings below and run this script.
clear; clc;

% Parameters.
opts = parameters;
runNum = opts.runNum;

% Datasets.
datasetNames = ["11_Tumors.mat", "9_Tumors.mat", "Adenocarcinoma.mat", ...
    "ALLAML.mat", "arcene.mat", "Brain_Tumor1.mat", "BreastCancer1.mat", ...
    "BreastCancer2.mat", "CLL-SUB-111.mat", "Colon.mat", "GLI_85.mat", ...
    "Leukemia.mat", "Leukemia1.mat", "Leukemia2.mat", "Lung_Cancer.mat", ...
    "lung.mat", "Lymphoma.mat", "MLL_3.mat", "nci9.mat", "Prostate_GE.mat", ...
    "SMK-CAN-187.mat"];

% Paths.
projectDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(projectDir, 'DataSet');
resultDir = fullfile(projectDir, 'results');
if ~isfolder(resultDir)
    mkdir(resultDir);
end

% Load data, run the experiment, and save each completed dataset.
summary = table;
for datasetIndex = 1:numel(datasetNames)
    datasetName = datasetNames(datasetIndex);
    [feat, label] = load_dataset(fullfile(dataDir, datasetName));
    [datasetResult, row] = run_dataset(feat, label, datasetName, opts);
    save_dataset(datasetResult, resultDir);
    if isempty(summary)
        summary = row;
    else
        summary = [summary; row]; %#ok<AGROW>
    end
    writetable(summary, fullfile(resultDir, 'summary.csv'));
    save(fullfile(resultDir, 'summary.mat'), 'summary', 'opts', 'datasetNames', 'runNum');
end
fprintf('\nResults saved to: %s\n', resultDir);


function [datasetResult, row] = run_dataset(feat, label, datasetName, opts)
% Keep partition generation and optimizer calls in their original order.
runNum = opts.runNum;
numClasses = numel(unique(label));
fprintf('\n%s | N=%d | T=%d | runs=%d | k=%d | thresholds=%.2f/%.2f\n', ...
    datasetName, opts.N, opts.T, runNum, opts.k, ...
    opts.thresholdAccuracyStart, opts.thresholdSparseStart);

testAcc = zeros(runNum, 1);
trainCV = zeros(runNum, 1);
gbest = zeros(runNum, 1);
featureCount = zeros(runNum, 1);
elapsedSeconds = zeros(runNum, 1);
runSeeds = zeros(runNum, 1);
foldTestAccuracy = cell(runNum, 1);
foldTrainCV = cell(runNum, 1);
foldGbest = cell(runNum, 1);
foldFeatures = cell(runNum, 1);
selectedFeatures = cell(runNum, opts.outerK);
convergenceCurves = cell(runNum, opts.outerK);
outerTestIndices = cell(runNum, opts.outerK);
innerTrainIndices = cell(runNum, opts.outerK);
innerValIndices = cell(runNum, opts.outerK);
oofPredictions = zeros(size(feat, 1), runNum);

for runIndex = 1:runNum
    % Keep the original RNG order: generate ALL folds before optimization.
    runSeeds(runIndex) = opts.seed + runIndex;
    rng(runSeeds(runIndex), 'twister');
    outerCV = cvpartition(label, 'KFold', opts.outerK);
    for outerFold = 1:opts.outerK
        trainIndex = training(outerCV, outerFold);
        innerCV = cvpartition(label(trainIndex), 'KFold', opts.innerK);
        cvTrain = cell(opts.innerK, 1);
        cvVal = cell(opts.innerK, 1);
        for innerFold = 1:opts.innerK
            cvTrain{innerFold} = training(innerCV, innerFold);
            cvVal{innerFold} = test(innerCV, innerFold);
        end
        innerTrainIndices{runIndex, outerFold} = cvTrain;
        innerValIndices{runIndex, outerFold} = cvVal;
    end

    oofPrediction = nan(size(label));
    assigned = false(size(label));
    foldAcc = zeros(opts.outerK, 1);
    foldCV = zeros(opts.outerK, 1);
    foldFitness = zeros(opts.outerK, 1);
    foldCount = zeros(opts.outerK, 1);
    runTime = 0;
    for outerFold = 1:opts.outerK
        fprintf('  Run %d/%d | fold %d/%d\n', ...
            runIndex, runNum, outerFold, opts.outerK);
        trainIndex = training(outerCV, outerFold);
        testIndex = test(outerCV, outerFold);
        xTrain = feat(trainIndex, :);
        yTrain = label(trainIndex);
        xTest = feat(testIndex, :);
        yTest = label(testIndex);
        algorithmOptions = opts;
        algorithmOptions.cvTrain = innerTrainIndices{runIndex, outerFold};
        algorithmOptions.cvVal = innerValIndices{runIndex, outerFold};

        timer = tic;
        model = gASCLFS(xTrain, yTrain, algorithmOptions);
        prediction = jknn(xTrain(:, model.sf), yTrain, xTest(:, model.sf), opts.k);
        foldCV(outerFold) = subset_cv_accuracy( ...
            xTrain, yTrain, model.sf, algorithmOptions);
        runTime = runTime + toc(timer);

        oofPrediction(testIndex) = prediction;
        assigned(testIndex) = true;
        foldAcc(outerFold) = mean(prediction == yTest);
        foldFitness(outerFold) = model.gb;
        foldCount(outerFold) = model.nf;
        selectedFeatures{runIndex, outerFold} = model.sf;
        convergenceCurves{runIndex, outerFold} = model.c;
        outerTestIndices{runIndex, outerFold} = find(testIndex);
    end
    assert(all(assigned), 'Not all samples received an outer-fold prediction.');

    % Pool outer predictions; do not average unequal-sized test folds.
    testAcc(runIndex) = mean(oofPrediction == label);
    trainCV(runIndex) = mean(foldCV);
    gbest(runIndex) = mean(foldFitness);
    featureCount(runIndex) = mean(foldCount);
    elapsedSeconds(runIndex) = runTime;
    foldTestAccuracy{runIndex} = foldAcc;
    foldTrainCV{runIndex} = foldCV;
    foldGbest{runIndex} = foldFitness;
    foldFeatures{runIndex} = foldCount;
    oofPredictions(:, runIndex) = oofPrediction;
    fprintf('  TestAcc %.6f | TrainCV %.6f | Gbest %.6f | F %.2f | %.2fs\n', ...
        testAcc(runIndex), trainCV(runIndex), gbest(runIndex), ...
        featureCount(runIndex), runTime);
end

testClasses = repmat(compose('%d/%d', numClasses, numClasses), runNum, 1);
runTable = table((1:runNum)', testAcc, trainCV, gbest, featureCount, ...
    repmat(opts.outerK, runNum, 1), repmat(opts.innerK, runNum, 1), ...
    testClasses, elapsedSeconds, runSeeds, ...
    'VariableNames', {'Run', 'TestAcc', 'TrainCV', 'Gbest', 'F', ...
    'OuterK', 'InnerK', 'TestClasses', 'TimeSeconds', 'SplitSeed'});
datasetResult = struct;
datasetResult.dataset = datasetName;
datasetResult.algorithm = 'ASCL-FS';
datasetResult.protocol = sprintf( ...
    'Repeated nested CV: outer %d-fold, inner %d-fold', opts.outerK, opts.innerK);
datasetResult.options = opts;
datasetResult.runTable = runTable;
datasetResult.foldTestAccuracy = foldTestAccuracy;
datasetResult.foldTrainCV = foldTrainCV;
datasetResult.foldGbest = foldGbest;
datasetResult.foldFeatures = foldFeatures;
datasetResult.selectedFeatures = selectedFeatures;
datasetResult.convergenceCurves = convergenceCurves;
datasetResult.outerTestIndices = outerTestIndices;
datasetResult.innerTrainIndices = innerTrainIndices;
datasetResult.innerValIndices = innerValIndices;
datasetResult.runSeeds = runSeeds;
datasetResult.oofPredictions = oofPredictions;
datasetResult.seedScheme = 'rng(baseSeed + runIndex, twister) once per run';

row = table(datasetName, mean(testAcc), std(testAcc), mean(trainCV), ...
    std(trainCV), mean(gbest), std(gbest), mean(featureCount), ...
    std(featureCount), mean(elapsedSeconds), ...
    'VariableNames', {'Dataset', 'TestAccMean', 'TestAccStd', ...
    'TrainCVMean', 'TrainCVStd', 'GbestMean', 'GbestStd', ...
    'FeatureMean', 'FeatureStd', 'TimeMeanSeconds'});
end


function save_dataset(datasetResult, resultDir)
baseName = erase(datasetResult.dataset, '.mat');
writetable(datasetResult.runTable, fullfile(resultDir, baseName + "_runs.csv"));
save(fullfile(resultDir, baseName + "_result.mat"), 'datasetResult');
end


function accuracy = subset_cv_accuracy(feat, label, selected, opts)
correct = 0;
total = 0;
for fold = 1:numel(opts.cvTrain)
    tr = opts.cvTrain{fold};
    va = opts.cvVal{fold};
    prediction = jknn(feat(tr, selected), label(tr), feat(va, selected), opts.k);
    correct = correct + nnz(prediction == label(va));
    total = total + nnz(va);
end
accuracy = correct / total;
end


function [feat, label] = load_dataset(dataFile)
if ~isfile(dataFile)
    error('asclfs:DatasetMissing', 'Dataset not found: %s', dataFile);
end
data = load(dataFile);
if ~isfield(data, 'feat') || ~isfield(data, 'label')
    error('asclfs:DatasetVariables', '%s must contain feat and label.', dataFile);
end
feat = double(data.feat);
[~, ~, label] = unique(data.label(:));
label = double(label);
if isempty(feat) || size(feat, 1) ~= numel(label)
    error('asclfs:DatasetSize', 'Invalid data dimensions in %s.', dataFile);
end
if any(~isfinite(feat), 'all') || any(ismissing(data.label(:)))
    error('asclfs:NonFiniteData', 'Invalid features or labels in %s.', dataFile);
end
end
