function [run_acc, run_wAcc, run_conf] =  SeizureLSTMclassify(trainSeizureFile, trainNormalFile, runSeizureFile, runNormalFile)

if nargin()==0
    trainSeizureFile = 'SeizureSeconds.mat';
    trainNormalFile = 'NormalSeconds.mat';
    runSeizureFile = 'SeizureSeconds.mat';
    runNormalFile = 'NormalSeconds.mat';
end
%% Loading Training Data
trainSeizures = load(trainSeizureFile);
trainSeizures = trainSeizures.seizureSeconds;
trainNormal = load(trainNormalFile);
trainNormal = trainNormal.normalSeconds;

runSeizures = load(runSeizureFile);
runSeizures = runSeizures.seizureSeconds;
runNormal = load(runNormalFile);
runNormal = runNormal.normalSeconds;

[data,labels] = createData(trainSeizures, trainNormal);
[runData, runLabels] = createData(runSeizures, runNormal);
runLabels = runLabels';%need to transpose for getMetrics()
validate = 0;

if validate == 1
    
    VAL_FRAC=0.3;
    valInds=[];
    trainInds=[];

    for i=1:1:(max(labels)-min(labels)+1)
        inds=find(labels==(i-1));%finds indices of label
        inds=inds(randperm(length(inds)));%randomizes the indices
        valInds=[valInds;inds(1:round(VAL_FRAC*length(inds)))];%gets the validation indices
        trainInds=[trainInds;inds(round(VAL_FRAC*length(inds))+1:end)];%gets the training indices
    end

    validation_labels=labels(valInds)';
    training_labels=labels(trainInds)';

    xTrain=data(trainInds,:);
    xVal=data(valInds,:);
    
    %randomize training and validation data
    perm = randperm(length(xTrain));
    xTrain = xTrain(perm,:);
    training_labels = training_labels(perm);

    perm = randperm(length(xVal));
    xVal = xVal(perm,:);
    validation_labels = validation_labels(perm);
else
    xTrain=data(:,:);
    training_labels=labels';
    
end

xTrain = squeeze(num2cell(xTrain,2));
yTrain = training_labels;
yTrain = categorical(yTrain)';

if validate == 1
    xVal = squeeze(num2cell(xVal,2));
end


%% LSTM
%Defining NN params
inputSize = size(xTrain{1},1);
numHiddenUnits = 100;
numClasses = max(labels)+1;
classNames = categories(yTrain);

classWeights = get_weights(training_labels);

maxEpochs = 5;

miniBatchSize = 1024;

layers = [ ...
        sequenceInputLayer(inputSize)
        lstmLayer(numHiddenUnits,'OutputMode','last')
        %lstmLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(.15)
        fullyConnectedLayer(numClasses)
        softmaxLayer    
        classificationLayer   
        ];
    
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'GradientDecayFactor',0.9, ...
    'SquaredGradientDecayFactor',0.999, ...
    'Epsilon',1e-8, ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',2,...
    'LearnRateDropFactor',.99,...
    'Verbose',1, ...
    'Plots','training-progress');

%training
g = gpuDevice;
g.wait();
net = trainNetwork(xTrain,yTrain,layers,options);

train_probs = predict(net,xTrain, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
if validate == 1
    val_probs = predict(net,xVal, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

    [val_acc,val_wAcc,val_conf] = getMetrics(val_probs,validation_labels)
end

xRun = squeeze(num2cell(runData,2));
run_probs = predict(net,xRun, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

% figure(1)
% %plotconfusion(categorical(training_labels)',categorical(train_probs))
% figure(2)
% plotconfusion(categorical(validation_labels)',categorical(val_probs))
[train_acc,train_wAcc,train_conf] = getMetrics(train_probs,training_labels)
[run_acc,run_wAcc,run_conf] = getMetrics(run_probs,runLabels)
end
%% Function Definitions
function [data, labels] = createData(seizureSeconds, normalSeconds)
SeizureLabels = ones(length(seizureSeconds),1);
NormalLabels = zeros(length(normalSeconds),1);
data = [seizureSeconds;normalSeconds];
labels = [SeizureLabels;NormalLabels];
perm = randperm(length(data));
data = data(perm,:);
labels = labels(perm);
end
function [weights] = get_weights(labels)
NUM_CLASSES=max(labels)-min(labels)+1;

weights=zeros(NUM_CLASSES,1);
count=1;
counts=zeros(NUM_CLASSES,1);
for i=min(labels):1:max(labels)
    weights(count)=length(labels)/sum(labels==i);
    counts(count)=sum(labels==i);
    count=count+1;
end
weights=weights/(sum(weights.*counts)/length(labels));
end

function [acc,weightedAcc,conf]=getMetrics(probs,labels)
NUM_CLASSES=size(probs,2);
labels = labels';
% if you want to remove null ...
%probs(:,end)=0;
%probs=probs./sum(probs,2);

[~,oneHot]=max(probs,[],2);
oneHot=oneHot-1;

acc=sum(oneHot==labels)/length(labels);

weights=0*labels;
for i=1:1:NUM_CLASSES
    inds=find(labels==i-1);
    weights(inds)=1/length(inds);
end
weightedAcc=sum(weights.*(oneHot==labels))/sum(weights);

conf=zeros(NUM_CLASSES);
for i=1:1:NUM_CLASSES
    for j=1:1:NUM_CLASSES
        conf(i,j)=sum((labels==i-1).*(oneHot==j-1));
    end
end
end