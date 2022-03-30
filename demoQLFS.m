clc
clear all
close all
load vehicle
D=zscore(data); %normalizing the data
new_data=[D label];
z=size(D,2);
n=18;
[M,N]=size(new_data);
indices=crossvalind('Kfold',new_data(1:M,N),5); %K-Fold cross-validation (K=5)
for iter=1:5
    test=(indices==iter); %Creating logical indices for testing data
    train =~test; %Creating logical indices for training data
    TrainingData=new_data(train,:); %Fetching the actual data for training
    TestingData=new_data(test,:); %Fetching the actual data for testing
    X=[(TrainingData(:,1:end-1))]; % Preparing the Features
    Y=TrainingData(:,end); %Prep. the Class labels
    [r]=QLFS(TrainingData);r=r'; %% Applying Q-Learning for the Feature Ranking
    i1=0;
    for i=1:1:n
        i1=i1+1;
        tzno=i;
        TrainingData1=[TrainingData(:,r(1:tzno)),TrainingData(:,end)];
        TestingData1=[TestingData(:,r(1:tzno)),TestingData(:,end)];
        [accuracy(i1,iter),~]= f_SVM(TrainingData1,TestingData1);
    end
end
plot(accuracy,'-*');
legend('QLFS');
hold on;
ylabel('AC')
xlabel('number of features')  ;
set(gca, 'Fontname', 'Times newman', 'Fontsize', 18);