function [Acc]=f_SVM(tr,te)

Trd = tr(:,1:end-1); %Training data 
Trl = tr(:,end); %Training Label
Ted = te(:,1:end-1); %testing data
Tel = te(:,end); %Testing label

%% SVM Classification
classes=unique(Trl);
ms=length(classes);
SVMModels=cell(ms,1);
for j = 1:numel(classes)
    indx = zeros(size(Trl,1),1);
    [r,c] = find (Trl ==classes(j));% Create binary classes for each classifier
    indx(r,1) = 1;
    SVMModels{j}=fitcsvm(Trd,indx,'KernelFunction','polynomial');
end

N=size(Ted,1);
Scores=zeros(N,numel(classes));
for j=1:numel(classes)
    [~,score]=predict(SVMModels{j},Ted);
    Scores(:,j)=score(:,2); % Second column contains positive-class scores
end

[~,prdt]=max(Scores,[],2);

%% Find Accuracy
Acc=mean(Tel==prdt)*100;
end
