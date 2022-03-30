function [ranked] = QLFS(train_new_data)
feature=size(train_new_data,2);
number=size(train_new_data,1);
state=zeros(feature,number);
state(1,1:number)=1;
train_new_data=train_new_data';
state(2:end,:)=train_new_data(1:end-1,:);
state(feature+1,:)=train_new_data(end,:);
c=max(state(end,:));
w=zeros(c,feature);
ranked=zeros(1,feature-1);
t=0;
maxiter=1;
c1=randperm(number);
state1=state;
for paixu=1:number
    state(:,paixu)=state1(:,c1(paixu));
end
alpha=30; % learning rate
while t~=maxiter
    w=zeros(c,feature);
    for k=1:10
        w0=w;
        for i1=1:number
            if rand>0.8                            %choose action
                for z=1:c
                    v(z)=dot(state(1:end-1,i1),w(z,:));
                end
                d=find(v(:)==max(v(:)));
                d=d(1);
            else
                d=randi([1,c]);
            end
            if  d==state(end,i1)           %reward
                R=1;
            else
                R=-1;
            end
            if i1<number
                for z=1:c
                    v_next(z)=dot(state(1:end-1,i1+1),w(z,:));
                end
                R=R+0.1*max(v_next);
            end
            tidu=(R-dot(w(d,:),state(1:end-1,i1)));
            w(d,:)=w(d,:)+(1/((alpha)*(i1^0.2)))*(tidu*state(1:end-1,i1)');
        end
        if norm(w-w0)<0.00001
            break
        end
    end
    %% Delete feature
    WW=abs(w(:,2:end));
    WW=WW.^2;
    WW=sum(WW);
    sum_ww=sum(WW);
    del=find(WW<=(sum_ww/(feature-1)));
    zero=find(WW==0);
    t=size(zero,2);
    if t~=size(del,2)
        for i=1:size(del,2)
            state(del(i)+1,:)=0;
        end
        [~,s]=sort(WW(del(1:end)));
        for i=1:size(del,2)-t
            ranked(i+t)=del(s(i+t));
        end
    else
        [~,s]=sort(WW);
        ranked(1+t:end)=s(1+t:feature-1);
    end
    maxiter=size(del,2);
end
ranked=fliplr(ranked);
end