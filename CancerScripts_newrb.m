tic
clear all;
load WDBC.dat;
feature_size = 30;
x = WDBC(:, 1:feature_size);
y = WDBC(:, feature_size+1);
s = size(WDBC);
new_feature_size = 5;
coeff = pca(x, 'NumComponents', new_feature_size);
new_x = x * coeff;

clear WDBC;

WDBC(:, 1:new_feature_size) = new_x;
WDBC(:, new_feature_size+1) = y;
feature_size = new_feature_size;

bucket_size = floor(s(1)/5);
accuracy = zeros(5, 1);

tp = zeros(5,1);
tn = zeros(5,1);
fp = zeros(5,1);
fn = zeros(5,1);

goal = 0.001;
spread = 1;
neuron = 5;
epochs = 500;

for part=1:5
    all_index = 1:s(1);
    test_index = (part-1)*bucket_size+1:part*bucket_size;
    train_index = all_index(~ismember(all_index, test_index));
    Train = WDBC(train_index, :);
    Train_x = Train(:, 1:feature_size);
    Train_y = Train(:, feature_size+1);
    Test = WDBC(test_index, :);
    Test_x = Test(:, 1:feature_size);
    Test_y = Test(:, feature_size+1);
    
    
    net_rb = newrb(Train_x', Train_y', goal, spread, neuron);
    net_rb.trainFcn = 'trainlm';
    net_rb.trainParam.epochs = epochs;
    
    trainedNet = train(net_rb, Train_x', Train_y');
    
    label_1 = trainedNet(Test_x')';
    
    test_size = size(Test_y);
    correct_1 = zeros(test_size);
    for c = 1:test_size(1)
        if Test_y(c) == round(label_1(c))
            correct_1(c) = correct_1(c) + 1;
            if round(label_1(c)) == 0
                tp(part) = tp(part) + 1;
            else
                tn(part) = tn(part) + 1;
            end
        else
            if round(label_1(c)) == 0
                fp(part) = fp(part) + 1;
            else
                fn(part) = fn(part) + 1;
            end
        end
    end
    accuracy(part) = sum(correct_1)/test_size(1);
end

mean = mean(accuracy);
std = std(accuracy);
max = max(accuracy);
min = min(accuracy);

sensitivity = tp(1)/(tp(1) + fn(1));
specificity = tn(1)/(tn(1) + fp(1));

mcc = (tp(1)*tn(1) - fp(1)*fn(1))/sqrt((tp(1)+fp(1))*(tp(1)+fn(1))*(tn(1)+fp(1))*(tn(1)+fn(1)));

f1 = (2*tp(1))/(2*tp(1) + fp(1) + fn(1));

tim = toc;