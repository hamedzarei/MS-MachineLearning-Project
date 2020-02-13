tic
clear all;
load Heart.dat;
feature_size = 13;
x = Heart(:, 1:feature_size);
y = Heart(:, feature_size+1);
s = size(Heart);
coeff = pca(x, 'NumComponents',5);
new_x = x * coeff;

k = 3;


bucket_size = floor(s(1)/5);
accuracy = zeros(5, 1);

tp = zeros(5,1);
tn = zeros(5,1);
fp = zeros(5,1);
fn = zeros(5,1);

for part=1:5
    all_index = 1:s(1);
    test_index = (part-1)*bucket_size+1:part*bucket_size;
    train_index = all_index(~ismember(all_index, test_index));
    Train = Heart(train_index, :);
    Train_x = Train(:, 1:feature_size);
    Train_y = Train(:, feature_size+1);
    Test = Heart(test_index, :);
    Test_x = Test(:, 1:feature_size);
    Test_y = Test(:, feature_size+1);
    
    mdl = fitcknn(Train_x,Train_y,'NumNeighbors',k);
    [label_1,score_1,cost_1] = predict(mdl,Test_x);
    
    test_size = size(Test_y);
    correct_1 = zeros(test_size);
    for c = 1:test_size(1)
        if Test_y(c) == label_1(c)
            correct_1(c) = correct_1(c) + 1;
            if label_1(c) == 0
                tp(part) = tp(part) + 1;
            else
                tn(part) = tn(part) + 1; 
            end
        else
            if label_1(c) == 0
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
time = toc;