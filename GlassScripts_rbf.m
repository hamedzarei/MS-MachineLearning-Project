tic
clear all;
load Glass.dat;
feature_size = 10;
x = Glass(:, 1:feature_size);
y = Glass(:, feature_size+1);
s = size(Glass);
new_feature_size = 5;
coeff = pca(x, 'NumComponents', new_feature_size);
new_x = x * coeff;

clear Glass;

Glass(:, 1:new_feature_size) = new_x;
Glass(:, new_feature_size+1) = y;
feature_size = new_feature_size;

bucket_size = floor(s(1)/5);
accuracy = zeros(5, 1);

goal = 0.001;
spread = 1;
neuron = 5;
epochs = 500;

for part=1:5
    all_index = 1:s(1);
    test_index = (part-1)*bucket_size+1:part*bucket_size;
    train_index = all_index(~ismember(all_index, test_index));
    Train = Glass(train_index, :);
    Train_x = Train(:, 1:feature_size);
    Train_y = Train(:, feature_size+1);
    Test = Glass(test_index, :);
    Test_x = Test(:, 1:feature_size);
    Test_y = Test(:, feature_size+1);
    
    label_1 = rbf(Train, Test);
    
    test_size = size(Test_y);
    correct_1 = zeros(test_size);
    for c = 1:test_size(1)
        if Test_y(c) == round(label_1(c))
            correct_1(c) = correct_1(c) + 1;
        end
    end
    accuracy(part) = sum(correct_1)/test_size(1);
end

mean = mean(accuracy);
std = std(accuracy);
max = max(accuracy);
min = min(accuracy);

time = toc;