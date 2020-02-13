tic
clear all;
load Heart.dat;
feature_size = 13;
x = Pima(:, 1:feature_size);
y = Pima(:, feature_size+1);
s = size(Pima);
coeff = pca(x, 'NumComponents',5);
new_x = x * coeff;

k = 5;


bucket_size = floor(s(1)/5);
accuracy = zeros(5, 1);
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

mdl = fitcknn(Train_x,Train_y,'NumNeighbors',k);
[label_1,score_1,cost_1] = predict(mdl,Test_x);

test_size = size(Test_y);
correct_1 = zeros(test_size);
for c = 1:test_size(1)
    if Test_y(c) == label_1(c)
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