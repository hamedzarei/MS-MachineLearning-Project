tic
clear all;
load Glass.dat;
x = Glass(:, 1:10);
y = Glass(:, 11);

coeff = pca(x, 'NumComponents',5);
new_x = x * coeff;

k = 5;

s = size(Glass);
bucket_size = floor(s(1)/5);
accuracy = zeros(5,1);
for part=1:5
  all_index = 1:s(1);  
test_index = (part-1)*bucket_size+1:part*bucket_size;
train_index = all_index(~ismember(all_index, test_index));
Train = Glass(train_index, :);
Train_x = Train(:, 1:10);
Train_y = Train(:, 11);
Test = Glass(test_index, :);
Test_x = Test(:, 1:10);
Test_y = Test(:, 11);

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