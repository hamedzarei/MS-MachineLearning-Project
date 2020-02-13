function labels = rbf( train, test )

number_of_centers = 2;
max_diff = max(max(train))-min(min(train));
train_size = size(train);
test_size = size(test);
centers = max_diff*rand(number_of_centers, train_size(2)-1);
d_max = sqrt(sum((centers(2, :) - centers(1, :)).^2));
spread = d_max/sqrt(number_of_centers);

train_x = train(:, 1:train_size(2)-1);
train_y = train(:, train_size(2));

test_x = test(:, 1:test_size(2)-1);

fi = zeros(train_size(1), number_of_centers);

for i=1:train_size(1)
    fi(i, 1) = exp((sqrt(sum((train_x(i, :) - centers(1, :)).^2)))/-spread);
    fi(i, 2) = exp((sqrt(sum((train_x(i, :) - centers(2, :)).^2)))/-spread); 
end

w = pinv(fi)*train_y;

fi_test = zeros(test_size(1), number_of_centers);
for j=1:test_size(1)
    fi_test(j, 1) = exp((sqrt(sum((test_x(j, :) - centers(1, :)).^2)))/-spread);
    fi_test(j, 2) = exp((sqrt(sum((test_x(j, :) - centers(2, :)).^2)))/-spread); 
end

labels = fi_test*w;
end

