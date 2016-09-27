function [data, A, L] = GenerateSyntheticData(tensor_dims, rank, method);
%
% This function is used to generate the synthetic tensor data by four
% different methods:
% 1. Generate latent matrics in which U(i,:) and U(i+1,:) are correlated.
% 2. Randomly generate latent matrices; similarity matrics are calculated 
% based on generated latent matrices.
% 3. Completely random latent matrices with diagonal similarity matrices.
% 4. Similar method as method 1 except that epsilon prime is randomized 
% in this method.
% 
%--------------------------------------------------------------------------
% 
% Author: Hancheng Ge, Date: 09/20/2016
% 
%--------------------------------------------------------------------------


if nargin < 3
    method = 1;
end

if nargin < 2
    rank = 10;
end

if nargin < 1
    tensor_dims = [100,100,100];
end

% order of tensor
tensor_order = length(tensor_dims);
% dimensions of core tensor
core_dims    = rank * ones(1, tensor_order);

switch method
    case 1 % ------- Method 1: formulated U + sim matrix A -------
        % generate latent matrices        
        U = cell(1, tensor_order);
        for i = 1: tensor_order
            epsilons = rand(rank,2);
            U{i} = repmat((1:1:tensor_dims(i))',1,2).*repmat(epsilons(:,1)',...
                tensor_dims(i),1) + repmat(epsilons(:,2)',tensor_dims(i),1);
            [U{i}, aa1] = qr(U{i}, 0);
        end
        % generate the core tensor
        index_temp = ones(1, rank);
        C = tendiag(index_temp, core_dims); 
        % generate low-rank tensor
        X = ttensor(tensor(C), U);
        data = double(X);
        % generate similarity matrix
        A = cell(1, tensor_order);
        for i = 1: tensor_order
            A{i} = diag(ones(tensor_dims(i)-1,1),-1) + diag(ones(tensor_dims(i)-1,1),1);
        end
        % generate laplacian matrix
        L = cell(1, tensor_order);
        for i = 1: tensor_order
            L{i} = diag(sum(A{i},2))-A{i};
        end
    case 2 % ------- Method 2: random U + sim matrix A -------
        % generate latent, similar and laplacian matrices
        U = cell(1, tensor_order);
        A = cell(1, tensor_order);
        L = cell(1, tensor_order);
        for i = 1: tensor_order
            U{i} = rand(tensor_dims(i),rank);
            [U{i}, aa1] = qr(U{i}, 0);
            A{i} = ones(tensor_dims(i),tensor_dims(i)) - pdist2(U{i},U{i}) - eye(tensor_dims(i));
            L{i} = diag(sum(A{i},2))-A{i};
        end
        % generate the core tensor
        index_temp = ones(1, core_dims);
        C = tendiag(index_temp, core_dims);
        % generate low-rank tensor
        X = ttensor(tensor(C), U);
        data = double(X);
    case 3 % ------- Method 3: completely random data -------
        data = randn(tensor_dims);
        L = cell(1, tensor_order);
        for i = 1: tensor_order
            L{i} = eye(tensor_dims(i));
        end
    case 4 % ------- Method 4: random epsilon prime -------
        % generate latent, similar and laplacian matrices
        U = cell(1, tensor_order);
        A = cell(1, tensor_order);
        L = cell(1, tensor_order);
        for i = 1: tensor_order
            epsilons = randn(1,rank);
            U{i} = repmat((1:1:dims)',1,2).*repmat(epsilons,dims,1) + randn(dims,2)*0.1;
            [U{i}, aa1] = qr(U{i}, 0);
            A{i} = ones(dims,dims) - pdist2(U{i},U{i});
            L{i} = diag(sum(A{i},2))-A{i};
        end
        % generate the core tensor
        index_temp = ones(1, core_dims);
        C = tendiag(index_temp, core_dims);
        % generate low-rank tensor
        X = ttensor(tensor(C), U);
        data = double(X);
    otherwise
        fprintf('We do not have such method. Please specify a number from 1 to 4!\n');
end
