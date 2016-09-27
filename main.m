clc
clear
close all

addpath('tensor_toolbox')

% -------------- Generate the Synthetic Tensor Data -----------------------
% paramters setting up
tensor_dims = [100, 100, 100];
tensor_order = length(tensor_dims);
rank = 2;
method = 1;
% generate data
[data, A, L] = GenerateSyntheticData(tensor_dims, rank, method);
% -------------------------------------------------------------------------

% ------------------------------- Main  -----------------------------------
% paramters setting up
fraction = 0.8;                                 % fraction of missing data
Tsize = size(data);                             % size of tensor data
Omega = (rand(Tsize) > fraction);               % index of observed data
T     = data;
T(logical(1-double(Omega))) = 0;                % observed tensor data

% convert to sparse tensor format
idx = find(T~=0);
idx_rest = find(T==0);
[i, j, k] = ind2sub(size(T), idx);
subs = [i, j, k];
vals = T(idx);
T_sp = sptensor( subs, vals, Tsize, 0);

% Save records for this loop
RelErrs = [];
MsrErrs = [];
Times = [];
Iterations = [];

fprintf('---------------The fraction of missing data is %f.------------\n', fraction);

alpha     = 0.1*ones(1,tensor_order);
maxIter   = 3000;
epsilon   = 1e-5;
inDims    = rank;
lambda    = 1e-3;

tic
[X_O_aircp, errList_aircp] = AirCP(...
                                 T,...          % a tensor whose elements in Omega are used for estimating missing value
                                 Omega,...      % the index set indicating the obeserved elements
                                 L,...          % Graph Laplacians of similarity matrices from side information 
                                 alpha,...      % the coefficient of the objective function,  i.e., \alpha_i*\|U_{i}\|_{*}
                                 inDims,...     % the given rank of the tensor
                                 lambda,...
                                 maxIter,...    % the maximum iterations
                                 epsilon...     % the tolerance of the relative difference of outputs of two neighbor iterations 
                                 ); 
time = toc

% calculate relative error
relErr = norm(X_O_aircp(:) - data(:), 'fro')/ norm(data(:), 'fro');
fprintf('The relative error of AirCP is: %f.\n', relErr);
% calcuate MSR
msrErr = sqrt(mean((X_O_aircp(:)-data(:)).^2));
fprintf('The root mean square error of AirCP is: %f.\n', msrErr);

% plotting compared results
for j = 1:tensor_dims(3)
    subplot(1,3,1); imagesc(data(:,:,j));
    title(['Original: frame ', num2str(j)]);
    xlabel('Dim-1')
    ylabel('Dim-2')
    subplot(1,3,2); imagesc(T(:,:,j));
    title(['Sampled: frame ', num2str(j)]);
    xlabel('Dim-1')
    ylabel('Dim-2')
    subplot(1,3,3); imagesc(X_O_aircp(:,:,j));
    title(['Recovered: frame ', num2str(j)]);
    xlabel('Dim-1')
    ylabel('Dim-2')
end

RelErrs(end+1) = relErr;
MsrErrs(end+1) = msrErr;
Times(end+1) = time;
Iterations(end+1) = length(find(errList_aircp~=0));