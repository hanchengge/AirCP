function [X, errList] = AirCP(T, Omega, L, alpha, tensor_rank, lambda, maxIter, tol);
% 
% This routine solves the auxiliary information regularized CP tensor 
% completion via Alternation Direction Method of Multipliers (ADMM), which 
% has been presented in our paper:
% 
% 1. Hancheng Ge, James Caverlee, Nan Zhang, Anna Squicciarini:
% Uncovering the Spatio-Temporal Dynamics of Memes in the Presence of 
% Incomplete Information, CIKM, 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Our ADMM algorithm for Nuclear Norm regularized CP tensor completion:
% 
%
% min_{U^{(n)}X} \frac{1}{2}\|X-[\![U^{(1)},U^{(2)},U^{(3)}]\!]\|^2_F 
%                + \frac{\lambda}{2}\sum_{n=1}^{3}\|U^{(n)}\|_{F}^{2}
%                + \sum_{n=1}^{3}\alpha_{n}\mathrm{tr}({Z^{(n)}}^T L_n Z^{(n)})
% s.t., \bm{\Omega}*X=\bm{T, U^{(n)} = Z^{(n)} \geq 0, n=1,2,3,
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Author: Hancheng Ge, Date: 09/20/2016
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Default parameters
if nargin < 8
    tol = 1e-5;  
end

if nargin < 7
    maxIter = 500; 
end
% tunning parameter for Tikhonov regularization terms
if nargin < 6
    lambda = 1/sqrt(max(size(T))); 
end

if nargin < 5
    tensor_rank = 40;
end

ndims_T = ndims(T);
if nargin < 4
    alpha = ones(ndims_T, 1);
    alpha = alpha / sum(alpha);
end


% ------------------------ Parameter Setting Up ---------------------------
% step size
eta = 1e-4;
% increasing factor
rho = 1.05;
% initialize the estimated complete tensor
X = T;
X(logical(1-Omega)) = mean(T(Omega));
% initialize error list used to record training errors
errList = zeros(maxIter, 1);
% number of dimensions of the observed tensor
ndims_T = ndims(T);
% norm of the observed tensor
normT   = norm(T(:));
% size of the observed tensor
T_size  = size(T);
% initialize latent matrices U(1), U(2), ..., U(n) with random number, Z(1),
% Z(2), ..., Z(n) with zeros and Y(1), Y(2), ..., Y(n) with zeros.
size_II = []; 
for i = 1:ndims_T       
    U{i} = rand(T_size(i),  tensor_rank);
    Y{i} = zeros(T_size(i), tensor_rank);
    Z{i} = zeros(T_size(i), tensor_rank);
    size_II = [size_II, tensor_rank];
end
index_temp = ones(1, tensor_rank);
II = tendiag(index_temp, size_II);
% record the previous estimated results
X_pre = X;

% % --------------------------- Iteration Scheme ----------------------------
for k = 1: maxIter   
    if mod(k, 20) == 0
        fprintf('AirCP: iterations = %d   difference=%f\n', k, errList(k-1));
    end
    
    % update step eta
    eta = eta * rho;

    % update Z 
    for i = 1:ndims_T
        temp_1 = eta*U{i} - Y{i};
        temp_2 = eta*eye(size(L{i})) + alpha(i)*L{i};
        Z{i} = (temp_2 + 0.00001*eye(size(temp_2)))\temp_1;
    end    

    % update U
    for i = 1:ndims_T
        % calculate intermedian tensor and its mode-n unfolding
        midT = tensor(II);
        % calculate Kronecker product of U(1), ..., U(i-1),U(i+1), ...,U(n)
        for m = 1:ndims_T
            if m == i
            continue;
            end
            midT = ttm(midT, U{m}, m);
        end
        unfoldD_temp = tenmat(midT, i);  

        temp_Z = eta*Z{i} + Y{i}; 
        temp_B = unfoldD_temp.data*unfoldD_temp.data';
        temp_B = temp_B + eta*eye(tensor_rank,tensor_rank) + lambda*eye(tensor_rank,tensor_rank);
        temp_B = temp_B + 0.00001*eye(size(temp_B));
        temp_C = tenmat(X, i);
        temp_D = temp_C.data*unfoldD_temp.data';
        U{i}   = (temp_D + temp_Z)/temp_B;
    end;
    clear unfoldD_temp temp_B temp_Z temp_C  
  
    % update X  
    midT = tensor(II);
    midT = ttm(midT, U, [1:ndims_T]);  
    X = midT.data;
    X(Omega) = T(Omega);
  
    % update Lagrange multiper   
    for i = 1:ndims_T      
        Y{i} = Y{i} + eta*(Z{i} - U{i});
    end
    
    % checking the stop criteria
    stopC = norm(X_pre(:) - X(:))/normT; 
    X_pre = X; 
    errList(k) = stopC;
 
    if stopC < tol
       break;
    end  
end