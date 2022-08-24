function [loss, alphas] = ctcForward(X, T, XMask, TMask, blankIdx)
```matlab
% ctcForward     

% Compute the loss and the forward variables alphas by looping forward through each time step t in X to find alphas(:, :, t) 
% 通过向前循环 X 中的每个时间步 t 来计算损失和前向变量 alphas 以找到 alphas(:, :, t)。
% (i.e. total probability). （即总概率）

% Begin by initializing alphas(:, :, 1) with entries in X 
% 首先用 X 中的条目初始化 alphas(:, :, 1)
% (i.e. total probability at first time step equal to probability at first time step), 
% （即第一个时间步的总概率等于第一个时间步的概率），
% and recursively find the alphas(:, :, t+1) at the next step by multiplying alphas(:, :, t) by X(:, :, t) 
% 然后通过将 alphas(:, :, t) 乘以 X(:, :, t)在下一步递归找到 alphas(:, :, t+1)
% (i.e. total probabilities at next time step equal to total probabilities at current time step multiplied by probabilities at current time step). 
% （即下一个时间步的总概率等于当前时间步的总概率乘以当前时间步的概率）。
% For more information, refer to original paper https://www.cs.toronto.edu/~graves/icml_2006.pdf
% 有关详细信息，请参阅原始论文

% Extract the number of observations
% X will always be 'CBT'. This is ensured in the ctc dlarray method.
numObs = size(X, 2);

% Calculate the loss and the forward variables for each observation in a
% vectorized manner
[loss, alphas] = iComputeAlphasAndLoss(X, T, XMask, TMask, blankIdx, numObs);

end

%% Helper functions

function [loss, alphas] = iComputeAlphasAndLoss(X, T, XMask, TMask, blankIdx, numObs)
%% Initialization

% Extract the sequence lengths, max sequence lengths, and the augmented
% targets 'augT'
[ XLens, ~, augTLens, XLenMax, ~, augTLenMax, augT ] = ...
    nnet.internal.cnnhost.util.initializeCtc(X, T, XMask, TMask, blankIdx, numObs);

% Initialize the forward variables (i.e. the total probabilities at each
% time step)
alphas = zeros( numObs, augTLenMax, XLenMax, 'like', X );

% Initialize the normalization factors
Ct = ones( numObs, XLenMax, 'like', X );

%% Populate the entries of alphas( :, :, 1 ) and normalize

% The first entry will always be a blank character for every observation
alphas( :, 1, 1 ) = X( blankIdx, :, 1 );

% The second entry will be the first target entry for each observation
indC = T( :, 1 )';
indB = 1 : numObs;
indT = ones( 1, numObs );
alphas( :, 2, 1 ) = X( sub2ind( size(X), indC, indB, indT ) );

% Calculate the normalization factors at first time step
Ct( :, 1 ) = sum( alphas( :, :, 1 ), 2 );

% Normalize forward variables to prevent underflow
alphas( :, :, 1 ) = nnet.internal.cnnhost.util.vecDivide( alphas( :, :, 1 ), ...
    Ct( :, 1 ) );

for t = 2:XLenMax
    %% Initialization at each time step t

    % Initialize a vector which will be multiplied by X(augT, :, t) (i.e. the 
    % probabilities at time step t for augmented target sequence) to give 
    % alphas(:, :, t) (i.e. total probabilities at time step t)
    probMultiplier = zeros( numObs, augTLenMax, 'like', X );

    % Find valid positions given by the range begin : finish where begin and 
    % finish are arrays of length 'B' (i.e. a begin and finish index for
    % each observation)
    begin = max( 1, augTLens - 2 * ( XLens - t ) - 1 );
    finish = min( 2 * t, augTLens ); 
    
    % Find the multidimensional row and column indices for begin : finish 
    [ indObsValid, indAugTValid ] = nnet.internal.cnnhost.util.vecColon2Ind( ...
        begin, finish );
    
    %% Determine entries which depend on alphas( :, s, t - 1 )
    % All entries with s = begin : finish depend on alphas( :, s , t - 1 ), 
    % for each observation
    
    % Update probMultiplier by adding the dependency on 
    % alphas( :, s, t - 1 ) for indices indObsValid and indAugTValid
    probMultiplier = iAddTermsToProbMultiplier( probMultiplier, alphas, ...
        indObsValid, indAugTValid, 0, t);
    
    %% Determine entries which also depend on alphas( :, s - 1, t - 1 ) 
    % All entries with s = begin : finish where s ~= 1 depend on 
    % alphas( :, s - 1, t - 1 ), for each observation
    
    % Initialize the multidimensional indices to add the second term
    indObsSecond = indObsValid;
    indAugTSecond = indAugTValid;
    
    % Remove any indices which are equal to 1
    indAugT1 = indAugTSecond == 1;
    indObsSecond( indAugT1 ) = [];
    indAugTSecond( indAugT1 ) = [];
    
    % Update probMultiplier by adding the dependency on 
    % alphas( :, s - 1, t - 1 ) for indices indObsSecond and indAugTSecond
    probMultiplier = iAddTermsToProbMultiplier( probMultiplier, alphas, ...
        indObsSecond, indAugTSecond, 1, t);
    
    %% Determine entries which also depend on alphas( :, s - 2, t - 1 )
    % All entries with s = begin : finish where s is not odd, s ~= 2 and 
    % AugT(s) ~= AugT(s) - 2 (i.e. repeat character) depend on 
    % alphas( :, s - 2, t - 1 ), for each observation
    
    % Initialize the multidimensional indices to add the third term
    indObsThird = indObsSecond;
    indAugTThird = indAugTSecond;
    
    % Remove any odd indices
    indOdd = mod( indAugTThird, 2 ) == 1;
    indObsThird( indOdd ) = [];
    indAugTThird( indOdd ) = [];
    
    % Remove any indices which are equal to 2
    indAugT2 = indAugTThird == 2;
    indObsThird( indAugT2 ) = [];
    indAugTThird( indAugT2 ) = [];
    
    % Remove any indices which are repeats (i.e. AugT == AugT - 2)
    indRepeat = augT( sub2ind( size(augT), indObsThird, indAugTThird ) ) ...
        == augT( sub2ind( size(augT), indObsThird, indAugTThird - 2 ) );
    indObsThird( indRepeat ) = [];
    indAugTThird( indRepeat ) = [];
    
    % Update probMultiplier by adding the dependency on 
    % alphas( :, s - 2, t - 1 ) for indices indObsThird and indAugTThird
    probMultiplier = iAddTermsToProbMultiplier( probMultiplier, alphas, ...
        indObsThird, indAugTThird, 2, t);
    
    %% Populate the entries of alphas( :, :, t ) and normalize

    % Find channel, batch, and time indices to index into X
    indC = reshape( augT, 1, [] );
    indB = reshape( ( 1 : numObs )' .* ones( size(augT) ), 1, [] );
    indT = t * ones( 1, numel(augT) );
    
    % Multiply the probMultiplier by X(augT, :, t)
    alphas(:, :, t) = probMultiplier .* reshape( ...
        X( sub2ind( size(X), indC, indB, indT ) ), ...
        numObs, augTLenMax );
    
    % Normalize forward variables to prevent underflow
    Ct( :, t ) = sum( alphas( :, :, t ), 2 );
    alphas( :, :, t ) = nnet.internal.cnnhost.util.vecDivide( alphas( :, :, t ), ...
        Ct( :, t ) );
    
end

%% Calculate loss

% Calculate loss as the negative sum of the log of the normalized forward
% variables, using XMask to ignore values which should not count
loss = -sum( log(nnet.internal.cnn.util.boundAwayFromZero(Ct)) .* XMask, 2);

loss = mean(loss);

% Scale back up for later calculation in the backward method
alphas = alphas .* repmat( permute( Ct, [ 1 3 2 ] ), [ 1 augTLenMax 1 ] );
end

function probMultiplier = iAddTermsToProbMultiplier( probMultiplier, alphas, indObs, indAugT, minusIndex, t)
% Find the linear indices for probMultiplier
indProbMultiplier = sub2ind( size(probMultiplier), indObs, ...
    indAugT );

% Find the linear indices for alphas
indAlphas = sub2ind( size(alphas), indObs, indAugT - minusIndex, ...
    ( t - 1 ) * ones( size(indAugT) ) );

% Add terms to probMultiplier
probMultiplier( indProbMultiplier ) = probMultiplier( indProbMultiplier ) ...
    + alphas( indAlphas );
end
