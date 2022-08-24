function dX = ctcBackward(dZ, X, T, XMask, TMask, blankIdx, alphas)
% ctcBackward     Compute the gradients dX using the forward and backward 
% variables, alphas and betas. Begin by computing betas by looping
% backwards in time through each time step t in X to find betas(:, :, t).
% First, initialize betas(:, augTLens, t) where augTLens corresponds to
% the final time step for each observation. Recursively find betas(:, :, t+1)
% by multiplying betas(:, :, t) by X(:, :, t). Next, compute dX use the 
% product of alphas and betas to compute dX, as detailed in original paper
% https://www.cs.toronto.edu/~graves/icml_2006.pdf

%   Copyright 2020 The MathWorks, Inc.

% Extract the number of observations
% X will always be 'CBT'. This is ensured in the ctc dlarray method.
numObs = size(X, 2);

% Calculate the gradients for each observation in a vectorized manner
dX = iComputeBetasAndGrads(alphas, X, T, XMask, TMask, blankIdx, numObs);

dX = dZ .* dX ./ numObs;

end

%% Helper functions

% Compute the gradients using the backward variable betas
function dX = iComputeBetasAndGrads(alphas, X, T, XMask, TMask, blankIdx, numObs)
%% Initialization

% Extract the sequence lengths, max sequence lengths, and the augmented
% targets 'augT'
[ XLens, TLens, augTLens, XLenMax, ~, augTLenMax, augT ] = ...
    nnet.internal.cnnhost.util.initializeCtc(X, T, XMask, TMask, blankIdx, numObs);

% Find the row and column indices for 1 : augTLens where augTLens is a
% vector
[indObsTrue, indAugTrue] = nnet.internal.cnnhost.util.vecColon2Ind( ones( ...
    size(augTLens) ), augTLens );

% Create the mask for the augmented target sequence
augTMask = false( numObs, augTLenMax );
augTMask( sub2ind( size(augTMask), indObsTrue, indAugTrue ) ) = true;

% Initialize the backward variables
betas = zeros( numObs, augTLenMax, XLenMax, 'like', X );

% Initialize the normalization factors
Dt = ones( numObs, XLenMax, 'like', X );

%% Populate the entries of betas( :, :, XLens ) and normalize
% XLens might be different for each observation, so use clever ...
% indexing to find the correct entries

% The last entry will always be a blank character for every observation
indC = blankIdx * ones( numObs, 1 );
indB = ( 1 : numObs )';
indT = XLens;
betas( sub2ind( size(betas), indB, augTLens, indT ) ) = ...
    X( sub2ind( size(X), indC, indB, indT ) );

% The second to last entry will always be the last target entry for each
% observation
indC = T( sub2ind( size(T), indB, TLens ) ) .* ones( numObs, 1 );
betas( sub2ind( size(betas), indB, augTLens - 1, indT ) ) = ...
    X( sub2ind( size(X), indC, indB, indT ) );

% Determine indices of Dt( :, XLens )
indDt = sub2ind( size(Dt), indB, indT );

% Determine indices of betas( :, 1:augTLenMax, XLens )
indBetas = sub2ind( size(betas), ...
    reshape( indB .* ones( numObs, augTLenMax ), 1, []), ...
    reshape( ones( numObs, augTLenMax ) .* ( 1 : augTLenMax ), 1, [] ), ...
    reshape( indT .* ones( numObs, augTLenMax ), 1, [] ) );

% Calculate the normalization factors at first time step
Dt( indDt ) = sum( reshape( betas( indBetas ), numObs, augTLenMax ), 2 );

% Normalize backward variables to prevent underflow
betas( indBetas ) = nnet.internal.cnnhost.util.vecDivide( ...
    reshape( betas( indBetas ), numObs, augTLenMax), ...
    Dt( indDt ) );

for t = XLenMax-1:-1:1
    %% Initialization at each time step t

    % Initialize a vector which will be multiplied by X(augT, :, t) (i.e. the 
    % probabilities at time step t for augmented target sequence) to give 
    % betas(:, :, t) 
    probMultiplier = zeros( numObs, augTLenMax, 'like', X );
    
    % Find valid positions given by the range begin : finish where begin and 
    % finish are arrays of length 'B' (i.e. a begin and finish index for
    % each observation)
    begin = max( 1, augTLens - 2 * ( XLens - t ) - 1 );
    finish = min( 2 * t, augTLens ); 
    
    % Find the multidimensional row and column indices for begin : finish 
    [ indObsValid, indAugTValid ] = nnet.internal.cnnhost.util.vecColon2Ind( ...
        begin, finish );
    
    %% Determine entries which depend on betas( :, s, t + 1 )
    % All entries with s = begin : finish depend on betas( :, s , t + 1 ), 
    % for each observation
    
    % Update probMultiplier by adding the dependency on 
    % betas( :, s, t + 1 ) for indices indObsValid and indAugTValid
    probMultiplier = iAddTermsToProbMultiplier( probMultiplier, betas, ...
        indObsValid, indAugTValid, 0, t);
    
    %% Determine entries which also depend on betas( :, s + 1, t + 1 )
    % All entries with s = begin : finish where s ~= augTLens depend on 
    % betas( :, s + 1, t + 1 ), for each observation
    
    % Initialize the multidimensional indices to add the second term
    indObsSecond = indObsValid;
    indAugTSecond = indAugTValid;
    
    % Remove any indices which are equal to augTLens
    % Each observation will likely remove a different index so use
    % logical indexing to determine the index to be removed
    indAugTLen = any( indObsSecond == ( 1 : numObs )' & ...
        indAugTSecond == augTLens, 1 );
    indObsSecond( indAugTLen ) = [];
    indAugTSecond( indAugTLen ) = [];
    
    % Update probMultiplier by adding the dependency on 
    % betas( :, s + 1, t + 1 ) for indices indObsSecond and indAugTSecond
    probMultiplier = iAddTermsToProbMultiplier( probMultiplier, betas, ...
        indObsSecond, indAugTSecond, 1, t);
    
    %% Determine entries which also depend on betas( :, s + 2, t + 1 )
    % All entries with s = begin : finish where s is not odd, s ~= 
    % augTLens - 1 and AugT(s) ~= AugT(s) + 2 (i.e. repeat character) 
    % depend on betas( :, s + 2, t + 1 ), for each observation
    
    % Initialize the multidimensional indices to add the third term
    indObsThird = indObsSecond;
    indAugTThird = indAugTSecond;
    
    % Remove any odd indices
    indOdd = mod( indAugTThird, 2 ) == 1;
    indObsThird( indOdd ) = [];
    indAugTThird( indOdd ) = [];
    
    % Remove any indices which are equal to augTLens - 1
    indAugTLensMinusOne = any( indObsThird == ( 1 : numObs )' & ...
        indAugTThird == augTLens - 1, 1 );
    indObsThird( indAugTLensMinusOne ) = [];
    indAugTThird( indAugTLensMinusOne ) = [];
    
    % Remove any indices which are repeats (i.e. indAugT3 == indAugT3 + 2)
    indRepeats = augT( sub2ind( size(augT), indObsThird, indAugTThird ) ) == ...
        augT( sub2ind( size(augT), indObsThird, indAugTThird + 2 ) );
    indObsThird( indRepeats ) = [];
    indAugTThird( indRepeats ) = [];
    
    % Update probMultiplier by adding the dependency on 
    % betas( :, s + 2, t + 1 ) for indices indObsThird and indAugTThird
    probMultiplier = iAddTermsToProbMultiplier( probMultiplier, betas, ...
        indObsThird, indAugTThird, 2, t);
    
    %% Populate the entries of betas( :, :, t ) and normalize

    % Find channel, batch, and time indices to index into X
    indC = reshape( augT, 1, [] );
    indB = reshape( ( 1 : numObs )' .* ones( size(augT) ), 1, [] );
    indT = t * ones( 1, numel(augT) );
    
    % Logical array specifying if t is equal to XLens
    isEqualToXLens = t == XLens;
    
    % Multiply the probMultiplier by X(augT, :, t)
    % If t == XLens, have already calculated betas( :, :, t ) before the 
    % main loop, so just set betas( ind, :, t ) = betas( ind, :, t ) where
    % XLens(ind) == t
    betas( :, :, t ) = not( isEqualToXLens ) .* probMultiplier .* ...
        reshape( X( sub2ind( size(X), indC, indB, indT ) ), numObs, augTLenMax ) ...
        + ( isEqualToXLens .* betas( :, :, t) );
    
    % Normalize forward variables to prevent underflow
    % The above logic applies here as well.
    Dt( :, t ) = not( isEqualToXLens ) .* sum( betas( :, :, t ), 2 ) + ...
        isEqualToXLens .* Dt( :, t );
    betas( :, :, t ) = not( isEqualToXLens ) .* ...
        nnet.internal.cnnhost.util.vecDivide( betas( :, :, t ), Dt( :, t ) ) + ...
        isEqualToXLens .* betas( :, :, t );
end

%% Calculate gradients

% Initialize gradients
dX = zeros(size(X), 'like', X);

% Scale betas back up
betas = betas .* repmat( permute( Dt, [ 1 3 2 ] ), [ 1 augTLenMax 1 ] );

% Elementwise product of forward and backward variables
ab = alphas .* betas;

% Bound X away from zero as we will be dividing by X in the following loop
X = nnet.internal.cnn.util.boundAwayFromZero(X);

% Batch and time indices to index into dX and X
indB = reshape( ( 1 : numObs )' .* ones( numObs, XLenMax ), 1, [] );
indT = reshape( repmat( 1 : XLenMax, numObs, 1 ), 1, [] );

for s = 1:augTLenMax
    % Find the channel dimension by indexing into augT( :, s )
    indC = reshape( augT( :, s ) .* ones( numObs, XLenMax ), 1, [] );
    
    % Linear index for X and dX
    indX = sub2ind( size(dX), indC, indB, indT );
    
    % Calculate intermediate gradient value, ignoring any values s > augTLens
    dX( indX ) = dX( indX ) + ...
        reshape( ab( :, s, : ) .* augTMask( :, s ), 1, [] );
    
    % Normalize the elementwise product of forward and backward variables
    ab( :, s, : ) = ab( :, s, : ) ./ ...
        reshape( X( indX ), numObs, 1, XLenMax );
end

% Sum over the augT dimension
Zt = sum( ab, 2 );
Zt = reshape( Zt, [ numObs, XLenMax ] );

% Bound Zt away from zero as will be dividing by Zt
Zt = nnet.internal.cnn.util.boundAwayFromZero(Zt);

% Use repmat and permute to make Zt and XMask the same dim as dX and X
Zt = repmat( permute( Zt, [ 3 1 2 ] ), [ size( X, 1 ) 1 1 ] ) ;
XMask = repmat( permute( XMask, [ 3 1 2 ] ), [ size( X, 1 ) 1 1 ] );

% Calculate gradients, using XMask to ignore values which should not count
dX( XMask ) = - dX( XMask ) ./ ( X( XMask ).^2 .* Zt( XMask ) );

end

function probMultiplier = iAddTermsToProbMultiplier( probMultiplier, betas, indObs, indAugT, addIndex, t)
% Find the linear indices for probMultiplier
indProbMultiplier = sub2ind( size(probMultiplier), indObs, ...
    indAugT );

% Find the linear indices for betas
indBetas = sub2ind( size(betas), indObs, indAugT + addIndex, ...
    ( t + 1 ) * ones( size(indAugT) ) );

% Add terms to probMultiplier
probMultiplier( indProbMultiplier ) = probMultiplier( indProbMultiplier ) ...
    + betas( indBetas );
end