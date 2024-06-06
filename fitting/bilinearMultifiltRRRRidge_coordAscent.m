function [wU,wVt,wwfilts,fval] = bilinearMultifiltRRRRidge_coordAscent(Xin,Yout,rnks,lambda,opts)
    % [wU,wVt,wwfilts,fval] = bilinearMultifiltRRR_coordAscent(Xin,Yout,rnks,lambda,opts)
    % 
    % Computes low-rank regression estimate using coordinate ascent in the
    % multi-filter reduced rank regression (RRR) problem setting 
    % the parameter vector.
    %
    % Finds solution to: 
    %
    %   argmin_{Wi} ||Y - \sum_i Xi Wi||_F^2 + lambda*\sum_i ||W_i||_F^2
    %
    % where the matrices {Wi} are all low rank, parametrized as
    %
    %   Wi = Ui Vi^T, 
    %
    % and Ui and Vi^T denote row and column vector matrices, respectively.
    %
    % Inputs:
    % -------
    %    Xin [{1 x k}]  = cell array of input population responses {(T x nin1), ..., (T x nink)}
    %   Yout [T x nout] = output population response matrix 
    %   rnks [1 x k]    = rank of each low-rank filter
    %
    %   lambda [1 x 1]  = ridge parameter (optional)  
    %
    %     opts [struct] =   options struct (optional)
    %         fields: 'MaxIter' [25], 'TolFun' [1e-6], 'Display' ['iter'|'off']
    %
    % Outputs:
    % -------
    %        wU [{k x 1}] = cell array of column vector matrices 
    %       wVt [{k x 1}] = cell array of row vector matrices    
    %   wwfilts [{k x 1}] = cell array of low rank filters
    %      fval [1 x 1]   = loss function (squared error)
    
    
    % ---------------------------------------------------
    % set optimization options
    % ---------------------------------------------------
    if (nargin < 5) || isempty(opts)
        opts.default = true;
    end
    if ~isfield(opts, 'MaxIter'); opts.MaxIter = 25; end
    if ~isfield(opts, 'TolFun'); opts.TolFun = 1e-6; end
    if ~isfield(opts, 'Display'); opts.Display = 'iter'; end
    % ---------------------------------------------------
    % Extract sizes of inputs
    % ---------------------------------------------------
    
    nin = cellfun(@(x)size(x,2),Xin); % get # of cells in each population
    ninpops = length(nin); % number of input populations
    nintot = sum(nin);     % total number of input neurons
    nout = size(Yout,2);
    
    % check inputs
    if length(rnks)~=ninpops
        error('length of ``rnks'' doesn''t match # of input populations (# cells in Xin)');
    end
    
    % ---------------------------------------------------
    % Compute Sufficient Statistitics
    % ---------------------------------------------------
    
    % Compute sufficient statistics
    Xfull = cell2mat(Xin);
    XX = (Xfull'*Xfull);
    XY = Xfull'*Yout; % input-output cross-covariance
    
    % Divide sufficient statistics into blocks of a cell array
    XXc = mat2cell(XX,nin,nin);    % divide XX into blocks
    XXc_cols = mat2cell(XX,nintot,nin); % divide XX into columns 
    XYc = mat2cell(XY,nin,nout);   % divide XY into blocks
    rnks_cell = num2cell(rnks(:)); % filter ranks as cell array
    totrnks = sum(rnks);
    
    % ---------------------------------------------------
    % Initialize fit using SVD of ridge regression estimate
    % ---------------------------------------------------
    % (Note: an alternative would be to initialize with RRR)
    
    % compute ridge regression estimate
    wridge = XX\XY; % ridge regression solution 
    
    % do SVD on each relevant portion of w0
    wridgefilts = mat2cell(wridge,nin,nout); % convert into cell array for individual filters
    wU = cell(ninpops,1);
    wVt = cell(ninpops,1);
    for jj = 1:ninpops
        [u0,s,v0] = svd(wridgefilts{jj},'econ'); % SVD
        ii = 1:rnks(jj); % indices of singular vectors to keep
        wU{jj} = u0(:,ii)*sqrt(s(ii,ii));  % left singular vectors
        wVt{jj} = sqrt(s(ii,ii))*v0(:,ii)'; % right singular vectors (weighted by singular values)
    end
    wwfilts = cellfun(@mtimes,wU,wVt,'UniformOutput',false); % full weights in cell array
    
    % ---------------------------------------------------
    % Set up coordinate ascent 
    % ---------------------------------------------------
    
    % Convert initial filters into a matrix
    ww0 = cellfun(@mtimes,wU,wVt,'UniformOutput',false);
    ww0 = cell2mat(ww0);
    
    % Evaluate the loss function at the start of training
    fval = 0.5*trace(ww0'*XX*ww0) - sum(sum((ww0.*XY))) + 0.5*lambda*sum(sum(ww0.^2));
    fchange = inf; % change in function value
    iter = 1;  % iteration counter
    
    % Report initial loss (if 'Display' set to 'iter')
    if strcmp(opts.Display, 'iter')
        fprintf('--- Coordinate descent of multi-filter RRR loss ---\n');
        fprintf('Iter 0: fval = %.4f\n',fval);
    end
    
    % % define some useful cell functions we'll need for 
    fun_Ureshape = @(v,r)(reshape(v,[],r));    % function to reshape U matrices 
    fun_mtimesBTrp = @(A,B)(vec(mtimes(A,B'))); % function to compute kronecker of transpose with matrix
    fun_mtimesATrp = @(A,B)((mtimes(A',B))); % function to compute kronecker of transpose with matrix

    % ridge penalty term
    ridge_U = lambda * speye(totrnks); % add to uXXu when updatin V
    ridge_V = lambda * speye(rnks*nin'); % add to vXXv when updating U
    
    % ---------------------------------------------------
    % Optimize:  Run alternating coordinate ascent on U and Vt
    % ---------------------------------------------------
    
    while (iter <= opts.MaxIter) && (abs(fchange) > opts.TolFun)

        % calculate semi-orthogonal U / V for ridge update
        [U_unit,S,V_unit] = cellfun(@(x) svd(x), wwfilts,'uni',0);
        wU = arrayfun(@(ii) U_unit{ii}(:,1:rnks(ii)), 1:ninpops, 'uni', 0)';
        wVt = arrayfun(@(ii) V_unit{ii}(:,1:rnks(ii))', 1:ninpops, 'uni', 0)';
        
        % ===========================
        % Update U (column vectors)
        % ===========================
            
        % Compute Covariance (Vt' o Vt)(X o X)
        Vt = cell2mat(wVt);
        VtV = mat2cell(Vt*Vt',rnks,rnks);
        vXXv = cell2mat(cellfun(@kron,VtV,XXc,'UniformOutput',false));
        
        % Compute Vectorized Cross-covariance (Vt' o X')Y
        vXY = cell2mat(cellfun(fun_mtimesBTrp,XYc,wVt,'UniformOutput',false));
            
        % solve regression problem for U
        Unew = (vXXv+ridge_V)\vXY; 
        
        % Insert Unew into cell array
        wU = mat2cell(Unew,nin.*rnks,1); % convert to cell array
        wU = cellfun(fun_Ureshape,wU,rnks_cell,'UniformOutput',false); % reshape each filter to correct shape
    
        % ===========================
        % Update Vt (row vectors)
        % ===========================
        
        % Compute covariance U' X' X U
        XXu = cellfun(@mtimes, XXc_cols, wU', 'UniformOutput',false); % multiply columns of XX by wU
        XXu = mat2cell(cell2mat(XXu),nin,totrnks);  % reshape so we can left-multiply by wU
        uXXu = cell2mat(cellfun(fun_mtimesATrp,wU,XXu, 'UniformOutput',false)); % left multiply by wU^T and concatenate

        % Compute cross-covariance U' X' Y
        uXY = cell2mat(cellfun(fun_mtimesATrp,wU,XYc, 'UniformOutput',false));

        % solve regression problem for V^T
        Vtnew = (uXXu+ridge_U)\uXY;

        % insert into cell array
        wVt = mat2cell(Vtnew,rnks,nout); % convert to cell array
        
        % ===========================
        % Compute each weight matrix from its low-rank components
        % ===========================
        
        wwfilts = cellfun(@mtimes,wU,wVt,'UniformOutput',false); % full weights in cell array
        
        % ===========================
        % Evaluate loss function & size of change
        % ===========================
        
        ww = cell2mat(wwfilts);  % put filters into a single weight matrix
        fvalnew = 0.5*sum(sum(ww.*(XX*ww))) - sum(sum((ww.*XY))) + 0.5*lambda*sum(sum(ww.^2));
        fchange = fval-fvalnew;
        fval = fvalnew;
        iter = iter+1;
        
        % Report change in error (if desired)
        if strcmp(opts.Display, 'iter')
            fprintf('Iter %d: fval = %.4f,  fchange = %.4f\n',iter-1,fval,fchange);
        end
    end

    if (abs(fchange) > opts.TolFun)
        warning('bilinearMultifiltRRRRidge_coordAscent: did not converge to desired tolerance');
    end
    
    