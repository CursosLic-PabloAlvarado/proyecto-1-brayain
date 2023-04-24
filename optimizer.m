% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 4

% (C) 2023 <Team brAIn>


## optimizer Execute a gradient descent process on a given error function.
classdef optimizer < handle

  properties (Access = public)
    ## Training parameters and their default values
    minibatch = 16;

    alpha   = 0.01;  ## Learning rate
    beta1   = 0.70;  ## Momentum: 0 para no usar momentum
    beta2   = 0.99;  ## Polo de filtro de cuadrados (0: no usar Adam))
    epsilon = 1e-7;  ## Evite divisiones por cero en Adam
    maxiter = 200;   ## Número de iteraciones

    mbmode  = 'withrep'; ## Minibatch mode with replacement

    method  = 'batch';  ## "batch", "stochastic", "momentum", "rmsprop", "adam"
    show    = 'progress';
  endproperties

  properties (Access = private)

    ## Remaining samples used while training with no-replacement
    remainingIndices=[];

    ## Training state
    v=[]; ## Last gradient

  endproperties

  methods (Access = private)

    ## ----------------------------------------------------------------------
    ## Updaters
    ##
    ## Each updater is used to update the theta parameter vector/matrix
    ## It needs the current parameters tc, and the current gradient g
    ## and it returns the updated parameters vector tn depending on the
    ## method.


    ## theta update with momentum
    function tn = updateMomentum(self,tc,g)
      self.v = self.beta1*self.v + (1-self.beta1)*g;
      tn = tc - self.alpha*self.v;
    endfunction
    ## theta update with remaining methods
    function tn = update(self, tc, g)
      assert(size(g)==size(tc));
      tn = tc - self.alpha*g;
    endfunction



    ## ----------------------------------------------------------------------
    ## Show progress methods

    ## Shows nothing (silent mode)
    function showNothing(self,iteration,currentError)
      ## Nothing is done
    endfunction

    ## Shows a dot at each iteration
    function showDots(self,iteration,currentError)
       printf(".");
    endfunction

    ## Shows iteration number and loss value
    function showLoss(self,iteration,currentError)
       printf("Iteration %i/%i: %f\n",iteration,self.maxiter,currentError);
    endfunction

    ## Show progress with 70 steps
    function showProgress(self,iteration,currentError)
      pc=round(100*iteration/self.maxiter);
      done=round(pc*0.7);

      printf("%03i%% %s\r",pc,repmat('=',1,done));
    endfunction


    ## ----------------------------------------------------------------------
    ## Sampling without replacement
    function idx=samplerMBnr(self,X)
      while (length(self.remainingIndices) < self.minibatch)
        newIdx=randperm(rows(X))';
        self.remainingIndices = [self.remainingIndices;newIdx];
      endwhile
      idx=self.remainingIndices(1:self.minibatch);
      self.remainingIndices=self.remainingIndices(self.minibatch:end);
    endfunction

  endmethods

  methods (Access = public)

    % Construct an optimizer with the desired configuration.
    % See the configure method for available options
    function self=optimizer(varargin)

      pkg load statistics;

      self.configure(varargin{:});

    endfunction

    % Configure the optimizer
    %
    % The following parameter pairs are possible (if ommited, the current value
    % is kept):
    % "method", (string): Use "batch","sgd", "momentum", "rmsprop", "adam"
    % "alpha", (float): learning rate (default: 0.05)
    % "beta1", (float): beta1 parameter for momenum (default: 0.7)
    % "beta2",(float): beta2 parameter for adam (default: 0.99)
    % "maxiter",(int): maximum number of iterations (default: 200)
    % "epsilon",(float): tolerance error for convergence (default: 0.001)
    % "minibatch",(int): size of minibatch (default: 16)
    % "mbmode", (strint): Use "withrep","norep" (default: "withrep")
    % "show", (string): Use "nothing","dots","loss","progress"
    function configure(self,varargin)

      parser = inputParser();

      validMethods={"batch","sgd","momentum","rmsprop","adam"};
      checkMethod = @(x) any(validatestring(x,validMethods));
      addParameter(parser,'method',self.method,checkMethod);

      checkBeta = @(x) isreal(x) && isscalar(x) && x>=0 && x<=1;
      checkRealPosScalar = @(x) isreal(x) && isscalar(x) && x>0;

      addParameter(parser,'alpha',self.alpha,checkBeta);
      addParameter(parser,'beta1',self.beta1,checkBeta);
      addParameter(parser,'beta2',self.beta2,checkBeta);
      addParameter(parser,'maxiter',self.maxiter,checkRealPosScalar);
      addParameter(parser,'epsilon',self.epsilon,checkRealPosScalar);
      addParameter(parser,'minibatch',self.minibatch,checkRealPosScalar);


      validMBMode={"withrep","norep"};
      checkMBMode=@(x) any(validatestring(x,validMBMode));
      addParameter(parser,"mbmode",self.mbmode,checkMBMode);

      validShow={"nothing","dots","loss","progress"};
      checkShow=@(x) any(validatestring(x,validShow));
      addParameter(parser,'show',self.show,checkShow);

      parse(parser,varargin{:});

      self.method    = parser.Results.method;    ## String with desired method
      self.alpha     = parser.Results.alpha;     ## Learning rate
      self.beta1     = parser.Results.beta1;     ## Momentum parameters beta1
      self.beta2     = parser.Results.beta2;     ## ADAM paramter beta2
      self.maxiter   = parser.Results.maxiter;   ## maxinum number of iterations
      self.epsilon   = parser.Results.epsilon;   ## convergence error tolerance
      self.minibatch = parser.Results.minibatch; ## minibatch size
      self.mbmode    = parser.Results.mbmode;    ## minibatch replacement mode
      self.show      = parser.Results.show;      ## show progress information
    endfunction

    % Find the minimum of a function J.
    %
    % Starting at point theta0, search for the minimum of the target
    % function J by iteratively applying a gradient descent process, where
    % the gradient of the target function is given by gJ.
    %
    % The training input data is expected in the design matrix Xo and
    % the training data labels are expected in Yo, one-hot encoded.
    %
    % The target function J and its gradient gJ must follow the interface
    %
    %   function loss = J(theta,Xo,Yo)
    %
    % and
    %
    %   function grad = gJ(theta,Xo,Yo)
    %
    % where theta is a vector holding a set of parameters for the used
    % hypothesis, which might be a row vector (as in logistic regression) or
    % a matrix (as in softmax).
    %
    % The loss function J must return a scalar value, and the gradient
    % gJ a vector with  the same dimensions as theta.
    %
    % The objetive of this minimize function is to find the values of theta
    % that produce the minimum value of J.
    %
    % The following parameters are required:
    %
    % J: target function computing the loss (or error)
    % gJ: gradient of the target function J
    % theta0: initial point for the iterative optimization process
    % Xo: vector holding the training design matrix
    % Yo: vector holding the training labels, one-hot encoded.
    %
    %
    % The function should return all intermediate theta values in pos, as
    % well as the corresponding error in each of those positions.
    %
    % Example:
    %   [pos,errors]=optimizer(@loss,@gradloss,[0 0 0.5],X,Y,0.1,"method","adam","maxIter",10)
    function [pos,errors]=minimize(self,J,gJ,theta0,Xo,Yo)

      ## theta0 must be a row vector
      if (isvector(theta0))
        theta0=theta0(:).';
      endif

      ## Xo must have the same number of rows as Yo
      if (rows(Xo)!=rows(Yo))
        error("Xo must be the same number of rows as Yo");
      endif

      ## Perform the gradient descent
      ts={}; ## initialize cell array
      ts{1}=theta0; # sequence of t's


      ## The samplers are functions used to get some/all samples from
      ## the design matrix.  "samplerB" is used for batch training
      ## and it simply returns the whole set, while "samplerMB" is
      ## used to randomly peek a subset of samples used in minibatch
      ## training.
      ##
      ## Depending on the minibatch mode (mbmode) the subset returned
      ## by samplerMB uses sampling with-replacement or
      ## without-replacement.

      ## batch sampler, just passes through the indices of the whole input set
      samplerB = @(X) [1:rows(X)]';

      samplerMB=[];

      switch(self.mbmode)
        case "withrep"
          ## "With-replacement" means that the random samples can appear
          ## several times, since once taken, they are placed back into the
          ## whole set.
          samplerMB = @(X) round(unifrnd(1,rows(X),self.minibatch,1));
        case "norep"
          ## "Without-replacement" means that the samples are unique,
          ## because once taken, they are not returned back to the set.
          samplerMB = @(X) self.samplerMBnr(X);
        otherwise
          error("Minibatch mode unknown");
      endswitch


      switch (self.method)
        case "momentum"
          sampler=samplerMB;
          updater=@(tc,g) self.updateMomentum(tc,g);
        case "batch"
          sampler=samplerB;
          updater=@(tc,g) self.update(tc,g);
        case "sgd"
          sampler=samplerMB;
          updater=@(tc,g) self.update(tc,g);
        case "rmsprop"
          sampler=samplerMB;
          updater=@(tc,g) self.update(tc,g);
        case "adam"
          sampler=samplerMB;
          updater=@(tc,g) self.update(tc,g);
        otherwise
          error("Method not implemented yet");
      endswitch

      progress = [];

      switch (self.show)
        case "nothing"
          progress = @(it,err) self.showNothing(it,err);
        case "dots"
          progress = @(it,err) self.showDots(it,err);
        case "loss"
          progress = @(it,err) self.showLoss(it,err);
        case "progress"
          progress = @(it,err) self.showProgress(it,err);
        otherwise
          error("Unknown show method");
      endswitch

      # Initialization of the gradients for momentum, rmsprop and adam

      sample=sampler(Xo); # which rows of Xo should be used?
      X=Xo(sample,:);
      Y=Yo(sample,:);
      errors = [J(theta0,X,Y)];
      self.v = gJ(theta0,X,Y);  # Grad @ current position

      for i=[1:self.maxiter] # max iterations
        tc = ts{end}; # Current position

        sample=sampler(Xo); # Use MB random samples
        X=Xo(sample,:);
        Y=Yo(sample,:);
        gn = gJ(tc,X,Y);  # Grad @ current position
        tn = updater(tc,gn); # Next position

        ts{end+1} = tn;
        errors = [errors;J(tn,X,Y)];

        progress(i,errors(end));
      endfor
      printf("\n");
      pos = ts;


    endfunction

  endmethods
endclassdef


