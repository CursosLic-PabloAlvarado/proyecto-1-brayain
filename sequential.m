## Copyright (C) 2021-2023 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5857 Aprendizaje Automático
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Modelo secuencial
##
## Esta clase encapsula una red neuronal hacia adelante, con métodos para
## agregar capas, almacenar, cargar, entrenar y predecir.
##
## El método add() permite agregar capas al modelo.  La primera capa debe
## ser del tipo "input_layer" para indicar la dimensión de los datos de entrada.
##
## Luego se agregan capas de normalización, combinación, activación.
##
## La última capa de la red debe ser una capa de error o pérdida.
classdef sequential < handle

  properties ( Access = private )
    ## Constantes:
    layers={};

    ## Training parameters
    maxiter=2000;
    minibatch=128;

    alpha=0.01;     ## Learning rate
    dalpha=0.005;   ## Decay
    beta1 = 0.95    ## Momentum: 0 para no usar momentum
    beta2 = 0.99    ## Polo de filtro de cuadrados (0: no usar Adam))
    epsilon = 1e-9; ## Evite divisiones por cero en Adam

    method = 'adam';  ## "batch", "sgd", "momentum", "rmsprop", "adam", "autoclip"
    use_decay=false;
    mbmode = 'withrep';  ## Minibatch mode with replacement
    show   = 'progress';
    secondMoments;
    t;
    ## Remaining samples used while training with no-replacement
    remainingIndices=[];

    ## Last output dimension of a layer while adding layers
    lastOutput = -1;

    ## States for each layer, which depend on the optimization method.
    filteredGradients={}

  endproperties

  methods (Access = private)


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

    ## ----------------------------------------------------------------------
    ## Updaters
    ##
    ## Each updater is used to update the theta parameter vector/matrix
    ## It needs the current parameters tc, and the current gradient g
    ## and it returns the updated parameters vector tn depending on the
    ## method.

    ## State update for batch and SGD
    function newState = updateSimple(self,layerIdx,
                                     currentState,
                                     stateGradient)
      newState = currentState - self.alpha*stateGradient;
    endfunction


    ## State update with momentum
    function newState = updateMomentum(self,layerIdx,
                                       currentState,
                                       stateGradient)

      if ((layerIdx>length(self.filteredGradients)) ||
          isempty(self.filteredGradients{layerIdx}))
        ## Momentum needs an initial filtered gradient.  If not
        ## available yet, it means this is the first time we pass by,
        ## and we must provide a meaningful value
        self.filteredGradients{layerIdx}=stateGradient;
      endif

      self.filteredGradients{layerIdx} = ...
        self.beta1*self.filteredGradients{layerIdx} + ...
        (1-self.beta1)*stateGradient;

      newState = currentState - self.alpha*self.filteredGradients{layerIdx};
    endfunction

    ## State update for rmsprop
    function newState = updatermsprop(self, layerIdx, currentState, stateGradient)
      if ((layerIdx > length(self.filteredGradients)) || isempty(self.filteredGradients{layerIdx}))
        self.filteredGradients{layerIdx} = stateGradient .^ 2;
      else
        self.filteredGradients{layerIdx} = self.beta2 * self.filteredGradients{layerIdx} + (1 - self.beta2) * (stateGradient .^ 2);
      endif
      newState = currentState - (self.alpha ./ (sqrt(self.filteredGradients{layerIdx}) + self.epsilon)) .* stateGradient;
    endfunction

    ## State update for adam
    function newState = updateadam(self, layerIdx, currentState, stateGradient)
      if ((layerIdx > length(self.filteredGradients)) || isempty(self.filteredGradients{layerIdx}))
        self.filteredGradients{layerIdx} = stateGradient;
        self.secondMoments{layerIdx} = stateGradient .^ 2;
      else
        self.filteredGradients{layerIdx} = self.beta1 * self.filteredGradients{layerIdx} + (1 - self.beta1) * stateGradient;
        self.secondMoments{layerIdx} = self.beta2 * self.secondMoments{layerIdx} + (1 - self.beta2) * (stateGradient .^ 2);
      endif
      mHat = self.filteredGradients{layerIdx} ./ (1 - self.beta1 ^ (self.t));
      vHat = self.secondMoments{layerIdx} ./ (1 - self.beta2 ^ (self.t));
      newState = currentState - (self.alpha ./ (sqrt(vHat) + self.epsilon)) .* mHat;
      self.t += 1;
    endfunction

    ## State update for autoclip
    function newState = updateautoclip(self, layerIdx, currentState, stateGradient)
      if ((layerIdx > length(self.filteredGradients)) || isempty(self.filteredGradients{layerIdx}))
        self.filteredGradients{layerIdx} = stateGradient;
        self.secondMoments{layerIdx} = stateGradient .^ 2;
        clipValue = norm(stateGradient);
      else
        clipValue = max(norm(stateGradient), norm(self.filteredGradients{layerIdx}));
        self.filteredGradients{layerIdx} = stateGradient;
        self.secondMoments{layerIdx} = stateGradient .^ 2;
      endif
      clippedStateGradient = min(clipValue, norm(stateGradient)) * stateGradient / norm(stateGradient);
      newState = currentState - self.alpha * clippedStateGradient;
    endfunction

    ## State update for radam
    function newState = updateradam(self, layerIdx, currentState, stateGradient)
      if ((layerIdx > length(self.filteredGradients)) || isempty(self.filteredGradients{layerIdx}))
        self.filteredGradients{layerIdx} = stateGradient;
        self.secondMoments{layerIdx} = stateGradient .^ 2;
      else
        self.filteredGradients{layerIdx} = self.beta1 * self.filteredGradients{layerIdx} + (1 - self.beta1) * stateGradient;
        self.secondMoments{layerIdx} = self.beta2 * self.secondMoments{layerIdx} + (1 - self.beta2) * (stateGradient .^ 2);
      endif
      mHat = self.filteredGradients{layerIdx} ./ (1 - self.beta1 ^ (self.t));
      vHat = self.secondMoments{layerIdx} ./ (1 - self.beta2 ^ (self.t));

      pInf = 2 / (1 - self.beta2) - 1;
      pt = pInf - 2 * self.t * (self.beta2 ^ self.t) / (1 - self.beta2 ^ self.t);

      if pt > 4
        r = sqrt(((pt - 4) * (pt - 2) * pInf) / ((pInf - 4) * (pInf - 2) * pt));
        newState = currentState - (self.alpha * r ./ (sqrt(vHat) + self.epsilon)) .* mHat;
      else
        newState = currentState - self.alpha * mHat;
      endif
      self.t += 1;
    endfunction

  endmethods

  methods (Access = public)

    % Construct a sequential model with the desired configuration.
    % See the configure method for available options
    function self=sequential(varargin)

      warning('off','Octave:shadowed-function');
      pkg load statistics;

      self.configure(varargin{:});

      layers={};
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

      validMethods={"batch","sgd","momentum","rmsprop","adam","autoclip","radam"};
      checkMethod = @(x) any(validatestring(x,validMethods));
      addParameter(parser,'method',self.method,checkMethod);
      addParameter(parser,'use_decay',self.use_decay);
      checkBeta = @(x) isreal(x) && isscalar(x) && x>=0 && x<=1;
      checkRealPosScalar = @(x) isreal(x) && isscalar(x) && x>0;

      addParameter(parser,'alpha',self.alpha,checkBeta);
      addParameter(parser,'dalpha',self.dalpha,checkBeta);
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
      self.dalpha     = parser.Results.dalpha;   ## Decay rate
      self.beta1     = parser.Results.beta1;     ## Momentum parameters beta1
      self.beta2     = parser.Results.beta2;     ## ADAM paramter beta2
      self.maxiter   = parser.Results.maxiter;   ## maxinum number of iterations
      self.use_decay = parser.Results.use_decay;
      self.epsilon   = parser.Results.epsilon;   ## convergence error tolerance
      self.minibatch = parser.Results.minibatch; ## minibatch size
      self.mbmode    = parser.Results.mbmode;    ## minibatch replacement mode
      self.show      = parser.Results.show;      ## show progress information
      self.secondMoments      = {};
      self.t = 1;
    endfunction



    function add(self,layer)
      ## Agregue una capa al modelo secuencial
      ## La primera capa debe ser una capa del tipo "input_layer" para así
      ## indicar la dimensión esperada de cada dato de entrada
      ## La última capa debe ser una capa con la función de pérdida/error.
      ##
      ## Esa capa será ignorada en la predicción.
      if (isa(layer,"cell"))
        add(self,layer{1}); ## Llame de nuevo la función con solo una capa
        if (length(layer)>1)
          add(self,{layer{2:end}}); ## y luego con el resto
        endif
      elseif (isa(layer,"input_layer"))
        self.lastOutput=layer.units;
        printf("Input layer configured with dimension %i\n",self.lastOutput);
      else
        if (self.lastOutput>0)
          printf("Agregando capa '%s'(%i -> ",class(layer),self.lastOutput);
          self.layers = {self.layers{:},layer};
          self.lastOutput=self.layers{end}.init(self.lastOutput);
          printf("%i)\n",self.lastOutput);
        else
          error("Debe agregar primero una capa de entrada");
        endif
      endif
    endfunction

    function losslog=train(self,X,Y,valSetX=[],valSetY=[])
      ## Entrene el modelo
      ## X: matriz de diseño (datos de entrenamiento en filas)
      ## y: matriz de salida, cada file codificada one-hot
      ## valSetX: set de validación (opcional) (entradas en filas)
      ## valSetY: set de validación (opcional) (salidas en filas)
      ## losslog: protocolo con loss por época, para set de
      ##          entrenamiento y opcionalmente el set de validación

      ## Number of layers
      numLayers = length(self.layers);

      if (numLayers<1)
        error("No network structure configured yet.  Layers need to be added first.\n");
      endif

      ## Set up the progress information
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


      ## The samplers are functions to get some/all samples from
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
        case "batch"
          sampler=samplerB;
          updater=@(li,tc,g) self.updateSimple(li,tc,g);
        case "momentum"
          sampler=samplerMB;
          updater=@(li,tc,g) self.updateMomentum(li,tc,g);
        case "sgd"
          sampler=samplerMB;
          updater=@(li,tc,g) self.updateSimple(li,tc,g);
        case "rmsprop"
          sampler=samplerMB;
          updater=@(li,tc,g) self.updatermsprop(li,tc,g);
        case "adam"
          sampler=samplerMB;
          updater=@(li,tc,g) self.updateadam(li,tc,g);
        case "autoclip"
          sampler=samplerMB;
          updater=@(li,tc,g) self.updateautoclip(li,tc,g);
        case "radam"
          sampler=samplerMB;
          updater=@(li,tc,g) self.updateradam(li,tc,g);
        otherwise
          error("Method not implemented yet");
      endswitch


      ## Initialize loss history tracking
      losslog=[];
      loss=0;
      samplesProcessed=0;

      ## Iterate on minibatches
      for ep=1:self.maxiter

        idx=sampler(X); # Which sample indices to use next

        subX=X(idx,:);
        subY=Y(idx,:);
        lr = learning_rate_decay(self.alpha, ep, self.dalpha, self.use_decay);
        self.alpha=lr;
        ## Forward propagation
        y=self.layers{1}.forward(subX);
        for l=2:numLayers-1
          y=self.layers{l}.forward(y);
        endfor
        currentLoss=self.layers{numLayers}.forward(y,subY);
        loss+=currentLoss;

        ## Back propagation
        g=self.layers{numLayers}.backward(1);
        for l=numLayers-1:-1:1
          g=self.layers{l}.backward(g);

          ## If the layer has a state, use the updater to compute the
          ## next state
          if (self.layers{l}.hasState())

            self.layers{l}.setState(updater(l,
                                            self.layers{l}.state(),
                                            self.layers{l}.stateGradient()));
          endif
        endfor

        ## An epoch is the presentation of all samples in the training
        ## set.  We iterate a minibatch at a time (except in batch
        ## mode), so we have to check when an epoch has passed.

        samplesProcessed += rows(subX);
        if (samplesProcessed>=rows(X))
          # an epoch has passed
          if (isempty(valSetX))
            ## No validation data available: just store thre training loss
            losslog = vertcat(losslog,[loss]);
          else
            ## Compute validation loss, and store both: training and validation
            [vY,vL]=computeLoss(self,valSetX,valSetY);
            losslog = vertcat(losslog,[loss vL]);
          endif

          progress(ep,losslog(end));

          samplesProcessed=0;
          loss=0;

        endif

      endfor ## for each iteration

      printf("\n");

    endfunction


    ## Predicción con modelo preentrenado
    function y=test(self,X)
      numLayers=length(self.layers);

      y=self.layers{1}.forward(X,true); % true indica que es predicción
      for l=2:numLayers-1
        y=self.layers{l}.forward(y,true); % true indica que es predicción
      endfor

    endfunction

    ## Predicción con modelo preentrenado
    function [y,loss]=computeLoss(self,vX,vY)
      numLayers=length(self.layers);

      ## Forward prop
      y=self.layers{1}.forward(vX);
      for l=2:numLayers-1
        y=self.layers{l}.forward(y);
      endfor
      loss=self.layers{numLayers}.forward(y,vY);

    endfunction


    function layer=convertStructToLayer(self,structure,layertype)
      ## Método usado para coercionar la estructura self en una clase de tipo
      ## layertype.
      ##
      ## Es necesaria para solventar el problema de que octave no puede
      ## serializar classdef aún.
      layer=eval(layertype);
      for fn=fieldnames(structure)'
        try
          layer.(fn{1}) = structure.(fn{1});
        catch
          warning("Could not copy field %s",fn{1});
        end_try_catch
      endfor
    endfunction


    function save(self,file)
      ## Guarde red en el archivo.  Posteriormente puede cargar el archivo
      ## con load()
      ##
      ## Octave convierte las classdef a struct y por tanto pierde el tipo
      ## concreto de cada capa.
      ##
      ## Como camino alterno almacenamos los nombres de los tipos primero,
      ## para luego poder recrearlos, y una vez que se tienen instancias
      ## vacías podemos convertir las estructuras almacenadas en las clases
      ## concretas.

      ## Extraemos primero los nombres de las clases en un cell-array
      ## y convertimos las capas a estructuras de octave
      names={};
      layers={};
      warning('off','Octave:classdef-to-struct');

      for i=1:length(self.layers)
        names = { names{:} , class(self.layers{i}) };
        layers = { layers{:}, struct(self.layers{i}) };
      endfor

      ## save no entiende atributos de una clase, así que necesitamos
      ## pasar los parámetros de la clase a una estructura
      param.maxiter=self.maxiter;
      param.use_decay=self.use_decay;
      param.minibatch=self.minibatch;
      param.alpha=self.alpha;
      param.dalpha=self.dalpha;
      param.beta1=self.beta1;
      param.beta2=self.beta2;
      param.epsilon=self.epsilon;
      param.method=self.method;

      save("-v7",file,"param","names","layers");
    endfunction

    function o=load(self,file)

      ## Cargue red desde el archivo almacenado con save.
      names={};
      layers={};
      param=[];

      load("-v7",file,"param","names","layers");

      if (length(names) != length(layers))
        error("Corrupted file.  Inconsistent number of stored layers and types");
        return
      endif

      for fn=fieldnames(param)'
        try
          self.(fn{1}) = param.(fn{1});
        catch
          warning("Could not copy field %s",fn{1});
        end_try_catch
      endfor

      ## De los nombres, recreemos las instancias con los tipos correctos
      for i=1:length(names)
        printf("Loading layer %s\n",names{i});
        self.layers{i}=self.convertStructToLayer(layers{i},names{i});
      endfor
    endfunction

  endmethods
endclassdef
