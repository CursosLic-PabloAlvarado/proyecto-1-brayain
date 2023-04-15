## Copyright (C) 2021-2023 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5857 Aprendizaje Automático
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Normalización por lotes
classdef batchnorm < handle
  properties
    ## TODO: Agregue las propiedades que requiera.  No olvide inicializarlas
    ##       en el constructor o el método init si hace falta.
    alpha % parametro de escalamiento
    moving_mean % moving average of the mean
    moving_var % moving average of the varianc
    momentum % momentum para promedio
    inputsX=[]; ## Entrada de valores en la propagación hacia adelante
    x_norm=[]; % normalized input data
    mu % batch mean
    sigma % batch variance

    ## Número de unidades (neuronas) en la capa
    units=0;

    ## Resultados después de la propagación hacia adelante
    outputs=[];

    ## Resultados después de la propagación hacia atrás
    gradientX=[];

    ## Parámetro usado por el filtro que estima la varianza y media completas
    beta=0.9;

    ## Valor usado para evitar divisiones por cero
    epsilon=1e-10;

  endproperties

  methods
    ## Constructor
    ##
    ## beta es factor del filtro utilizado para aprender
    ## epsilon es el valor usado para evitar divisiones por cero

    function self=batchnorm(beta=0.9,epsilon=1e-10,alpha,momentum)
      self.beta=beta;
      self.epsilon=epsilon;

      ## TODO:

      self.alpha = alpha;
      self.momentum = momentum;
      self.moving_mu = 0;%ruido
      self.moving_sigma = 0;
      self.gradientX=[];
    endfunction

    ## Inicializa el estado de la capa (p.ej. los pesos si los hay)
    ##
    ## La función devuelve la dimensión de la salida de la capa y recibe
    ## la dimensión de los datos a la entrada de la capa
    function outSize=init(self,inputSize)
      outSize=inputSize;

      ## TODO:

    endfunction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## La capa de normalización no tiene estado que se aprenda con
    ## la optimización.
    function st=hasState(self)
      st=false;
    endfunction

    ## Propagación hacia adelante normaliza por media del minilote
    ## en el entrenamiento, pero por la media total en la predicción.
    ##
    ## El parámetro 'prediction' permite determinar si este método
    ## está siendo llamado en el proceso de entrenamiento (false) o en el
    ## proceso de predicción (true)

    function y=forward(self,X,prediction=false)
      self.inputsX = inputsX;
      m = size(inputsX,1);
      n = size(inputsX,2);
      if (prediction)

        ## TODO: Qué hacer en la predicción? %se usa todos los pasos

        self.x_norm = (inputsX-self.moving_mu)./sqrt(self.moving_sigma+self.epsilon);
        out = self.alpha.*self.x_norm + self.beta;
        y=out; %respuesta
      else
        if (rows(X)==1)
          ## Imposible normalizar un solo dato.  Devuélvalo tal y como es
          y=X;
        else
          ## TODO: Qué hacer en el entrenamiento?% utilizar solamente minilote
          self.mu = mean(inputsX);
          self.sigma = var(inputsX)*(m-1)/m;
          self.x_norm = (inputsX-self.mu)./sqrt(self.sigma+self.epsilon);
          out = self.alpha.*self.x_norm + self.beta;
          % update moving averages of mean and variance
          self.moving_mu = self.momentum*self.moving_mu + (1-self.momentum)*self.mu;
          self.moving_sigma = self.momentum*self.moving_sigma + (1-self.momentum)*self.sigma;

          y=out;
##############################################################################
        endif
        cache.x_norm = self.x_norm;
        cache.alpha = self.alpha;
        cache.sigma = self.sigma;
        cache.mu = self.mu;
        cache.inputsX = inputsX;
        cache.epsilon = self.epsilon;
      endif
    endfunction

    ## Propagación hacia atrás recibe dJ/ds de siguientes nodos del grafo,
    ## y retorna el gradiente necesario para la retropropagación. que será
    ## pasado a nodos anteriores en el grafo.
    function g=backward(self,dJds,cache)

##
##      dx_norm = dout.*cache.alpha;
##      x_mu = cache.inputsX - cache.mu;
##      var_sqrt_inv = 1./sqrt(cache.sigma + cache.epsilon);
##      dvar = sum(dx_norm.*x_mu.*(-0.5).*var_sqrt_inv.^3, 1);
##      dmu = sum(dx_norm.*-var_sqrt_inv, 1) + dvar.*mean(-2.*x_mu, 1);
##      dx = dx_norm.*var_sqrt_inv + dvar.*(2./size(self.inputs	,1).*x_mu) + dmu./size(self.inputs	,1);
##      self.alpha = sum(dout.*cache.x_norm, 1);
##      self.beta = sum(dout, 1);
      g=dJds./sqrt(cache.sigma + cache.epsilon);
      self.gradientX=g;
      %g=dx;

    endfunction
  endmethods
endclassdef
