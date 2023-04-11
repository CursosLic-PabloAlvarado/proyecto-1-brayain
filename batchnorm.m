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

    ## Número de unidades (neuronas) en la capa
    units=0;

    ## Pesos de la capa densa sin sesgo
    W=[];

    ## Entrada de valores en la propagación hacia adelante
    inputsX=[];

    ## Resultados después de la propagación hacia adelante
    outputs=[];

    ## Resultados después de la propagación hacia atrás
    gradientW=[];
    gradientX=[];

    mu=[];
    sigma=[];

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

    function self=batchnorm(beta=0.9,epsilon=1e-10)
      self.beta=beta;
      self.epsilon=epsilon;

      ## TODO:

      if (nargin > 0)
        self.units=units;
      else
        self.units=0;
      endif

      self.mu=[];
      self.sigma=[];

      self.inputsX=[];
      self.W=[];
      self.outputs=[];

      self.gradientX=[];
      self.gradientW=[];

    endfunction

    ## Inicializa el estado de la capa (p.ej. los pesos si los hay)
    ##
    ## La función devuelve la dimensión de la salida de la capa y recibe
    ## la dimensión de los datos a la entrada de la capa
    function outSize=init(self,inputSize)
      outSize=inputSize;

      ## TODO:

      ## Dimensiones de la matriz de pesos para calcular Y=XW
      cols = self.units;
      rows = inputSize;
      ## LeCun Normal (para selu)
      self.W=normrnd(0,1/sqrt(cols),rows,cols);
    endfunction

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
    function [mu, sigma] = media_desvi(X)
        mu = mean(X, 1);
        sigma = std(X, 1);
    endfunction

    function X_n = normalizar(X, mu, sigma)
        X_n = (X - mu) ./ sigma;
    endfunction


    function y=forward(self,X,prediction=false)

      if (prediction)

        ## TODO: Qué hacer en la predicción? %se usa todos los pasos
        %y=X; ## BORRAR esta línea cuando tenga la verdadera solución
        [m, n] = size(X);
        mu = zeros(1, n);
        sigma = zeros(1, n);

        %Media y desviación estándar de cada característica en un lote de datos
        for i = 1:m
            [batch_mu, batch_sigma] = media_desvi(X(i,:));
            mu = mu + batch_mu;
            sigma = sigma + batch_sigma .^ 2;
        endfor

        mu = mu ./ m; %media total
        sigma = sqrt(sigma ./ m + epsilon);%desviacion total

        % Normalización
        X_n = zeros(size(X));
        for i = 1:m
            X_n(i,:) = normalizar(X(i,:), mu, sigma);
        endfor
        y=X_n %respuesta
      else
        if (rows(X)==1)
          ## Imposible normalizar un solo dato.  Devuélvalo tal y como es
          y=X;
        else
          ## TODO: Qué hacer en el entrenamiento?% utilizar solamente minilote
          y=X; ## BORRAR esta línea cuando tenga la verdadera solución
##############################################################################
        endif
      endif
    endfunction

    ## Propagación hacia atrás recibe dJ/ds de siguientes nodos del grafo,
    ## y retorna el gradiente necesario para la retropropagación. que será
    ## pasado a nodos anteriores en el grafo.
    function g=backward(self,dJds)
      g=dJds; ## TODO: CORREGIR, pues esto no es el verdadero gradiente
      %regla de la cadena ej

      dJ_ds = sum(dL_dout .* (X_n - mu), 1) ./ (m .* sigma);
      dL_dmu = sum(-dL_dout ./ sigma, 1) - 2 .* sum((X_n - mu) .* dL_dsig, 1) ./ m;
      dL_dX = dL_dout ./ sigma + dL_ds .* (1 ./ m + 2 .* (X_norm - mu) ./ m);
      dL_dgamma = sum(dL_dout .* X_norm, 1);
      dL_dbeta = sum(dL_dout, 1);



    endfunction
  endmethods
endclassdef
