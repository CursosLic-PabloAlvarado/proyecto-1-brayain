## Copyright (C) 2021-2023 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5857 Aprendizaje Automático
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Capa ejemplo
##
## Este código es un ejemplo de implementación de una capa, con todos los métodos
## e interfaces que deben respetarse.
classdef lelelu < handle

  ## En GNU/Octave "< handle" indica que la clase se deriva de handle
  ## lo que evita que cada vez que se llame un método se cree un
  ## objeto nuevo.  Es decir, en esta clase forward y backward alternan
  ## la instancia actual y no una copia, como sería el caso si no
  ## se usara "handle".

  properties
    ## Ejemplo Número de unidades (neuronas) en la capa
    ## (solo un ejemplo, puede borrarse)
    units = 0;
    W = []; ## matriz de pesos
    b = []; ## vector de bias
    a = 0.1; ## alpha para la Leaky ReLU

  endproperties

  methods
    ## Constructor inicializa todo vacío
    function self=lelelu(units)
      if (nargin > 0)
        self.units=units;
      else
        self.units=0;
      endif

      ## TODO: Inicialice sus propiedades aquí
      ## Inicializar pesos y bias aleatoriamente
      self.W = randn(self.units, inputSize) * 0.01;
      self.b = zeros(self.units, 1);
    endfunction

    ## Inicializa el estado de la capa (p.ej. los pesos si los hay)
    ##
    ## La función devuelve la dimensión de la salida de la capa y recibe
    ## la dimensión de los datos a la entrada de la capa
    function outSize=init(self,inputSize)
      self.W = randn(self.units, inputSize) * 0.01;
      self.b = zeros(self.units, 1);
      outSize = self.units;
    endfunction

    ## Retorna true si la capa tiene un estado que adaptar (como pesos).
    ##
    ## En ese caso, es necesario tener las funciones stateGradient(),
    ## state() y setState()
    function st=hasState(self)
      st=true;
    endfunction


    ## Si hasState() retorna false, las siguientes tres funciones
    ## pueden borrarse:

    ## Retorne el gradiente del estado, que existe solo si esta capa tiene
    ## algún estado que debe ser aprendido
    ##
    ## Este gradiente es utilizado por el modelo para actualizar el estado
    ##
    ## Si la capa no tiene estado que actualizar (como pesos), y si hasState()
    ## returna false, entonces puede eliminarse este método.
    function g=stateGradient(self)
      g = [self.W(:); self.b(:)];
    endfunction

    ## Retorne el estado aprendido
    ##
    ## Si la capa no tiene estado que actualizar (como pesos), y si hasState()
    ## returna false, entonces puede eliminarse este método.
    function st=state(self)
      s = [self.W(:); self.b(:)];
    endfunction

    ## Reescriba el estado aprendido
    ##
    ## Si la capa no tiene estado que actualizar (como pesos), y si hasState()
    ## returna false, entonces puede eliminarse este método.
    function setState(self,W)
      W_size = self.units * size(state, 1) / (size(state, 1) + 1);
      b_size = self.units - W_size;

      self.W = reshape(state(1:W_size), self.units, []);
      self.b = reshape(state(W_size+1:end), b_size, []);
    endfunction

    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



    ## Propagación hacia adelante realiza W*x
    ## X puede ser un vector columna o una matriz.
    ##
    ## Si X es un vector columna es interpretado como un dato.  Si X
    ## es una matriz, se asume que es una matriz de diseño convencional,
    ## con cada dato en una fila.
    ##
    ## El parámetro 'prediction' permite determinar si este método
    ## está siendo llamado en el proceso de entrenamiento (false) o en el
    ## proceso de predicción (true)
    function y=forward(self,X,prediction=false)
      z = self.W * X + self.b;
      y = max(self.a * z, z);
    endfunction

    ## Propagación hacia atrás recibe dL/ds de siguientes nodos del grafo,
    ## y retorna el gradiente necesario para la retropropagación. que será
    ## pasado a nodos anteriores en el grafo.
    function g=backward(self,dJds)
      dJdz = dJdy .* (self.W >= 0) + self.a * dJdy .* (self.W < 0);
      g = {dJdz, self.W * dJdz};
    endfunction

    function [dJdW, dJdb] = backwardState(self, dJds)
      dJdW = dJds{1} * self.W';
      dJdb = sum(dJds{1})';
    endfunction

    function [dJdX, dJdW, dJdb] = backwardParams(self, X, dJds)
      dJdX = dJds{2} * self.W;
      dJdW = X' * dJds{2};
      dJdb = sum(dJds{2})';
    endfunction
  endmethods
endclassdef
