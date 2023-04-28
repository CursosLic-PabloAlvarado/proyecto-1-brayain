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
classdef softmax < handle

  ## En GNU/Octave "< handle" indica que la clase se deriva de handle
  ## lo que evita que cada vez que se llame un método se cree un
  ## objeto nuevo.  Es decir, en esta clase forward y backward alternan
  ## la instancia actual y no una copia, como sería el caso si no
  ## se usara "handle".

  properties
    units=0;
    outputs=[];
    gradient=[];
    inputsX=[];
    input_shape;
    softmax;
  endproperties

 methods
    ## Constructor ejecuta un forward si se le pasan datos
    function self=softmax(input_shape)
##      if nargin > 0
##          self.dim = dim;
##      else
##          self.dim = 1;
##      end
      self.outputs=[];
      self.gradient=[];
      self.inputsX=[];
      self.input_shape = input_shape;

    endfunction

    ## En funciones de activación el init no hace mayor cosa más que
    ## indicar que la dimensión de la salida es la misma que la entrada.
    ##
    ## La función devuelve la dimensión de la salida de la capa
    function outSize=init(self,inputSize)
      outSize=inputSize;
    endfunction

    ## Retorna false si la capa no tiene un estado que adaptar
    function st=hasState(self)
      st=false;
    endfunction

    function y=forward(self,X,prediction=false)

      self.inputsX=X;
      exp_input = exp(X);
      % Calculate the sum of the exponential values across the rows
      row_sums = sum(exp_input, 2);
      % Reshape the row_sums to match the input_data shape
      row_sums = reshape(row_sums, [], 1);
      % Calculate the output of the softmax layer
      self.outputs = exp_input ./ row_sums;
      y=self.outputs;
    endfunction

    ## Propagación hacia atrás recibe dL/ds de siguientes nodos del grafo,
    ## y retorna el gradiente necesario para la retropropagación. que será
    ## pasado a nodos anteriores en el grafo.
    function g=backward(self,dJds) %OBTENIDO DE CHATGPT

    softmax = self.forward(self.inputsX);
    num_classes = self.input_shape(1);
    jacobian = zeros(num_classes, num_classes);
    for i = 1:num_classes
        for j = 1:num_classes
            if i == j
                jacobian(i, j) = softmax(i) * (1 - softmax(i));
            else
                jacobian(i, j) = -softmax(i) * softmax(j);
            end
        end
    end
    % Calculate the gradients of the loss function with respect to the input
    gradients =  dJds*jacobian;
    g=gradients;
    endfunction
  endmethods
endclassdef
