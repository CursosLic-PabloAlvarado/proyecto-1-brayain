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
    output=[];
    gradient=[];
    soft=1;
  endproperties

 methods
    ## Constructor ejecuta un forward si se le pasan datos
    function self=softmax()
      self.outputs=[];
      self.gradient=[];
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
      val= exp(X);
      n=sum(val)-ones(1,columns(val));
      self.output = val ./ n;
      y=self.output;
    endfunction

    ## Propagación hacia atrás recibe dL/ds de siguientes nodos del grafo,
    ## y retorna el gradiente necesario para la retropropagación. que será
    ## pasado a nodos anteriores en el grafo.
    function g=backward(self,dJds)
      localGrad = self.output.*(1-self.output);
      self.gradient = localGrad.*dJds;
      g=self.gradient;
    endfunction
  endmethods
endclassdef
