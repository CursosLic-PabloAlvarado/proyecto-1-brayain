## Copyright (C) 2021-2023 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5857 Aprendizaje Automático
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## "Capa" de entrada a la red neuronal
##
## La función de esta capa es indicar en un modelo la dimensión de los
## datos de entrada esperados.  
## La capa no realiza ningún cálculo, y ni tan siquiera es almacenada como tal.
classdef input_layer < handle
  properties
    ## Dimensión de la entrada
    units=0;
  endproperties

  methods
    ## Constructor crea un objeto vacío
    function self=input_layer(units)
      self.units=units;
    endfunction    
    
    ## Propagación hacia adelante no se usa en esta capa.  
    ## Pasa los datos de entrada
    function y=forward(self,X)
      y=X
    endfunction

    ## Propagación hacia atrás recibe dJ/ds de siguientes nodos
    function g=backward(self,dJds)
      g=dJds;
    endfunction
  endmethods
endclassdef
