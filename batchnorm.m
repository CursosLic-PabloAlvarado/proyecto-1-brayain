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
      
    endfunction

    ## Inicializa el estado de la capa (p.ej. los pesos si los hay)
    ##
    ## La función devuelve la dimensión de la salida de la capa y recibe
    ## la dimensión de los datos a la entrada de la capa
    function outSize=init(self,inputSize)
      outSize=inputSize;
      
      ## TODO: 
      
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
    function y=forward(self,X,prediction=false)
      
      if (prediction)
        
        ## TODO: Qué hacer en la predicción?
        y=X; ## BORRAR esta línea cuando tenga la verdadera solución
        
      else
        if (rows(X)==1)
          ## Imposible normalizar un solo dato.  Devuélvalo tal y como es
          y=X;          
        else
          ## TODO: Qué hacer en el entrenamiento?
          y=X; ## BORRAR esta línea cuando tenga la verdadera solución
      
        endif
      endif
    endfunction

    ## Propagación hacia atrás recibe dJ/ds de siguientes nodos del grafo,
    ## y retorna el gradiente necesario para la retropropagación. que será
    ## pasado a nodos anteriores en el grafo.
    function g=backward(self,dJds)      
      g=dJds; ## TODO: CORREGIR, pues esto no es el verdadero gradiente
    endfunction
  endmethods
endclassdef
