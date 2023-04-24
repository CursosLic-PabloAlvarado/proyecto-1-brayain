% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Su Copyright AQUÍ>

% Loss function used in softmax
function err=softmax_loss(theta,X,y)

  m = length(y); % número de ejemplos de entrenamiento
  h = softmax_hyp(theta,X); % calcular la hipótesis
  err = sum((1 / (m)) * sum((h - y) .^ 2)); % calcular el error de MSE
endfunction
