% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Su Copyright AQUÍ>

% Gradient of the loss function used in softmax
%
% The size of the returned gradient must be equal to the size of Theta
function grad=softmax_gradloss(theta,X,y)

  m = length(y); % número de ejemplos de entrenamiento
  h = softmax_hyp(theta,X); % calcular la hipótesis
  grad = (2/m)*((X' * (h-y)(:,1:end-1))); % para que quede size(theta)

endfunction
