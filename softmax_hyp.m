% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Team brAIn>

% Hypothesis function used in softmax
% Theta: matrix, its columns are each related to one
%        particular class.
% returns the hypothesis, which has only k-1 values for each sample
%         as the last one is computed as 1 minus the sum of all the rest.
function hh=softmax_hyp(Theta,X)
  h=exp( X*Theta );
  nor=sum(h,2) + ones(1,columns(h)); ## the ones 'cause exp(0) for k
  h = h ./ nor;
  hh=[h 1-sum(h,2)]; %agregar la columna 3
endfunction
