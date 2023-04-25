## Copyright (C) 2021-2023 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5857 Aprendizaje Automático
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Ejemplo de configuración de red neuronal Y su entrenamiento

1;
clear classes;
warning('off','Octave:shadowed-function');
pkg load statistics;

numClasses=5;

##datashape='spirals';
##datashape='curved';
##datashape='voronoi';
##datashape='vertical';
datashape='pie';

[oX,oY]=create_data(numClasses*100,numClasses,datashape); ## Training

## Partition created data into training (60%) and test (40%) sets
idx=randperm(rows(oX));

tap=round(0.6*rows(oX));
idxTrain=idx(1:tap);
idxTest=idx(tap+1:end);

X = oX(idxTrain,:);
Y = oY(idxTrain,:);

vX = oX(idxTest,:);
vY = oY(idxTest,:);
figure(1,"name","Datos de entrenamiento");
hold off;
plot_data(X,Y);

ann=sequential("maxiter",3000,
               "alpha",0.05,
               "beta2",0.99,
               "beta1",0.9,
               "minibatch",32,
               "method","momentum",
               "show","loss");

file="ann.dat";

reuseNetwork = false;

if (reuseNetwork && exist(file,"file")==2)
  ann.load(file);
else
  ann.add({input_layer(2),
           dense(16),
           prelu(),
           dense(16),
           prelu(),
           dense(numClasses),
           prelu()});

  #ann.add(input_layer(2));
  #ann.add(dense(16));
  #ann.add(sigmoid());
  #ann.add(dense(16));
  #ann.add(sigmoid());
  #ann.add(dense(numClasses));
  #ann.add(sigmoid());


  ann.add(olsloss());
endif

loss=ann.train(X,Y,vX,vY);
ann.save(file);

## TODO: falta agregar el resto de pruebas Y visualizaciones


############################
figure(2);
   hold off;
   plot(loss(:,1),";training;")

   hold on;
   plot(loss(:,2),";validation;")

   xlabel('Iteration')
   ylabel('Error')
   title('Error vs. Iteration')
   hold on;

############################

function [confusionMatrix] = calculateConfusionMatrix(actualLabels, predictedLabels, numClasses)
% Función que calcula la matriz de confusión para un conjunto de etiquetas de
% verdad de referencia Y etiquetas predichas.

% confusionMatrix es la matriz de confusión de tamaño (numClasses, numClasses)

confusionMatrix = zeros(numClasses, numClasses);

for i = 1:numClasses
    for j = 1:numClasses
        % calcular la cantidad de instancias que pertenecen a la clase i Y que han sido clasificadas como clase j
        confusionMatrix(i, j) = sum(actualLabels(:, i) & predictedLabels(:, j));
    end
end

end


function [accuracy, precision, recall] = evalresults(actualLabels, predictedLabels, numClasses)

% accuracy: exactitud de la clasificación
% precision: precisión de la clasificación
% recall: exhaustividad de la clasificación

confusionMatrix = calculateConfusionMatrix(actualLabels, predictedLabels, numClasses);




% calcular exactitud
accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));

% calcular precisión Y exhaustividad por clase
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);

for i = 1:numClasses
    precision(i) = confusionMatrix(i, i) / sum(confusionMatrix(:, i));
    recall(i) = confusionMatrix(i, i) / sum(confusionMatrix(i, :));
end



fprintf('Resultados de clasificacion de la matriz de confusion \v');

fprintf('Clase \t Precision \t Exhaustividad \t Exactitud\n');
for i = 1:5
    fprintf('%d \t %f \t %f \t %f\n', i, precision(i), recall(i), accuracy);
end



end

%Llamar a la función
evalresults(Y(1:200,:), vY, numClasses);


softmaxvisualizer(X,Y)
