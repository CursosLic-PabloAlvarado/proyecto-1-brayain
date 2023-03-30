## Copyright (C) 2021-2023 Pablo Alvarado
##
## Este archivo forma parte del material del Proyecto 1 del curso:
## EL5857 Aprendizaje Automático
## Escuela de Ingeniería Electrónica
## Tecnológico de Costa Rica

## Ejemplo de configuración de red neuronal y su entrenamiento

1;

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

ann=sequential("maxiter",1500,
               "alpha",0.1,
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
           dense_unbiased(16),
           sigmoid(),
           dense_unbiased(16),
           sigmoid(),
           dense_unbiased(numClasses),
           sigmoid()});

  #ann.add(input_layer(2));
  #ann.add(dense_unbiased(16));
  #ann.add(sigmoid());
  #ann.add(dense_unbiased(16));
  #ann.add(sigmoid());
  #ann.add(dense_unbiased(numClasses));
  #ann.add(sigmoid());


  ann.add(olsloss());
endif

loss=ann.train(X,Y,vX,vY);
ann.save(file);

## TODO: falta agregar el resto de pruebas y visualizaciones
