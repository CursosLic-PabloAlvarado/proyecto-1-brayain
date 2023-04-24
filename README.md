[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10664653&assignment_repo_type=AssignmentRepo)
# Proyecto 1

S1 2023<br/>
EL5857 Aprendizaje Automático<br/>
Tecnológico de Costa Rica <br/>

Código base para el proyecto 1 del curso de aprendizaje automático.

- create_data.m 
  Módulo usado para crear conjuntos de datos bidimensionales siguiendo
  varias distribuciones y varios números de clases.
- plot_data.m
  Módulo para mostrar los datos creados con create_data.m
- sequential.m
  Modelo secuencial, encargado de encadenar capas y pasar la
  información hacia adelante y hacia atrás, así como aplicar las
  técnicas de optimización de parámetros (aprendizaje).
- input_layer.m
  Capa de entrada (100% funcional)
- dense_unbiased.m
  Capa densa sin sesgo (100% funcional)
- logistic.m
  Función sigmoide.  Esto es la función matemática como tal.
- sigmoide.m
  Capa de activación que usa la función sigmoide (implementada en
  logistic.m) para cada entrada, e implementa los métodos hacia adelante y 
  hacia atrás (100% funcional)
- olsloss.m
  Capa de cálculo de pérdida con OLS (100% funcional)
- layer_template.m
  Plantilla para cualquier tipo de capa.  Este archivo enumera y explica
  los métodos esperados por el módulo sequential.m para poder conectar las
  capas y pasar la información hacia adelante y hacia atrás.
- batchnorm.m 
  Plantilla (no funcional) como base para la implementación de la
  normalizacíón por lotes.
- train.m
  Archivo general usado como ejemplo base para crear y mostrar datos, armar
  una red neuronal y entrenarla.  Se presentan dos formas de armar la
  red agregando todas las capas de una sola vez usando un arreglo de celdas,
  o agregando las capas una a una (comentado).
- softmaxvisualizer.m
  Script para poder visualizar los resultados bidimensionales de forma "pie", recibe las matrices X y resultado Y
