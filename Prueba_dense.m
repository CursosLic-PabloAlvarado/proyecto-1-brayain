## Probemos dense.m como regresor lineal.
## Primero, generemos un plano
clear classes;
m=[2; 1];
b=1;
[x1,x2]=meshgrid(linspace(-1,1,31));
X=[x1(:) x2(:)];
y=X*m + b;
figure(1,'name','Plano');
hold off;
plot3(x1(:),x2(:),y,'o');
## Ahora, hagamos una red para regresión lineal
ann = sequential('maxiter',500,
 'alpha',0.005,
 'method','batch',
 'show','progress');
ann.add({input_layer(2),
 dense(1),
 olsloss()});
loss=ann.train(X,y);
figure(2,'name','Pérdida');
hold off;
plot(loss,'linewidth',2);
xlabel('Iteración');
ylabel('Pérdida');
## Finalmente, reconstruyamos con la red
py=ann.test(X);
figure(1);
hold on;
plot3(x1(:),x2(:),py,'x');

