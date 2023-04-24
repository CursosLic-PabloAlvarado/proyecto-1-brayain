function softmaxvisualizer(X,Y)
  ## Initial configuration for the optimizer
  opt=optimizer("method","sgd",
                "minibatch",11,
                "maxiter",600,
                "alpha",0.05);
  ###
  Xtr = [ones(length(X),1) X];
  NX=normalizer("normal");
  NXtr=NX.fit_transform(Xtr);
  theta0=rand(columns(NXtr),4)-0.5;%OJO como Y son 5 clases, theta tiene que acoplarse a este tamaño
  comp=100;
  columna1=0;
  columna2=0;
  for i=1:2
    for j=i+1:3%#cols de X

      feats=[i,j];
      x2=Xtr(:,feats);
      N2=normalizer("normal");
      nx2=N2.fit_transform(x2);

      opt.configure("method","batch"); ## Just change the method
      [ts,errs]=opt.minimize(@softmax_loss,@softmax_gradloss,theta0(feats,:),nx2,Y);
      theta2=ts{end};

      py2=softmax_hyp(theta2,nx2);
      err2=sum((py2>0.5)!=Y);
      tot2=100*(err2/rows(Y));

      if tot2<=comp
        comp=tot2;
        columna1=i;
        columna2=j;
      endif

      mins=min(x2);
      maxs=max(x2);

      e11=linspace(mins(1),maxs(1),500);
      e22=linspace(mins(2),maxs(2),500);
    endfor

  endfor

comp
  printf("el menor error obtenido es: %d al evaluar las columnas %d y %d \n", min(comp), columna1, columna2);



  [ee11,ee22]=meshgrid(e11,e22);
  x22test=N2.transform([ee11(:) ee22(:)]);

  ytest2=softmax_hyp(theta2,x22test);

  ygrap=zeros(size(ytest2(:,1)));
  for i=1:length(ygrap)
    c1=ytest2(i,1);
    c2=ytest2(i,2);
    c3=ytest2(i,3);
    c4=ytest2(i,4);
    c5=ytest2(i,5);

    if c1>c2 && c1>c3 && c1>c4 && c1>c5
      ygrap(i)=1;
    endif
    if c2>c1 && c2>c3 && c2>c4 && c2>c5
      ygrap(i)=2;
    endif
    if c3>c2 && c3>c1 && c3>c4 && c3>c5
      ygrap(i)=3;
    endif
    if c4>c2 && c4>c3 && c4>c1 && c4>c5
      ygrap(i)=4;
    endif
    if c5>c2 && c5>c3 && c5>c4 && c5>c1
      ygrap(i)=5;
    endif
  endfor


  cmap = [0.1,0.4,0.6; 0.5,0,0.5 ;0,0,1;0,1,0;1,0,0];
  img = reshape(ygrap,size(ee11));
  rgb_img = ind2rgb(img, cmap);
  figure(5,"name","Regiones de las clases ganadoras para el espacio de entrada bidimensional");
  image(rgb_img);
  xlabel("culmen length [mm]");
  ylabel("bodymass [g]");
  axis equal;
  hold on;

  y_prob = (ytest2) ./ sum((ytest2), 2);
  color_weight = y_prob * cmap; % Calcula los pesos para cada color
  mixed_color= reshape(color_weight, [500 500 3]);

  figure(6,"name","Ponderacion de colores asignados a las clases, de acuerdo a la probabilidad de pertenecer a esa clase");
  image(mixed_color);
  xlabel("culmen length [mm]");
  ylabel("bodymass [g]");
  axis equal;


endfunction
