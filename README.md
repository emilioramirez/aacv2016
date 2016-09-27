# aacv2016
Repo para laboratorios de Computer Vision

## Clase 20 sep

Ejecutando el codigo como está deberia dar 0.66 algo. Sino, avisar al profe.

### Se vio:
OVR (one-vs-Rest)
OVO (one vs one)

## Clase 27 sep

kmeans++ (cambia la forma de inicializacion) usando esta inicializacion esta
esta acotado. No implementar esto. Solo usarlo

Alternativa es usar el Kmeans de scikit learn

la distancia euclidea es la norma L2 de la da diferencia

```
|| a - b ||**2 = SUM(ai - bi)**2 = aT a - 2 aT b + bT b
                                 = ||a||**2 - 2aT b + ||b||**2
                   ||a - b ||2-2 = 2*(1 - aT b)


|| x ||**2 = xT x = SUM(xi, yi)
```

### Filminas
RIDGE (penalizar grandes valores de coeficientes) (normalizacion L2)
variando el valor de lambda puedo controlar el valor de los coeficientes (los quiero bajo)
esto se hace con cross validation, ir variando el valor de lambda y dejarlo donde convenga


Transformaciones L2/sqrt/
sqrt y despues L2 (es equivalente a hacer L1 y despues sqrt)



### Lab1 punto 1:

#### Dividir el dataset en
dataset = train + test
train = train* + validation (5k-fold) experimentar con:

#### Calcular Acc usando varios C (escala logaritmica)
C = 0.01 - Acc = 0.60
C = 0.10 - Acc = 0.62
C = 1.00 - Acc = 0.66 * (elijo este)
C = 10.0 - Acc = 0.50

#### Graficar
Quiere grafica de C vs Acc (con C=x, Acc=y)


### Lab1 punto 2:

Grafico de K=nro clusters vs Acc (K=x, Acc=y)


### Lab1 punto 3:
La transformacion en descriptores es:

Sqrt = (x1, ...., xd) -> (sign(x1)*sqrt(abs(x1)), ....., sign(xd)*sqrt(abs(xd)))  xi pertenece R

L2 = x / abs(x)

Aplicar sqrt solo, L2 solo, combinacion de ambas:

|          | SIFT  |  BoVW  |  SIFT+BoVW  |
|----------|-------|--------|-------------|
|  sqrt    |   -   |   -    |     -       |
|  L2      |   -   |   -    |     -       |
|  sqrt+L2 |   -   |   -    |     -       |



Esto dijo en la clase del 27 sep:

|  DAISI/Bovw   | L2 Norm  |  sqrt  |  L2 + sqrt  |
|---------------|----------|--------|-------------|
|  L2 norm      |   -      |   -    |     -       |
|  sqrt         |   -      |   -    |     -       |
|  sqrt+L2 norm |   -      |   -    |     -       |  media y desviacion estandar
