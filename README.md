# aacv2016
Repo para laboratorios de Computer Vision

## Clase 20 sep

Ejecutando el codigo como está deberia dar 0.66 algo. Sino, avisar al profe.

### Se vio:
OVR (one-vs-Rest)
OVO (one vs one)


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

        SIFT    BoVW    SIFT+BoVW
sqrt      -       -         -
L2        -       -         -
sqrt+L2   -       -         -

