# Gravedad Relativista Oscura (GRO)

Este documento explora una alternativa conceptual para explicar los efectos atribuidos a la materia oscura partiendo de la relatividad general de Einstein.

## 1. Ideas de Einstein sobre la gravedad
- La gravedad surge de la curvatura del espacio-tiempo.
- Introdujo la constante cosmológica \(\Lambda\) para permitir soluciones estáticas, aunque más tarde la calificó como un "error".

## 2. Contraste con la materia oscura
- Observaciones de rotación galáctica y lentes gravitacionales muestran más atracción de la que predicen las estrellas visibles.
- Normalmente se propone materia oscura como partículas invisibles para explicar estos fenómenos.

## 3. Hipótesis
- Una lectura moderna de \(\Lambda\) podría interpretarse como un residuo geométrico que produce efectos oscuros.
- La geometría del espacio-tiempo podría modificarse ligeramente a escalas galácticas sin requerir nueva materia.

## 4. Modelo "Gravedad Relativista Oscura"
- Propone un potencial gravitacional modificado:
  \[\Phi(r) = -\frac{GM}{r} \bigl(1 + \alpha e^{-r/r_0}\bigr)\]
- El término exponencial introduce una atracción adicional que decae con la distancia.
- Ajustando los parámetros \(\alpha\) y \(r_0\) se puede emular la rotación plana de las galaxias.

## 5. Ejemplo de simulación
El archivo [`gro_model.py`](../gro_model.py) muestra cómo calcular curvas de rotación usando esta idea sin recurrir a partículas invisibles.
