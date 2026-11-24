
Descripción 

El robot DARwIn-OP permite interactuar con un humanoide, es una herramienta open-source. Para usar este robot se puede simular su comportamiento mediante Webots.  

El objetivo del trabajo es desarrollar un entorno que permita entrenar agentes para participar en una carrera desde un punto inicial a un punto final: 

Agente propio – Usando el algoritmo propuesto del taller 2 

Agentes base 1 – Algoritmo PPO 

Agentes base 2 – Algoritmo DQN 

Agentes base 3 – Algoritmo Actor Critic 

El robot que llegue primero a la meta gana la carrera. 

 

Cada grupo deberá construir un entorno de carreras y desarrollar un sistema que permita al robot: 

Recorrer autónomamente una pista. 

Detectar y responder a obstáculos. 

Aprender a desplazarse y tomar decisiones. 

Competir contra los otros agentes. 

Mejorar su desempeño mediante entrenamiento. 

Restricciones del Proyecto 

La simulación deberá realizarse en Webots. 

Se deberá utilizar el robot humanoide DARwIn-OP. 

Cada grupo deberá construir su propio entorno de carreras. 

La pista deberá incluir obstáculos, curvas y diferentes condiciones de recorrido. 

La competencia deberá realizarse contra otros agentes, en este caso otros robots DARwIn-OP. 

 

Simulación 

 

La simulación debe permitir: 

Definir una línea de salida y una meta. 

Incorporar obstáculos y diferentes trayectorias. 

Ejecutar agentes en competencia. 

Visualizar y registrar el comportamiento de los robots. 

Medir el desempeño durante las carreras. 

Ejercicio 

 

Deberá entregar un informe, donde: 

 

[1 punto] Explique brevemente los algoritmos, su lógica general y las razones por las cuales podrían funcionar para el problema planteado. 

[0.5 puntos] Uso de safe learning, en donde el robot debe aprender tomando precauciones de no causarse daño, causar daño a los humanos u otros robots. 

[1 punto] Implementación y ejecución 

Uso de los algoritmos. Indicar si es implementación propia o uso algúna librería y que parámetros uso en cada algoritmo. 

Cada algoritmo debe: 

explorar el entorno 

Participar en la carrera 

[2.0 puntos] Experimentos y comparación 

Realice mínimo 20 experimentos variando: 

tamaño del espacio,  

posición inicial,  

posición de la meta,  

parámetros del algoritmo.  

Compare los algoritmos utilizando al menos dos de los siguientes criterios: 

Recompensa acumulada 

Tasa de éxito 

Los resultados deben presentarse mediante un informe en formato IEEE con: 
tablas,  
gráficos,  
métricas promedio. 
[0.5 puntos] Conclusiones y resultados. Identificar qué algoritmo funcionó mejor, en qué casos falló cada uno y explicar brevemente el motivo del fallo. Además, debe dar recomendaciones finales basadas en los resultados obtenidos. 

 
