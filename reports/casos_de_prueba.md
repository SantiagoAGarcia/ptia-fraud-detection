# Casos de prueba (entrada → salida esperada → salida obtenida)

Las variables V* son componentes PCA anónimas del dataset; se muestran las 4 más importantes del RF. La entrada completa (31 variables) está en `casos_de_prueba.csv/.json` y se puede enviar tal cual a `POST /predict`.

| caso | fila_dataset | hora | monto | V14 | V10 | V12 | V4 | esperado | obtenido | score_rf | score_if | score_final | umbral | regla_fija |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Acierto 1: fraude que la regla fija NO ve | 15810 | 7 | 99.99 | -6.326 | -8.234 | -6.526 | 6.378 | FRAUDE | FRAUDE | 0.9996 | 0.8876 | 0.9884 | 0.4673 | LEGITIMA |
| Acierto 2: legítima que la regla fija marca como fraude | 46841 | 11 | 12,910.93 | 2.354 | -5.533 | 0.903 | 9.925 | LEGITIMA | LEGITIMA | 0.0194 | 0.8603 | 0.1035 | 0.4673 | FRAUDE |
| Más significativo: fraude detectado donde el RF fue menos seguro | 70589 | 14 | 261.87 | -4.059 | -5.46 | -2.33 | -0.573 | FRAUDE | FRAUDE | 0.4998 | 0.4647 | 0.4963 | 0.4673 | LEGITIMA |
| Error 1: fraude NO detectado (falso negativo) | 219025 | 15 | 4.49 | -0.658 | -0.159 | 0.282 | 0.117 | FRAUDE | LEGITIMA | 0.0014 | 0.055 | 0.0067 | 0.4673 | LEGITIMA |
| Error 2: alerta falsa (falso positivo) | 16110 | 7 | 1.00 | -6.052 | -8.098 | -6.315 | 6.408 | LEGITIMA | FRAUDE | 0.9896 | 0.8883 | 0.9795 | 0.4673 | LEGITIMA |

- **Acierto 1: fraude que la regla fija NO ve** (fila 15810): Fraude detectado con el score final más alto entre los que la regla fija deja pasar.
- **Acierto 2: legítima que la regla fija marca como fraude** (fila 46841): Transacción legítima (la regla fija la bloquearía) que el ensemble deja pasar.
- **Más significativo: fraude detectado donde el RF fue menos seguro** (fila 70589): Ningún fraude de test depende solo del IF (el RF solo ya los marca). Este es el de menor score RF (0.50); el IF aporta 0.46.
- **Error 1: fraude NO detectado (falso negativo)** (fila 219025): Fraude con el score final más bajo: parece una transacción normal para ambos modelos.
- **Error 2: alerta falsa (falso positivo)** (fila 16110): Legítima con el score más alto entre las que el ensemble marcó como fraude.
