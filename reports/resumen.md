# Resumen de evaluación

Objetivo de ajuste en validación: **F1** · semilla 42 · split 60/20/20 estratificado.
Fraudes en test: 98 de 56,962 (0.172 %).

## Configuración elegida (validación)
- Pesos: RF **0.9** / IF **0.1** · umbral **0.4673**
- Original (Hito 2): 0.7 / 0.3 · 0.85
- Guardada en `models/config.json`: sí

Búsqueda de pesos en validación (cada fila usa su mejor umbral; las filas con w_if = 0 y 1 son RF solo e IF solo,
no candidatas; entre candidatas con F-beta a ≤ 0.005 del mejor se elige la de mayor PR-AUC en validación):

| w_rf | w_if | umbral | F1_val | PR-AUC_val | ROC-AUC_val |
|---|---|---|---|---|---|
| 1.0000 | 0.0000 | 0.4510 | 0.8191 | 0.7640 | 0.9717 |
| 0.9000 | 0.1000 | 0.4673 | 0.8172 | 0.7611 | 0.9672 |
| 0.8000 | 0.2000 | 0.4797 | 0.8172 | 0.7540 | 0.9649 |
| 0.7000 | 0.3000 | 0.4765 | 0.8172 | 0.7459 | 0.9623 |
| 0.6000 | 0.4000 | 0.4588 | 0.8172 | 0.7340 | 0.9600 |
| 0.5000 | 0.5000 | 0.4743 | 0.8065 | 0.7165 | 0.9579 |
| 0.0000 | 1.0000 | 0.7218 | 0.2857 | 0.1628 | 0.9400 |

## Métricas en test
| Modelo | Recall | Precision | F1 | ROC-AUC | PR-AUC | TP | FP | FN | TN | Umbral |
|---|---|---|---|---|---|---|---|---|---|---|
| Regla fija (monto > 2000 o madrugada) | 0.2347 | 0.0046 | 0.0090 | — | — | 23 | 4992 | 75 | 51872 | — |
| IF solo | 0.3469 | 0.3434 | 0.3452 | 0.9567 | 0.2459 | 34 | 65 | 64 | 56799 | 0.7220 |
| RF solo | 0.8367 | 0.8367 | 0.8367 | 0.9721 | 0.8109 | 82 | 16 | 16 | 56848 | 0.4510 |
| Ensemble original (0.7/0.3, u=0.85) | 0.4898 | 0.8276 | 0.6154 | 0.9684 | 0.7655 | 48 | 10 | 50 | 56854 | 0.8500 |
| Ensemble ajustado (0.9/0.1, u=0.467) | 0.8367 | 0.8454 | 0.8410 | 0.9707 | 0.7970 | 82 | 15 | 16 | 56849 | 0.4670 |

La regla fija usa los parámetros que maximizan F1 en validación (monto > 2000
o madrugada, hora ≤ 5); no tiene score continuo, por eso no tiene AUC.

El ensemble es **prácticamente equivalente** al RF solo (PR-AUC -0.014, F1 +0.004, Recall +0.000). Con 98 fraudes en test, un solo fraude mueve el Recall 0.010: esa diferencia no es concluyente. El aporte del IF es conceptual (cobertura ante fraudes nuevos), no una mejora medible en este dataset.

## Promesa: «30 % menos falsos positivos que las reglas»
- Falsos positivos en test: regla = 4992, ensemble ajustado = 15
  → reducción **99.7 %** (Recall regla 0.235 vs ensemble 0.837).
- Exigiendo al ensemble el MISMO Recall que la regla (0.235): 1 falsos positivos
  → reducción **100.0 %**.
- Veredicto: **se cumple**.

## Archivos
`metricas.csv|tex`, `matriz_confusion.tex`, `grid_pesos.csv`, `casos_de_prueba.csv|md|json`,
`curvas_pr_roc.png`, `matrices_confusion.png`, `comparacion_modelos.png`,
`distribucion_scores.png`, `importancia_variables.png`.
