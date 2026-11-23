"""Evaluación del ensemble RF + IF (Hito 3, secciones 6 y 7).

Uso:
    python evaluate.py                    # ajusta pesos/umbral (F1) y evalúa
    python evaluate.py --objective f2     # prioriza Recall sobre Precision
    python evaluate.py --no-save-config   # no toca models/config.json

Protocolo (para poder defenderlo en el documento):
  1. Los modelos se entrenan con TRAIN (60 %).
  2. Pesos y umbral se eligen con VALIDACIÓN (20 %)  -> nunca con test.
  3. Todas las métricas finales se calculan UNA vez sobre TEST (20 %).
  3b. Si varios pesos empatan en validación (diferencia < 0.005 de F-beta), se elige el
      de mayor PR-AUC en validación (métrica sin umbral, la más fiable con 0.17 % de fraudes).
  4. Todos los modelos comparados (RF, IF, ensemble) usan su propio umbral
     elegido en validación, para que la comparación sea justa.

Salidas en reports/: metricas.csv/.tex, matriz_confusion.tex, grid_pesos.csv,
resumen.md, casos_de_prueba.(csv|md|json) y las gráficas PNG.
Además escribe models/config.json con los pesos y el umbral elegidos.
"""
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src.ensemble import FraudEnsemble
from src.preprocessing import DATA_PATH, MODELS_DIR, ROOT_DIR, load_splits

REPORTS = ROOT_DIR / 'reports'
W_IF_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]   # 0.0 = RF solo, 1.0 = IF solo
W_IF_CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5]       # pesos válidos para el ensemble
ORIGINAL = {'w_rf': 0.7, 'w_if': 0.3, 'threshold': 0.85}
AMOUNT_GRID = [50, 100, 200, 300, 500, 1000, 2000]
TIE_TOL = 0.005   # pesos con F-beta a menos de esto del mejor se consideran empatados


# ----------------------------------------------------------------------------- utilidades
def best_threshold(y, score, beta=1.0):
    """Umbral que maximiza F-beta. Devuelve (umbral estricto, F-beta).

    El umbral devuelto se usa como `score > umbral` (igual que FraudEnsemble).
    """
    p, r, t = precision_recall_curve(y, score)
    p, r = p[:-1], r[:-1]
    b2 = beta ** 2
    den = b2 * p + r
    f = np.divide((1 + b2) * p * r, den, out=np.zeros_like(p), where=den > 0)
    i = int(np.argmax(f))
    strict = float(np.floor(np.nextafter(t[i], -np.inf) * 1e6) / 1e6)
    return strict, float(f[i])


def compute_metrics(y, pred, score=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'Recall': recall_score(y, pred, zero_division=0),
        'Precision': precision_score(y, pred, zero_division=0),
        'F1': f1_score(y, pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y, score) if score is not None else np.nan,
        'PR-AUC': average_precision_score(y, score) if score is not None else np.nan,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    }


def fit_rule(X, y):
    """Mejor regla fija posible (por F1 en validación): monto > X, opcionalmente o madrugada."""
    amount = np.expm1(X['LogAmount']).values
    best = None
    for thr in AMOUNT_GRID:
        for night in (False, True):
            pred = amount > thr
            if night:
                pred = pred | (X['Hour'].values <= 5)
            f1 = f1_score(y, pred.astype(int), zero_division=0)
            if best is None or f1 > best['f1_val']:
                best = {'amount': thr, 'night': night, 'f1_val': f1}
    return best


def apply_rule(X, rule):
    pred = np.expm1(X['LogAmount']).values > rule['amount']
    if rule['night']:
        pred = pred | (X['Hour'].values <= 5)
    return pred.astype(int)


def rule_name(rule):
    return f"Regla fija (monto > {rule['amount']}" + (" o madrugada)" if rule['night'] else ")")


def fp_at_recall(y, score, recall_target):
    """Falsos positivos que comete el score cuando se exige al menos `recall_target`."""
    fpr, tpr, _ = roc_curve(y, score)
    i = int(np.argmax(tpr >= recall_target))
    return fpr[i] * int((np.asarray(y) == 0).sum()), tpr[i]


def md_table(df, floatfmt='{:.4f}'):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return '—' if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = '| ' + ' | '.join(df.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(df.columns)) + '|'
    rows = ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join([head, sep] + rows)


def tex_table(df, caption, label, floatfmt='{:.4f}'):
    def esc(s):
        return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return '--' if np.isnan(v) else floatfmt.format(v)
        return esc(v)
    cols = ' & '.join(esc(c) for c in df.columns) + r' \\'
    body = [' & '.join(fmt(v) for v in r) + r' \\' for r in df.itertuples(index=False)]
    lines = [r'\begin{table}[H]', r'\centering', r'\small',
             r'\begin{tabular}{l' + 'r' * (len(df.columns) - 1) + '}', r'\hline', cols, r'\hline',
             *body, r'\hline', r'\end{tabular}',
             rf'\caption{{{esc(caption)}}}', rf'\label{{{label}}}', r'\end{table}']
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------------- gráficas
def plot_curves(y, curves, rule_point, rule_label):
    colors = {'RF solo': '#1f77b4', 'IF solo': '#2ca02c'}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, s in curves.items():
        c = colors.get(name, '#d62728')
        p, r, _ = precision_recall_curve(y, s)
        ax[0].plot(r, p, label=f"{name} (PR-AUC={average_precision_score(y, s):.3f})", color=c)
        fpr, tpr, _ = roc_curve(y, s)
        ax[1].plot(fpr, tpr, label=f"{name} (ROC-AUC={roc_auc_score(y, s):.3f})", color=c)
    ax[0].scatter(rule_point['Recall'], rule_point['Precision'], marker='X', s=90, color='k', label=rule_label)
    fpr_rule = rule_point['FP'] / (rule_point['FP'] + rule_point['TN'])
    ax[1].scatter(fpr_rule, rule_point['Recall'], marker='X', s=90, color='k', label=rule_label)
    ax[1].plot([0, 1], [0, 1], '--', color='gray', lw=0.8)
    ax[0].set(xlabel='Recall', ylabel='Precision', title='Curva Precision-Recall (test)')
    ax[1].set(xlabel='Tasa de falsos positivos', ylabel='Recall', title='Curva ROC (test)')
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / 'curvas_pr_roc.png', dpi=150)
    plt.close(fig)


def plot_confusions(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    for ax, (name, m) in zip(np.atleast_1d(axes), results.items()):
        cm = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center',
                        color='white' if cm[i, j] / cm[i].sum() > 0.5 else 'black', fontsize=11)
        ax.set_xticks([0, 1], ['Legítima', 'Fraude'])
        ax.set_yticks([0, 1], ['Legítima', 'Fraude'])
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.set_title(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / 'matrices_confusion.png', dpi=150)
    plt.close(fig)


def plot_bars(table):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(table))
    for k, metric in enumerate(['Recall', 'Precision', 'F1']):
        bars = ax.bar(x + (k - 1) * 0.26, table[metric], 0.26, label=metric)
        ax.bar_label(bars, fmt='%.2f', fontsize=7)
    ax.set_xticks(x, [n.replace(' (', '\n(') for n in table['Modelo']], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title('Comparación de modelos en test')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / 'comparacion_modelos.png', dpi=150)
    plt.close(fig)


def plot_score_distributions(y, rf, iff, final, thr):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (title, s) in zip(ax, [('Score RF', rf), ('Score IF (normalizado)', iff), ('Score final del ensemble', final)]):
        bins = np.linspace(0, 1, 41)
        a.hist(s[y == 0], bins=bins, alpha=0.7, label='Legítima', color='#1f77b4')
        a.hist(s[y == 1], bins=bins, alpha=0.8, label='Fraude', color='#d62728')
        a.set_yscale('log')
        a.set_title(title)
        a.set_xlabel('score')
        a.grid(alpha=0.3)
    ax[2].axvline(thr, color='k', ls='--', label=f'umbral = {thr:.3f}')
    ax[0].set_ylabel('transacciones (escala log)')
    ax[2].legend()
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(REPORTS / 'distribucion_scores.png', dpi=150)
    plt.close(fig)


def plot_importance(ens):
    imp = pd.Series(ens.rf.feature_importances_, index=ens.features).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax, color='#1f77b4')
    ax.set_title('Importancia de variables (Random Forest, top 15)')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / 'importancia_variables.png', dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- casos de prueba
def pick_cases(X, y, rf, iff, final, thr, rf_thr, rule_pred):
    d = X.copy()
    d['amount'] = np.expm1(X['LogAmount'])
    d['clase'] = y.values
    d['rf_score'], d['if_score'], d['final_score'] = rf, iff, final
    d['pred'] = (final > thr).astype(int)
    d['rf_solo_pred'] = (rf > rf_thr).astype(int)
    d['regla_pred'] = rule_pred

    def pick(mask, col, ascending=False):
        sub = d[mask & ~d.index.isin(list(used))]
        return None if sub.empty else sub.sort_values(col, ascending=ascending).index[0]

    chosen, used = [], set()

    def first(*candidates):
        # OJO: no usar `or` (el índice 0 es falso en Python)
        return next((c for c in candidates if c is not None), None)

    def add(name, idx, why):
        if idx is not None and idx not in used:
            used.add(idx)
            chosen.append((name, idx, why))

    add('Acierto 1: fraude que la regla fija NO ve',
        first(pick((d.clase == 1) & (d.pred == 1) & (d.regla_pred == 0), 'final_score'),
              pick((d.clase == 1) & (d.pred == 1), 'final_score')),
        'Fraude detectado con el score final más alto entre los que la regla fija deja pasar.')
    add('Acierto 2: legítima que la regla fija marca como fraude',
        first(pick((d.clase == 0) & (d.pred == 0) & (d.regla_pred == 1), 'amount'),
              pick((d.clase == 0) & (d.pred == 0), 'final_score')),
        'Transacción legítima (la regla fija la bloquearía) que el ensemble deja pasar.')
    idx_if = pick((d.clase == 1) & (d.pred == 1) & (d.rf_solo_pred == 0), 'if_score')
    if idx_if is not None:
        add('Más significativo: fraude detectado gracias al IF', idx_if,
            'Fraude que el RF solo NO marcaría; el aporte del IF empuja el score final sobre el umbral.')
    else:
        idx_lo = pick((d.clase == 1) & (d.pred == 1), 'rf_score', ascending=True)
        if idx_lo is not None:
            r = d.loc[idx_lo]
            add('Más significativo: fraude detectado donde el RF fue menos seguro', idx_lo,
                f"Ningún fraude de test depende solo del IF (el RF solo ya los marca). Este es el de menor score RF "
                f"({r.rf_score:.2f}); el IF aporta {r.if_score:.2f}.")
    add('Error 1: fraude NO detectado (falso negativo)',
        pick((d.clase == 1) & (d.pred == 0), 'final_score', ascending=True),
        'Fraude con el score final más bajo: parece una transacción normal para ambos modelos.')
    add('Error 2: alerta falsa (falso positivo)',
        pick((d.clase == 0) & (d.pred == 1), 'final_score'),
        'Legítima con el score más alto entre las que el ensemble marcó como fraude.')
    return d, chosen


def write_cases(d, chosen, thr, ens):
    top = list(pd.Series(ens.rf.feature_importances_, index=ens.features).sort_values(ascending=False).index[:4])
    rows, js = [], []
    for name, idx, why in chosen:
        r = d.loc[idx]
        row = {'caso': name, 'fila_dataset': int(idx), 'hora': int(r['Hour']), 'monto': round(float(r['amount']), 2),
               **{f: round(float(r[f]), 3) for f in top},
               'esperado': 'FRAUDE' if r['clase'] == 1 else 'LEGITIMA',
               'obtenido': 'FRAUDE' if r['pred'] == 1 else 'LEGITIMA',
               'score_rf': round(float(r['rf_score']), 4), 'score_if': round(float(r['if_score']), 4),
               'score_final': round(float(r['final_score']), 4), 'umbral': round(float(thr), 4),
               'regla_fija': 'FRAUDE' if r['regla_pred'] == 1 else 'LEGITIMA', 'por_que_se_eligio': why}
        rows.append(row)
        js.append({**row, 'entrada': {f: float(r[f]) for f in ens.features}})
    out = pd.DataFrame(rows)
    full = pd.DataFrame([{**row, **j['entrada']} for row, j in zip(rows, js)])
    full.to_csv(REPORTS / 'casos_de_prueba.csv', index=False)
    json.dump(js, open(REPORTS / 'casos_de_prueba.json', 'w'), indent=2, ensure_ascii=False)
    md = ['# Casos de prueba (entrada → salida esperada → salida obtenida)', '',
          f'Las variables V* son componentes PCA anónimas del dataset; se muestran las {len(top)} más importantes '
          'del RF. La entrada completa (31 variables) está en `casos_de_prueba.csv/.json` y se puede enviar tal cual '
          'a `POST /predict`.', '', md_table(out.drop(columns=['por_que_se_eligio']).assign(
              monto=lambda x: x['monto'].map('{:,.2f}'.format)), floatfmt='{:.4g}'), '']
    for r in rows:
        md.append(f"- **{r['caso']}** (fila {r['fila_dataset']}): {r['por_que_se_eligio']}")
    (REPORTS / 'casos_de_prueba.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DATA_PATH))
    ap.add_argument('--objective', choices=['f1', 'f2'], default='f1',
                    help='métrica a maximizar al elegir pesos y umbral en validación')
    ap.add_argument('--no-save-config', action='store_true')
    ap.add_argument('--force', action='store_true', help='evaluar aunque los modelos no coincidan con el split')
    args = ap.parse_args()
    beta = 2.0 if args.objective == 'f2' else 1.0
    REPORTS.mkdir(exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(args.data)

    info_path = MODELS_DIR / 'train_info.json'
    info = json.load(open(info_path)) if info_path.exists() else None
    if (info is None or info['n_train'] != len(X_train)) and not args.force:
        sys.exit("❌ Los modelos en models/ no se entrenaron con este split (o falta train_info.json).\n"
                 "   Ejecuta primero `python train.py` (así validación y test quedan sin ver).")

    ens = FraudEnsemble()
    rf_v, if_v = ens.component_scores(X_val)
    rf_t, if_t = ens.component_scores(X_test)
    yv, yt = y_val.values, y_test.values

    # 1) Ajuste de pesos y umbral SOLO con validación ---------------------------------
    grid = []
    for w_if in W_IF_GRID:
        w_rf = round(1 - w_if, 2)
        s = ens.combine(rf_v, if_v, w_rf, w_if)
        thr, fb = best_threshold(yv, s, beta)
        grid.append({'w_rf': w_rf, 'w_if': w_if, 'umbral': thr, f'F{int(beta)}_val': fb,
                     'PR-AUC_val': average_precision_score(yv, s), 'ROC-AUC_val': roc_auc_score(yv, s),
                     'candidato': w_if in W_IF_CANDIDATES})
    grid_df = pd.DataFrame(grid)
    fcol = f'F{int(beta)}_val'
    cand = grid_df[grid_df.candidato]
    near_best = cand[cand[fcol] >= cand[fcol].max() - TIE_TOL]
    best = near_best.sort_values('PR-AUC_val', ascending=False).iloc[0]   # desempate: mayor PR-AUC
    tuned = {'w_rf': float(best.w_rf), 'w_if': float(best.w_if), 'threshold': float(best.umbral)}
    rf_row = grid_df[grid_df.w_if == 0.0].iloc[0]
    if_row = grid_df[grid_df.w_if == 1.0].iloc[0]
    grid_df.to_csv(REPORTS / 'grid_pesos.csv', index=False)

    # 2) Regla fija (mejor posible, ajustada en validación) --------------------------
    rule = fit_rule(X_val, yv)
    rule_pred_t = apply_rule(X_test, rule)

    # 3) Evaluación en TEST ------------------------------------------------------------
    s_orig = ens.combine(rf_t, if_t, ORIGINAL['w_rf'], ORIGINAL['w_if'])
    s_tuned = ens.combine(rf_t, if_t, tuned['w_rf'], tuned['w_if'])
    name_rule = rule_name(rule)
    name_orig = f"Ensemble original ({ORIGINAL['w_rf']}/{ORIGINAL['w_if']}, u={ORIGINAL['threshold']})"
    name_tuned = f"Ensemble ajustado ({tuned['w_rf']}/{tuned['w_if']}, u={tuned['threshold']:.3f})"
    runs = {
        name_rule: (None, rule_pred_t, '—'),
        'IF solo': (if_t, (if_t > if_row.umbral).astype(int), if_row.umbral),
        'RF solo': (rf_t, (rf_t > rf_row.umbral).astype(int), rf_row.umbral),
        name_orig: (s_orig, (s_orig > ORIGINAL['threshold']).astype(int), ORIGINAL['threshold']),
        name_tuned: (s_tuned, (s_tuned > tuned['threshold']).astype(int), tuned['threshold']),
    }
    results = {n: compute_metrics(yt, p, s) for n, (s, p, _) in runs.items()}
    table = pd.DataFrame([{'Modelo': n, **m} for n, m in results.items()])
    table['Umbral'] = [u if isinstance(u, str) else round(float(u), 3) for _, _, u in runs.values()]
    table.to_csv(REPORTS / 'metricas.csv', index=False)
    (REPORTS / 'metricas.tex').write_text(tex_table(
        table[['Modelo', 'Recall', 'Precision', 'F1', 'ROC-AUC', 'PR-AUC']],
        'Métricas en el conjunto de prueba', 'tab:metricas'), encoding='utf-8')
    (REPORTS / 'matriz_confusion.tex').write_text(tex_table(
        table[['Modelo', 'TP', 'FP', 'FN', 'TN']], 'Matriz de confusión en el conjunto de prueba (conteos)',
        'tab:confusion'), encoding='utf-8')

    # 4) Promesa del documento: 30 % menos falsos positivos que las reglas ---------------
    m_rule, m_ens = results[name_rule], results[name_tuned]
    red_own = 1 - m_ens['FP'] / m_rule['FP'] if m_rule['FP'] else np.nan
    fp_eq, rec_eq = fp_at_recall(yt, s_tuned, m_rule['Recall'])
    red_eq = 1 - fp_eq / m_rule['FP'] if m_rule['FP'] else np.nan
    meets = (m_ens['Recall'] >= m_rule['Recall']) and red_own >= 0.30

    # 5) Gráficas y casos --------------------------------------------------------------
    plot_curves(yt, {'RF solo': rf_t, 'IF solo': if_t, name_tuned: s_tuned}, m_rule, name_rule)
    plot_confusions(results)
    plot_bars(table)
    plot_score_distributions(yt, rf_t, if_t, s_tuned, tuned['threshold'])
    plot_importance(ens)
    d, chosen = pick_cases(X_test, y_test, rf_t, if_t, s_tuned, tuned['threshold'], rf_row.umbral, rule_pred_t)
    cases = write_cases(d, chosen, tuned['threshold'], ens)

    # 6) Guardar config y resumen ------------------------------------------------------
    if not args.no_save_config:
        json.dump({**tuned, 'objective': args.objective, 'tuned_on': 'validation'},
                  open(MODELS_DIR / 'config.json', 'w'), indent=2)

    e, r_ = results[name_tuned], results['RF solo']
    d_pr, d_f1, d_rec = e['PR-AUC'] - r_['PR-AUC'], e['F1'] - r_['F1'], e['Recall'] - r_['Recall']
    one_fraud = 1 / max(int(yt.sum()), 1)
    if d_pr > 0.005 and d_f1 >= 0:
        ifmsg = (f"El Isolation Forest **mejora** al RF solo (PR-AUC {d_pr:+.3f}, F1 {d_f1:+.3f}, "
                 f"Recall {d_rec:+.3f}).")
    elif abs(d_pr) <= 0.02 and abs(d_f1) <= 0.02:
        ifmsg = (f"El ensemble es **prácticamente equivalente** al RF solo (PR-AUC {d_pr:+.3f}, F1 {d_f1:+.3f}, "
                 f"Recall {d_rec:+.3f}). Con {int(yt.sum())} fraudes en test, un solo fraude mueve el Recall "
                 f"{one_fraud:.3f}: esa diferencia no es concluyente. El aporte del IF es conceptual "
                 "(cobertura ante fraudes nuevos), no una mejora medible en este dataset.")
    else:
        ifmsg = (f"El ensemble **no mejora** al RF solo en este dataset (PR-AUC {d_pr:+.3f}, F1 {d_f1:+.3f}, "
                 f"Recall {d_rec:+.3f}). Repórtalo con honestidad: el IF aporta cobertura frente a fraudes "
                 "nuevos (no supervisado), pero no sube las métricas aquí.")
    best_line = grid_df[grid_df.candidato].to_string(index=False, float_format=lambda v: f'{v:.4f}')
    resumen = f"""# Resumen de evaluación

Objetivo de ajuste en validación: **{args.objective.upper()}** · semilla 42 · split 60/20/20 estratificado.
Fraudes en test: {int(yt.sum())} de {len(yt):,} ({yt.mean() * 100:.3f} %).

## Configuración elegida (validación)
- Pesos: RF **{tuned['w_rf']}** / IF **{tuned['w_if']}** · umbral **{tuned['threshold']:.4f}**
- Original (Hito 2): 0.7 / 0.3 · 0.85
- Guardada en `models/config.json`: {'no (--no-save-config)' if args.no_save_config else 'sí'}

Búsqueda de pesos en validación (cada fila usa su mejor umbral; las filas con w_if = 0 y 1 son RF solo e IF solo,
no candidatas; entre candidatas con F-beta a ≤ {TIE_TOL} del mejor se elige la de mayor PR-AUC en validación):

{md_table(grid_df.drop(columns=['candidato']), floatfmt='{:.4f}')}

## Métricas en test
{md_table(table[['Modelo', 'Recall', 'Precision', 'F1', 'ROC-AUC', 'PR-AUC', 'TP', 'FP', 'FN', 'TN', 'Umbral']])}

La regla fija usa los parámetros que maximizan F1 en validación (monto > {rule['amount']}
{'o madrugada, hora ≤ 5' if rule['night'] else 'sin condición de hora'}); no tiene score continuo, por eso no tiene AUC.

{ifmsg}

## Promesa: «30 % menos falsos positivos que las reglas»
- Falsos positivos en test: regla = {m_rule['FP']}, ensemble ajustado = {m_ens['FP']}
  → reducción **{red_own * 100:.1f} %** (Recall regla {m_rule['Recall']:.3f} vs ensemble {m_ens['Recall']:.3f}).
- Exigiendo al ensemble el MISMO Recall que la regla ({m_rule['Recall']:.3f}): {fp_eq:.0f} falsos positivos
  → reducción **{red_eq * 100:.1f} %**.
- Veredicto: **{'se cumple' if meets else 'NO se cumple en el punto de operación elegido: revisa la redacción de la promesa en el documento'}**.

## Archivos
`metricas.csv|tex`, `matriz_confusion.tex`, `grid_pesos.csv`, `casos_de_prueba.csv|md|json`,
`curvas_pr_roc.png`, `matrices_confusion.png`, `comparacion_modelos.png`,
`distribucion_scores.png`, `importancia_variables.png`.
"""
    (REPORTS / 'resumen.md').write_text(resumen, encoding='utf-8')

    pd.set_option('display.width', 200)
    print('\n=== Pesos (validación) ===\n' + best_line)
    print(f"\n>>> Elegido: RF {tuned['w_rf']} / IF {tuned['w_if']}, umbral {tuned['threshold']:.4f}")
    print('\n=== Métricas en TEST ===')
    print(table[['Modelo', 'Recall', 'Precision', 'F1', 'ROC-AUC', 'PR-AUC', 'FP', 'FN']].to_string(
        index=False, float_format=lambda v: f'{v:.4f}'))
    print(f"\nFalsos positivos vs regla: {m_ens['FP']} vs {m_rule['FP']} → {red_own * 100:.1f} % menos "
          f"({'cumple' if meets else 'NO cumple'} el 30 %). A igual Recall: {red_eq * 100:.1f} % menos.")
    print(f"\nReportes en {REPORTS}")
    if not args.no_save_config:
        print("⚠️  Se actualizó models/config.json → súbelo al repo y avisa a Nicolás para redesplegar.")


if __name__ == '__main__':
    main()
