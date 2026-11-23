"""Latencia por transacción del ensemble (promesa del Hito 2: < 200 ms).

Uso:  python latency.py            # 1000 predicciones sobre data/muestras.json
      python latency.py --n 3000

Mide `FraudEnsemble.predict` (escalado + RF + IF + fusión), sin HTTP.
La latencia de punta a punta de la API la mide el propio prototipo
(`latencia_ms` en la respuesta de /predict, o el tiempo del navegador).
"""
import argparse
import json
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.ensemble import FraudEnsemble
from src.preprocessing import ROOT_DIR

REPORTS = ROOT_DIR / 'reports'
LIMIT_MS = 200.0


def measure(ens, samples, n, warmup=30):
    for i in range(warmup):                       # calentamiento (carga perezosa, cachés)
        ens.predict(samples[i % len(samples)])
    times = np.empty(n)
    for i in range(n):
        s = samples[i % len(samples)]
        t0 = time.perf_counter()
        ens.predict(s)
        times[i] = (time.perf_counter() - t0) * 1000
    return times


def summary(t):
    return {'media_ms': float(t.mean()), 'p50_ms': float(np.percentile(t, 50)),
            'p95_ms': float(np.percentile(t, 95)), 'p99_ms': float(np.percentile(t, 99)),
            'max_ms': float(t.max()), 'pct_bajo_200ms': float((t < LIMIT_MS).mean() * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    samples = [m['features'] for m in json.load(open(ROOT_DIR / 'data' / 'muestras.json'))]
    ens = FraudEnsemble()

    out = {'n_predicciones': args.n, 'python': platform.python_version(),
           'maquina': platform.processor() or platform.machine()}
    runs = {}
    for label, n_jobs in [('Producción (n_jobs=1)', 1), ('Original (n_jobs=-1)', -1)]:
        ens.rf.n_jobs = n_jobs
        runs[label] = measure(ens, samples, args.n)
        out[label] = summary(runs[label])
    ens.rf.n_jobs = 1

    json.dump(out, open(REPORTS / 'latencia.json', 'w'), indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, t in runs.items():
        ax.hist(t, bins=60, alpha=0.7, label=f"{label}: p95={np.percentile(t, 95):.1f} ms")
    ax.axvline(LIMIT_MS, color='r', ls='--', label='límite prometido: 200 ms')
    ax.set(xlabel='latencia por transacción (ms)', ylabel='frecuencia', title='Latencia de predict()')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / 'latencia.png', dpi=150)
    plt.close(fig)

    print(f"{args.n} predicciones por configuración · Python {out['python']}")
    for label in runs:
        s = out[label]
        print(f"\n{label}\n  media {s['media_ms']:.1f} ms | p50 {s['p50_ms']:.1f} | p95 {s['p95_ms']:.1f} "
              f"| p99 {s['p99_ms']:.1f} | máx {s['max_ms']:.1f} | {s['pct_bajo_200ms']:.1f} % < 200 ms")
    ok = out['Producción (n_jobs=1)']['p99_ms'] < LIMIT_MS
    print(f"\nPromesa < 200 ms por transacción (p99): {'SE CUMPLE' if ok else 'NO se cumple'}")
    print(f"Reportes en {REPORTS}")


if __name__ == '__main__':
    main()
