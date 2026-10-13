import pandas as pd, numpy as np, joblib, os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = None

    def process(self, filepath):
        df = pd.read_csv(filepath)
        # Feature Engineering
        df['Hour'] = (df['Time'] // 3600) % 24
        df['DayOfWeek'] = (df['Time'] // 86400) % 7
        df['LogAmount'] = np.log1p(df['Amount'])

        base = [c for c in df.columns if c.startswith('V')]
        self.features = base + ['Hour', 'DayOfWeek', 'LogAmount']

        X = df[self.features]
        y = df['Class']

        X_scaled = self.scaler.fit_transform(X)
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')

        return train_test_split(pd.DataFrame(X_scaled, columns=self.features), y, test_size=0.2, random_state=42, stratify=y)
