import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class RandomForestPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.le_time = LabelEncoder()
        self.le_resultado = LabelEncoder()

    def traine(self, path):
        df = pd.read_csv(path)
        self.applyPreProcess(df)
    
    def applyPreProcess(self, df: pd.DataFrame):
        # Rotula target com base no vencedor
        df['vencedor'] = df.apply(self.rotulate_result, axis=1)

        # Codifica times como números
        df['mandante'] = self.le_time.fit_transform(df['mandante'])
        df['visitante'] = self.le_time.transform(df['visitante'])  # usa o mesmo encoder

        # Codifica coluna vencedor como número
        df['vencedor'] = self.le_resultado.fit_transform(df['vencedor'])

        x = df[['mandante', 'visitante']]
        y = df['vencedor']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(df)
        print(f"Acurácia: {acc:.2f}")

    def rotulate_result(self, row):
        if row['vencedor'] == row['mandante']:
            return "casa"
        elif row['vencedor'] == row['visitante']:
            return "fora"
        else:
            return "empate"


