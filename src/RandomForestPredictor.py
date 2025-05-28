import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

class RandomForestPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.encoder_X = OneHotEncoder(handle_unknown='ignore')  # para mandante e visitante
        self.encoder_y = LabelEncoder()  # para o target vencedor

    def train(self, path):
        df = pd.read_csv(path)
        self.applyPreProcess(df)
    
    def applyPreProcess(self, df: pd.DataFrame):
        # Rotula target com base no vencedor
        df['vencedor'] = df.apply(self.rotulate_result, axis=1)

        # X: codifica mandante e visitante com OneHotEncoder
        X = self.encoder_X.fit_transform(df[['mandante', 'visitante']])

        # y: codifica resultado com LabelEncoder (1D array)
        y = self.encoder_y.fit_transform(df['vencedor'])

        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treina o modelo
        self.model.fit(X_train, y_train)

        # Prediz e calcula acurácia
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Acurácia: {acc:.2f}")

    def rotulate_result(self, row):
        if row['vencedor'] == row['mandante']:
            return "casa"
        elif row['vencedor'] == row['visitante']:
            return "fora"
        else:
            return "empate"
