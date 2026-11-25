RANDOM_STATE = 42 # zapewnia powtarzalnosc losowania
CV_SEED = 2026
TEST_SIZE = 0.2 #odsiewa do zbioru testowego 20% danych żeby model był sprawdzany na innym zbiorze, takim, ktory widzi pierwszy raz

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    """ Wczytuje zbiór Wine. Zwraca X (cechy) i y (klasy) """
    data = load_wine(as_frame=True)
    X = data['data']
    y = data['target']
    return X, y

def make_split(X, y):
    """Dzieli dane na development i test. Zwraca indeksy obu części."""
    indeksy = list(range(len(X)))
    dev_idx, test_idx = train_test_split(
        indeksy,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return dev_idx, test_idx

def save_split(dev_idx, test_idx, path="results/splits.csv"):
    """Zapisuje przynależność każdej próbki do development lub testu."""
    splits = pd.DataFrame({
        "sample_id": list(dev_idx) + list(test_idx),
        "set": ["dev"] * len(dev_idx) + ["test"] * len(test_idx),
    })
    splits = splits.sort_values("sample_id")
    splits.to_csv(path, index=False)
    return splits

if __name__ == "__main__":
    X, y = load_data()
    print(X.shape, y.shape)

    dev_idx, test_idx = make_split(X, y)
    print(len(dev_idx), len(test_idx))
    print(y.iloc[dev_idx].value_counts().sort_index().tolist())
    print(y.iloc[test_idx].value_counts().sort_index().tolist())

    splits = save_split(dev_idx, test_idx)
    print(splits.head())
    print(splits["set"].value_counts())
