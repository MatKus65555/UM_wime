RANDOM_STATE = 42 # zapewnia powtarzalnosc losowania
CV_SEED = 2026
TEST_SIZE = 0.2 #odsiewa do zbioru testowego 20% danych żeby model był sprawdzany na innym zbiorze, takim, ktory widzi pierwszy raz

from sklearn.datasets import load_wine

def load_data():
    """ Wczytuje zbiór Wine. Zwraca X (cechy) i y (klasy) """
    data = load_wine(as_frame=True)
    X = data['data']
    y = data['target']
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    print(X.shape, y.shape)
