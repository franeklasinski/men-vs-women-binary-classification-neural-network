# Gender Classification Neural Network

## Klasyfikacja binarna - Rozpoznawanie płci (mężczyzna vs kobieta) z zdjęć

## Opis projektu

Projekt implementuje system klasyfikacji binarnej do rozpoznawania płci na zdjęciach przy użyciu głębokich sieci neuronowych. Porównuje skuteczność dwóch różnych architektur CNN (Convolutional Neural Networks) w zadaniu klasyfikacji obrazów na dwie klasy: mężczyzna i kobieta.

## Technologie i biblioteki

- **PyTorch** - framework do deep learning
- **torchvision** - biblioteka do przetwarzania obrazów
- **TensorBoard** - wizualizacja procesu treningu
- **matplotlib** - wizualizacja wyników
- **numpy** - operacje na tablicach

## Dataset
- Dataset: https://www.kaggle.com/datasets/snmahsa/human-images-dataset-men-and-women 
## Struktura projektu

```
gender_classification/
├── men vs women binary classification neural network.ipynb  # Główny notebook
├── README.md                                               # Ten plik
└── gender_dataset/                                        # Zbiór danych (nie w repo)
    ├── men/                                              # Zdjęcia mężczyzn
    └── women/                                            # Zdjęcia kobiet
```

## Architektura modeli

### 1. Model1
- **Warstwy konwolucyjne:** 2 bloki (64 i 128 filtrów)
- **Regularyzacja:** Batch Normalization, Dropout (0.25, 0.5)
- **Klasyfikator:** Dwie warstwy liniowe (256 → 2)
- **Rozmiar wejściowy:** 64×64 pikseli

### 2. Model2
- **Warstwy konwolucyjne:** 3 bloki (64, 128, 256 filtrów)
- **Regularyzacja:** Batch Normalization, Dropout (0.25, 0.3, 0.5)
- **Klasyfikator:** Trzy warstwy liniowe (512 → 128 → 2)
- **Rozmiar wejściowy:** 64×64 pikseli

## Augmentacja danych

Projekt wykorzystuje następujące techniki augmentacji:
- **Flip poziomy** - RandomHorizontalFlip()
- **Rotacja** - RandomRotation(10°)
- **Zmiany kolorów** - ColorJitter (brightness, contrast, saturation)
- **Normalizacja** - mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

## Wyniki

### Model1 (40 epok, ~60 minut treningu)
- **Accuracy treningowy:** 82.85%
- **Accuracy walidacyjny:** 85.97%
- **Loss treningowy:** 0.4059
- **Loss walidacyjny:** 0.2968

### Model2 (60 epok, ~75 minut treningu)
- **Accuracy treningowy:** 91.67%
- **Accuracy walidacyjny:** 96.04%
- **Loss treningowy:** 0.2147
- **Loss walidacyjny:** 0.1059

## Instalacja i uruchomienie

1. **Klonowanie repozytorium**
```bash
git clone <repository-url>
cd gender-classification
```

2. **Instalacja zależności**
```bash
pip install torch torchvision matplotlib numpy tensorboard
```

3. **Przygotowanie danych**
- Pobierz zbiór danych z obrazami
- Umieść w folderze `gender_dataset/` z podfolderami `men/` i `women/`

4. **Uruchomienie**
```bash
jupyter notebook "men vs women binary classification neural network.ipynb"
```

## Monitorowanie treningu

Projekt wykorzystuje TensorBoard do śledzenia:
- Loss funkcji (trening i walidacja)
- Accuracy (trening i walidacja)
- Wizualizacja krzywych uczenia

```bash
tensorboard --logdir runs
```

## Wymagania systemowe

- Python 3.7+
- PyTorch 1.9+
- CUDA (opcjonalnie, dla przyspieszenia na GPU)
- MPS (dla Apple Silicon)
- Minimum 8GB RAM
- Miejsce na dysku: ~2GB (zbiór danych + modele)

## Autor

**Franciszek Łasiński**  

## Licencja

Projekt edukacyjny - do użytku akademickiego.
