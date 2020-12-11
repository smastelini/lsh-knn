from river import evaluate
from river import metrics
from river import preprocessing
from river import synth
from river import neighbors

from k_nearest_neighbors import KNNClassifier

dataset = iter(synth.LED(irrelevant_features=False, seed=1).take(50000))

model = (
    # preprocessing.StandardScaler() |
    KNNClassifier(n_neighbors=8, window_size=1000,
                  r=1., c=1., seed=42
    )
    # neighbors.KNNClassifier(n_neighbors=8)
)
# print(f'L={model["KNNClassifier"]._buffer.L}')
# print(f'P1={model["KNNClassifier"]._buffer.P1}')
# print(f'P2={model["KNNClassifier"]._buffer.P2}')

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric, print_every=100, show_time=True,
                               show_memory=True)
