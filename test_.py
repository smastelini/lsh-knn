from river import evaluate
from river import metrics
from river import neighbors
from river import preprocessing
from river import synth

# from radius_neighbors import RadiusNeighborsRegressor
from k_nearest_neighbors import KNNRegressor

dataset = iter(synth.Friedman(seed=1).take(50000))

# model = (
#     preprocessing.StandardScaler() |
#     neighbors.KNNRegressor(window_size=1000)
# )

# model = (
#     preprocessing.StandardScaler() |
#     RadiusNeighborsRegressor(max_size=1000, r=1, aggregation='distance',
#                              delta=0.1, seed=42, k=4)
# )

model = (
    preprocessing.StandardScaler() |
    KNNRegressor(
        n_neighbors=3, window_size=1000, r=2, c=5, aggregation_method='mean',
        delta=0.1, seed=42, k=4, w=4
    )
)
print(f'L={model["KNNRegressor"]._buffer.L}')
print(f'P1={model["KNNRegressor"]._buffer.P1}')
print(f'P2={model["KNNRegressor"]._buffer.P2}')


# exit()

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric, print_every=100, show_time=True,
                               show_memory=True)
