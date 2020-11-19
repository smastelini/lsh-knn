from river import evaluate
from river import metrics
from river import neighbors
from river import preprocessing
from river import synth

from radius_neighbors import RadiusNeighborsRegressor

dataset = iter(synth.Friedman(seed=1).take(50000))

# model = (
#     preprocessing.StandardScaler() |
#     neighbors.KNNRegressor(window_size=1000)
# )

model = (
    preprocessing.StandardScaler() |
    RadiusNeighborsRegressor(max_size=1000, r=1, aggregation='distance',
                             delta=0.1, seed=42, k=4)
)

metric = metrics.MAE()

evaluate.progressive_val_score(dataset, model, metric, print_every=100, show_time=True,
                               show_memory=True)


# print(model['RadiusNeighborsRegressor']._buffer.L)
# print(model['RadiusNeighborsRegressor']._buffer.success_probability)
