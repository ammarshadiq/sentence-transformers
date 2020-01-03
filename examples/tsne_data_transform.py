import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

csv_data = pd.read_csv('/Users/darmanm/workspace/digitaltwin-experiments/clustering/material_embeddings.csv')
embeddings_data = csv_data.iloc[3:]
metadata = csv_data.iloc[:3]
print(embeddings_data.shape)

print("TSNE Transformation")
tsne = TSNE(n_components=3, n_iter=250)
tsne_result = tsne.fit_transform(embeddings_data)

print("Concating data, writing to CSV")
pd.concat([metadata, tsne_result], axis=1).write_csv('/Users/darmanm/workspace/digitaltwin-experiments/clustering/material_embeddings_tsne.csv')

