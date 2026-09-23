import numpy as np

embeddings = np.load("embeddings/product_embeddings.npy")

paths = np.load("embeddings/product_paths.npy")

print("Embedding shape:", embeddings.shape)
print("Number of image paths:", len(paths))

print("\nFirst embedding:")
print(embeddings[0])

print("\nFirst image:")
print(paths[0])