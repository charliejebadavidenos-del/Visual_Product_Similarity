import faiss
import numpy as np


# -----------------------------------
# 1. Load product embeddings
# -----------------------------------

embeddings = np.load(
    "embeddings/product_embeddings.npy"
).astype("float32")

print("Embeddings loaded:", embeddings.shape)


# -----------------------------------
# 2. Create FAISS index
# -----------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

print("FAISS index created")


# -----------------------------------
# 3. Add embeddings to FAISS
# -----------------------------------

index.add(embeddings)

print("Number of vectors in index:", index.ntotal)


# -----------------------------------
# 4. Save FAISS index
# -----------------------------------

faiss.write_index(
    index,
    "embeddings/product_index.faiss"
)

print("FAISS index saved successfully!")