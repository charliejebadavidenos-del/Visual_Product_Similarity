import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import faiss


# -----------------------------------
# 1. Load ResNet50
# -----------------------------------

weights = ResNet50_Weights.DEFAULT

model = models.resnet50(weights=weights)

model.fc = torch.nn.Identity()

model.eval()

preprocess = weights.transforms()

print("ResNet50 loaded")


# -----------------------------------
# 2. Load FAISS index
# -----------------------------------

index = faiss.read_index(
    "embeddings/product_index.faiss"
)

print("FAISS index loaded")
print("Products indexed:", index.ntotal)


# -----------------------------------
# 3. Load product paths
# -----------------------------------

product_paths = np.load(
    "embeddings/product_paths.npy"
)

print("Product paths loaded")


# -----------------------------------
# 4. Query image
# -----------------------------------

query_path = "data/query/backpack_0_1771054472729.jpg"

image = Image.open(query_path).convert("RGB")

input_tensor = preprocess(image)

input_batch = input_tensor.unsqueeze(0)


# -----------------------------------
# 5. Extract query embedding
# -----------------------------------

with torch.no_grad():

    query_embedding = model(input_batch)

query_embedding = F.normalize(
    query_embedding,
    p=2,
    dim=1
)


# Convert to NumPy
query_embedding = query_embedding.numpy().astype("float32")


# -----------------------------------
# 6. Search FAISS
# -----------------------------------

k = 5

scores, indices = index.search(
    query_embedding,
    k
)


# -----------------------------------
# 7. Display results
# -----------------------------------

print("\n==============================")
print("SIMILAR PRODUCTS")
print("==============================")

for rank, (score, index_id) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    print(
        f"{rank}. {product_paths[index_id]} "
        f"| Similarity: {score:.4f}"
    )