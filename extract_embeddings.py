import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
from pathlib import Path
import numpy as np


# -----------------------------------
# 1. Load pretrained ResNet50
# -----------------------------------

weights = ResNet50_Weights.DEFAULT

model = models.resnet50(weights=weights)

# Remove classification layer
model.fc = torch.nn.Identity()

# Evaluation mode
model.eval()

print("ResNet50 loaded successfully")


# -----------------------------------
# 2. Image preprocessing
# -----------------------------------

preprocess = weights.transforms()


# -----------------------------------
# 3. Find product images
# -----------------------------------

product_folder = Path("data/products")

image_paths = []

for extension in ["*.jpg", "*.jpeg", "*.png"]:
    image_paths.extend(product_folder.glob(extension))

print("Number of product images:", len(image_paths))


# -----------------------------------
# 4. Extract embeddings
# -----------------------------------

embeddings = []

for image_path in image_paths:

    print("Processing:", image_path.name)

    try:
        image = Image.open(image_path).convert("RGB")

        input_tensor = preprocess(image)

        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = model(input_batch)

        # Normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)

        embeddings.append(embedding.squeeze(0).numpy())

    except Exception as e:
        print("Error processing:", image_path.name)
        print(e)


# -----------------------------------
# 5. Convert to NumPy array
# -----------------------------------

embeddings = np.array(embeddings, dtype="float32")

print("Final embeddings shape:", embeddings.shape)


# -----------------------------------
# 6. Save embeddings
# -----------------------------------

output_folder = Path("embeddings")

output_folder.mkdir(exist_ok=True)

np.save(output_folder / "product_embeddings.npy", embeddings)

np.save(
    output_folder / "product_paths.npy",
    np.array([str(path) for path in image_paths])
)

print("Embeddings saved successfully!")