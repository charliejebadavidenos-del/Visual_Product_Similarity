import pandas as pd
import numpy as np
from pathlib import Path


# Load product paths
paths = np.load(
    "embeddings/product_paths.npy"
)


records = []

for i, path in enumerate(paths):

    filename = Path(path).stem

    # Category from filename
    category = filename.split("_")[0].lower()

    # DEMO price only
    demo_price = 500 + (i % 20) * 250

    # DEMO availability
    demo_available = (i % 5 != 0)

    records.append({
        "image_path": path,
        "category": category,
        "price": demo_price,
        "available": demo_available
    })


df = pd.DataFrame(records)


df.to_csv(
    "embeddings/product_metadata.csv",
    index=False
)


print("Metadata created successfully!")
print("Products:", len(df))
print("\nCategories:")
print(df["category"].value_counts())

print("\nPrice range:")
print(df["price"].min(), "to", df["price"].max())

print("\nAvailability:")
print(df["available"].value_counts())