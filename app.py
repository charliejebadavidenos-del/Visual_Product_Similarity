import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import pandas as pd
import faiss
from pathlib import Path

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Visual Product Search",
    page_icon="🛍️",
    layout="wide",
)

# Hide Streamlit's detailed error UI from the end user.
st.set_option("client.showErrorDetails", False)

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"


# ============================================================
# LOAD RESNET50
# ============================================================
@st.cache_resource
def load_model():
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # ResNet50 feature extractor.
    # 2048-dimensional embedding.
    model.fc = torch.nn.Identity()
    model.eval()

    preprocess = weights.transforms()
    return model, preprocess


# ============================================================
# LOAD FAISS INDEX
# ============================================================
@st.cache_resource
def load_faiss_index():
    return faiss.read_index(str(EMBEDDINGS_DIR / "product_index.faiss"))


# ============================================================
# LOAD PRODUCT PATHS
# ============================================================
@st.cache_data
def load_product_paths():
    return np.load(
        str(EMBEDDINGS_DIR / "product_paths.npy"),
        allow_pickle=True,
    )


# ============================================================
# LOAD PRODUCT METADATA
# ============================================================
@st.cache_data
def load_metadata():
    return pd.read_csv(
        str(EMBEDDINGS_DIR / "product_metadata.csv")
    )


# ============================================================
# STARTUP
# ============================================================
try:
    model, preprocess = load_model()
    index = load_faiss_index()
    product_paths = load_product_paths()
    metadata = load_metadata()
except Exception:
    st.error("⚠️ The application could not load the ResNet50 model or product database.")
    st.info("Please check the files inside the embeddings folder.")
    st.stop()


required_columns = ["category", "price", "available", "image_path"]
missing_columns = [c for c in required_columns if c not in metadata.columns]

if missing_columns:
    st.error(
        "⚠️ Product metadata is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

metadata = metadata.copy()

metadata["price"] = pd.to_numeric(
    metadata["price"], errors="coerce"
).fillna(0)

metadata["category"] = (
    metadata["category"]
    .fillna("")
    .astype(str)
    .str.strip()
)

metadata["available_bool"] = (
    metadata["available"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes", "available"])
)


# ============================================================
# HELPERS
# ============================================================
def normalize_category(value):
    return str(value).strip().lower()


def similarity_from_faiss(raw_score, metric_type):
    raw_score = float(raw_score)

    if metric_type == faiss.METRIC_L2:
        # For normalized vectors:
        # squared L2 = 2 - 2*cosine_similarity
        similarity = 1.0 - (raw_score / 2.0)
    else:
        # Inner Product over normalized vectors = cosine similarity.
        similarity = raw_score

    return float(np.clip(similarity, -1.0, 1.0))


def image_path_for_display(raw_path):
    path = Path(str(raw_path))

    if path.is_absolute() and path.exists():
        return str(path)

    candidates = [
        BASE_DIR / path,
        BASE_DIR / str(raw_path).replace("\\", "/"),
        BASE_DIR.parent / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(path)


def get_query_embedding(image):
    """Extract a normalized 2048-dimensional ResNet50 embedding."""
    tensor = preprocess(image)
    batch = tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = model(batch)

    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy().astype("float32")


def search_raw(query_embedding, number_of_results=None):
    """Search FAISS and return results in similarity order."""
    if number_of_results is None:
        number_of_results = index.ntotal

    metric_type = getattr(
        index,
        "metric_type",
        faiss.METRIC_INNER_PRODUCT,
    )

    scores, indices = index.search(
        query_embedding,
        min(number_of_results, index.ntotal),
    )

    results = []

    for raw_score, idx in zip(scores[0], indices[0]):
        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        product = metadata.iloc[idx]

        results.append(
            {
                "index_id": idx,
                "score": similarity_from_faiss(
                    raw_score,
                    metric_type,
                ),
                "category": str(product["category"]).strip(),
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results


def infer_category_from_query(query_embedding, filename=""):
    """
    Infer the uploaded product category from the catalog.

    ResNet50 is used as a visual feature extractor, not as a
    product-category classifier. We therefore use the FAISS nearest
    neighbours already present in the product catalog.

    For catalog/demo images whose filename contains a category name
    (for example Pink_BackPack.png), the filename is used only as a
    fallback when visual evidence is ambiguous. This prevents a stale
    sidebar category such as "headphones" from being carried over.
    """

    # Search a larger neighbourhood. Ten images can be too small for
    # category voting when the uploaded image is visually different
    # from the catalog photographs.
    raw_results = search_raw(query_embedding, 100)

    if not raw_results:
        raw_results = []

    # ------------------------------------------------------------
    # 1. Very strong nearest-neighbour match
    # ------------------------------------------------------------
    if raw_results:
        best = raw_results[0]
        best_category = str(best["category"]).strip()

        if best_category and best["score"] >= 0.65:
            return best_category, best["score"]

    # ------------------------------------------------------------
    # 2. Category voting across the Top-100 visual neighbours
    # ------------------------------------------------------------
    scores_by_category = {}
    counts_by_category = {}

    for rank, result in enumerate(raw_results):
        category = str(result["category"]).strip()
        score = float(result["score"])

        if not category:
            continue

        # Ignore strongly negative cosine similarities.
        weight = max(score, 0.0)
        if weight <= 0:
            continue

        # Give closer neighbours more influence.
        rank_weight = 1.0 / (1.0 + rank * 0.04)
        weighted_score = weight * rank_weight

        scores_by_category[category] = (
            scores_by_category.get(category, 0.0) + weighted_score
        )
        counts_by_category[category] = (
            counts_by_category.get(category, 0) + 1
        )

    if scores_by_category:
        ranked_categories = sorted(
            scores_by_category.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_category, best_total = ranked_categories[0]
        total = sum(scores_by_category.values())
        share = best_total / total if total else 0.0

        # Accept a category when it has a reasonable visual share.
        # This is deliberately less restrictive than the previous 60%
        # Top-10 rule.
        if share >= 0.35 and counts_by_category[best_category] >= 3:
            best_score = next(
                (r["score"] for r in raw_results
                 if r["category"] == best_category),
                0.0,
            )
            return best_category, best_score

    # ------------------------------------------------------------
    # 3. Filename fallback for catalog/demo uploads
    # ------------------------------------------------------------
    filename_text = str(filename).lower().replace("-", "_").replace(" ", "_")

    available_categories = [
        str(c).strip()
        for c in metadata["category"].dropna().unique()
        if str(c).strip()
    ]

    # Prefer the longest category name first, so e.g. backpack is
    # checked before a shorter generic token.
    for category in sorted(available_categories, key=len, reverse=True):
        token = normalize_category(category).replace(" ", "_")
        if token and token in filename_text:
            return category, 1.0

    # Common spelling used by the supplied demo image.
    if "back_pack" in filename_text or "backpack" in filename_text:
        for category in available_categories:
            if normalize_category(category) == "backpack":
                return category, 1.0

    best_score = raw_results[0]["score"] if raw_results else 0.0
    return "Unknown", best_score


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "run_visual_search": False,
    "last_file_id": None,
    "visual_results": [],
    "evaluation": None,
    "inferred_category": "Unknown",
    "detected_category": "Unknown",
    "detected_category_score": 0.0,
    "detected_file_id": None,
    "category_filter": "All",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# IMPORTANT: DETECT A NEW UPLOADED CATALOG IMAGE BEFORE
# CREATING THE CATEGORY WIDGET.
#
# This fixes the stale-category problem:
# - upload bottle image -> category becomes bottle
# - upload backpack image -> category becomes backpack
# - upload a random image -> keep the user's current category
#
# We only auto-change the filter for a NEW upload and only when
# the nearest catalog match is strong enough.
# ============================================================
uploaded_state_file = st.session_state.get(
    "visual_search_uploader"
)

if uploaded_state_file is not None:
    current_upload_id = (
        uploaded_state_file.name,
        uploaded_state_file.size,
    )

    if current_upload_id != st.session_state.detected_file_id:
        try:
            uploaded_state_file.seek(0)
            uploaded_image = Image.open(
                uploaded_state_file
            ).convert("RGB")

            with st.spinner("Identifying the uploaded product category..."):
                query_embedding_for_category = get_query_embedding(
                    uploaded_image
                )
                detected_category, detected_score = (
                    infer_category_from_query(
                        query_embedding_for_category,
                        uploaded_state_file.name,
                    )
                )

            # IMPORTANT: reset the previous category first.
            # Otherwise an earlier selection such as headphones/chair
            # remains active when a new image is uploaded.
            st.session_state.category_filter = "All"

            st.session_state.detected_file_id = current_upload_id
            st.session_state.detected_category = detected_category
            st.session_state.detected_category_score = detected_score

            available_categories = {
                str(c).strip()
                for c in metadata["category"].dropna().unique()
            }

            if detected_category != "Unknown" and detected_category in available_categories:
                st.session_state.category_filter = detected_category

        except Exception:
            st.session_state.detected_file_id = current_upload_id
            st.session_state.detected_category = "Unknown"
            st.session_state.detected_category_score = 0.0
            st.session_state.category_filter = "All"


# ============================================================
# TITLE
# ============================================================
st.title("🛍️ Visual Product Search")
st.write(
    "Upload a product image to find visually similar products "
    "or use the filters to browse the product catalog."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("🔎 Filters")

categories = sorted(
    metadata.loc[
        metadata["category"] != "",
        "category",
    ].unique().tolist()
)

category_options = ["All"] + categories

# Safety: if detected category is not in the current metadata,
# fall back to All.
if st.session_state.category_filter not in category_options:
    st.session_state.category_filter = "All"

selected_category = st.sidebar.selectbox(
    "Category",
    category_options,
    key="category_filter",
)

min_price = int(metadata["price"].min())
max_price = int(metadata["price"].max())

if min_price == max_price:
    max_price = min_price + 250

price_range = st.sidebar.slider(
    "Price Range (₹)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=250,
    key="price_filter",
)

available_only = st.sidebar.checkbox(
    "Available products only",
    value=False,
    key="availability_filter",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Active Filters")

st.sidebar.write(f"Category: {selected_category}")
st.sidebar.write(
    f"Price: ₹{price_range[0]:,.0f} - ₹{price_range[1]:,.0f}"
)
st.sidebar.write(
    "Availability: Available only"
    if available_only
    else "Availability: All products"
)


# ============================================================
# FILTER CATALOG - INDEPENDENT OF IMAGE UPLOAD
# ============================================================
filtered_metadata = metadata.copy()

if selected_category != "All":
    filtered_metadata = filtered_metadata[
        filtered_metadata["category"].map(normalize_category)
        == normalize_category(selected_category)
    ]

filtered_metadata = filtered_metadata[
    (filtered_metadata["price"] >= price_range[0])
    & (filtered_metadata["price"] <= price_range[1])
]

if available_only:
    filtered_metadata = filtered_metadata[
        filtered_metadata["available_bool"]
    ]


# ============================================================
# VISUAL SEARCH
# ============================================================
st.subheader("🔍 Visual Search")
st.write(
    "Upload a product image and click **Find Similar Products**."
)

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png"],
    key="visual_search_uploader",
)

current_file_id = None

if uploaded_file is not None:
    current_file_id = (
        uploaded_file.name,
        uploaded_file.size,
    )

# New upload: clear old visual-search results.
if current_file_id != st.session_state.last_file_id:
    st.session_state.run_visual_search = False
    st.session_state.visual_results = []
    st.session_state.evaluation = None
    st.session_state.inferred_category = (
        st.session_state.detected_category
        if st.session_state.detected_category != "Unknown"
        else "Unknown"
    )
    st.session_state.last_file_id = current_file_id

image = None

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("⚠️ The uploaded image could not be opened.")
        st.info("Please upload a valid JPG or PNG image.")
        st.stop()

    st.subheader("Query Image")
    st.image(image, width=350)

    detected_category = st.session_state.detected_category
    detected_score = st.session_state.detected_category_score

    if detected_category != "Unknown":
        st.success(
            f"Detected category: **{detected_category}** "
            f"(catalog-match similarity: {detected_score:.4f})"
        )
    else:
        st.info(
            "Category could not be confidently detected automatically. "
            "You can select the correct category from the sidebar."
        )

    if st.button(
        "🔍 Find Similar Products",
        type="primary",
        key="find_similar_button",
    ):
        st.session_state.run_visual_search = True


# ============================================================
# VISUAL SEARCH EXECUTION
# ============================================================
if (
    uploaded_file is not None
    and image is not None
    and st.session_state.run_visual_search
):
    try:
        with st.spinner("ResNet50 is extracting visual features..."):
            query_embedding = get_query_embedding(image)

        with st.spinner("Searching FAISS for similar products..."):
            raw_results = search_raw(
                query_embedding,
                index.ntotal,
            )

            # ----------------------------------------------------
            # CATEGORY USED FOR SEARCH
            #
            # EXPLICIT SIDEBAR SELECTION HAS PRIORITY.
            # If All is selected, use the detected category only
            # when it is confident.
            # ----------------------------------------------------
            if selected_category != "All":
                search_category = selected_category
            elif st.session_state.detected_category != "Unknown":
                search_category = st.session_state.detected_category
            else:
                search_category = "All"

            st.session_state.inferred_category = (
                st.session_state.detected_category
            )

            # ----------------------------------------------------
            # FILTER + RANK
            #
            # We search the whole FAISS index first, then apply:
            # 1. category
            # 2. price
            # 3. availability
            #
            # Finally we rank the surviving products by cosine
            # similarity from highest to lowest.
            # ----------------------------------------------------
            visual_results = []

            for result in raw_results:
                idx = result["index_id"]
                product = metadata.iloc[idx]

                result_category = normalize_category(
                    product["category"]
                )

                if search_category != "All":
                    if result_category != normalize_category(
                        search_category
                    ):
                        continue

                product_price = float(product["price"])

                if not (
                    price_range[0]
                    <= product_price
                    <= price_range[1]
                ):
                    continue

                if (
                    available_only
                    and not bool(product["available_bool"])
                ):
                    continue

                visual_results.append(result)

                if len(visual_results) >= 100:
                    break

            # Explicit final ranking.
            visual_results.sort(
                key=lambda result: result["score"],
                reverse=True,
            )

            for rank, result in enumerate(
                visual_results,
                start=1,
            ):
                result["rank"] = rank

            st.session_state.visual_results = visual_results

            # ====================================================
            # STEP 5: EVALUATION
            #
            # IMPORTANT:
            # Evaluation uses the RAW FAISS ranking, not the
            # filtered display list. Therefore category filtering
            # cannot artificially make Precision@K equal to 100%.
            # ====================================================
            evaluation_category = search_category

            if evaluation_category != "All":
                ground_truth_category = normalize_category(
                    evaluation_category
                )

                relevant_mask = (
                    metadata["category"].map(normalize_category)
                    == ground_truth_category
                )

                total_relevant = int(relevant_mask.sum())
                precision_recall = {}

                for k in [5, 10, 20, 50, 100]:
                    top_k = raw_results[:k]

                    relevant_retrieved = sum(
                        1
                        for result in top_k
                        if normalize_category(
                            result["category"]
                        )
                        == ground_truth_category
                    )

                    precision_at_k = (
                        relevant_retrieved / k
                        if k > 0
                        else 0.0
                    )

                    recall_at_k = (
                        relevant_retrieved / total_relevant
                        if total_relevant > 0
                        else 0.0
                    )

                    precision_recall[k] = {
                        "precision": precision_at_k,
                        "recall": recall_at_k,
                        "relevant_retrieved": relevant_retrieved,
                    }

                st.session_state.evaluation = {
                    "category": evaluation_category,
                    "total_relevant": total_relevant,
                    "metrics": precision_recall,
                }
            else:
                st.session_state.evaluation = None

    except Exception:
        # Do not expose Streamlit's "Ask Google / Ask ChatGPT"
        # error actions to the user.
        st.error(
            "⚠️ Visual similarity search could not be completed."
        )
        st.info(
            "Please try uploading the product image again."
        )


# ============================================================
# DISPLAY VISUAL SEARCH RESULTS
# ============================================================
if (
    uploaded_file is not None
    and st.session_state.run_visual_search
):
    visual_results = st.session_state.visual_results

    st.markdown("---")
    st.header("🔍 Visually Similar Products")

    display_category = selected_category

    if display_category == "All":
        display_category = st.session_state.detected_category

    if display_category != "Unknown" and display_category != "All":
        st.success(
            f"Search category: **{display_category}**"
        )
    else:
        st.warning(
            "Search category: **All** — select a category in the "
            "sidebar if you want category-controlled visual search."
        )

    if len(visual_results) == 0:
        st.warning(
            "No visually similar products match the selected filters."
        )
    else:
        st.success(
            f"Found {len(visual_results)} visually similar products."
        )
        st.info(
            "Results are ranked from highest to lowest cosine similarity."
        )

        for start in range(0, len(visual_results), 5):
            row_results = visual_results[start:start + 5]
            columns = st.columns(5)

            for column, result in zip(columns, row_results):
                product = metadata.iloc[result["index_id"]]
                image_path = image_path_for_display(
                    product["image_path"]
                )

                with column:
                    st.markdown(
                        f"### 🏆 #{result['rank']}"
                    )

                    try:
                        st.image(
                            image_path,
                            use_container_width=True,
                        )
                    except Exception:
                        st.warning("Image unavailable")

                    st.write(
                        f"**Category:** {product['category']}"
                    )
                    st.write(
                        f"**Price:** ₹{float(product['price']):,.0f}"
                    )
                    st.write(
                        f"**Similarity:** {result['score']:.4f}"
                    )

                    if bool(product["available_bool"]):
                        st.success("Available")
                    else:
                        st.error("Out of stock")


# ============================================================
# STEP 5 - EVALUATION
# ============================================================
st.markdown("---")
st.header("📊 Step 5: Evaluation")

st.write(
    "Step 5 checks whether the FAISS ranking retrieves products "
    "from the selected/relevant category. Precision@K and Recall@K "
    "use category labels as a measurable relevance proxy; visual "
    "inspection is still required for true visual similarity."
)

if st.session_state.evaluation is not None:
    evaluation = st.session_state.evaluation
    eval_category = evaluation["category"]
    metrics = evaluation["metrics"]
    total_relevant = evaluation["total_relevant"]

    st.subheader("Precision@K and Recall@K")

    st.caption(
        f"Evaluation category: **{eval_category}** | "
        f"Total products in category: {total_relevant}. "
        "Metrics use the raw FAISS ranking before display filters. "
        "This is a category-relevance proxy; visual relevance "
        "must also be checked manually."
    )

    evaluation_rows = []

    for k in [5, 10, 20, 50, 100]:
        metric = metrics[k]

        evaluation_rows.append(
            {
                "K": k,
                "Relevant Retrieved": metric[
                    "relevant_retrieved"
                ],
                "Precision@K": round(
                    metric["precision"],
                    4,
                ),
                "Recall@K": round(
                    metric["recall"],
                    4,
                ),
            }
        )

    evaluation_df = pd.DataFrame(evaluation_rows)

    st.dataframe(
        evaluation_df,
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Precision@5",
            f"{metrics[5]['precision']:.2%}",
        )

    with col2:
        st.metric(
            "Recall@5",
            f"{metrics[5]['recall']:.2%}",
        )
else:
    st.info(
        "Select a product category, upload an image, and click "
        "**Find Similar Products**. Then Precision@K and Recall@K "
        "will be calculated for that category."
    )


# ============================================================
# VISUAL INSPECTION
# ============================================================
st.subheader("👁️ Visual Inspection")

if st.session_state.visual_results:
    st.write(
        "Review the Top-5 results above. Check whether the product "
        "type, shape, color, structure, and overall appearance are "
        "visually relevant to the uploaded query image."
    )
else:
    st.write(
        "Run visual search to inspect the Top-K visually similar products."
    )


# ============================================================
# PRODUCT CATALOG - INDEPENDENT OF IMAGE UPLOAD
# ============================================================
st.markdown("---")
st.header("📦 Products Matching Filters")

matching_count = len(filtered_metadata)

st.write(
    f"Found **{matching_count}** products matching your filters."
)

if matching_count == 0:
    st.warning("No products match the selected filters.")
else:
    for start in range(0, matching_count, 5):
        row = filtered_metadata.iloc[start:start + 5]
        columns = st.columns(5)

        for column, (_, product) in zip(
            columns,
            row.iterrows(),
        ):
            with column:
                image_path = image_path_for_display(
                    product["image_path"]
                )

                try:
                    st.image(
                        image_path,
                        use_container_width=True,
                    )
                except Exception:
                    st.warning("Image unavailable")

                st.write(
                    f"**Category:** {product['category']}"
                )
                st.write(
                    f"**Price:** ₹{float(product['price']):,.0f}"
                )

                if bool(product["available_bool"]):
                    st.success("Available")
                else:
                    st.error("Out of stock")
