**********************************************************************
Architecture Diagram :
**********************************************************************
 VISUAL PRODUCT SEARCH
                           │
             ┌─────────────┴─────────────┐
             │                           │
       PRODUCT CATALOG              IMAGE SEARCH
             │                           │
     Category / Price /         Upload image
     Availability                     │
             │                    ResNet50
             │                       │
             │                  2048 embedding
             │                       │
             │                     FAISS
             │                       │
             │               Category-filtered
             │                   candidates
             │                       │
             │                  Similarity
             │                       │
             │                  Top 100
             │                       │
             └───────────────┬───────┘
                             │
                       STEP 5
                       Evaluation
                         │
                  ┌──────┼──────┐
                  │      │      │
             Precision Recall Visual
                @K      @K   Inspection

**********************************************************************
Application Working
**********************************************************************
User selects category
example 

Category
[ backpack ▼ ]

Uploaded image
      ↓
ResNet50
      ↓
FAISS
      ↓
ONLY backpack products
      ↓
Rank by similarity
      ↓
Top 100

**********************************************************************
Similarity Ranking 
**********************************************************************

This ranking of Product based on the Similarity value.

Backpack image
      ↓
ResNet50
      ↓
2048 embedding
      ↓
FAISS
      ↓
Backpack candidates
      ↓
Cosine similarity
      ↓
Highest → Lowest
      ↓
Top 100 


Step 1  → Dataset
Step 2  → ResNet50 Feature Extraction
Step 3  → FAISS Similarity Matching
Step 4  → Ranking & Filtering
Step 5  → Evaluation
             ├── Precision@K
             ├── Recall@K
             └── Visual Inspection
			 
**********************************************************************
Precision and Recall 
**********************************************************************	
Precision@K= Relevant products in Top-K / K

Recall@K=Relevant products retrieved​ / Total relevant products

			 
**********************************************************************
Visual Inspection
**********************************************************************			 
			 
Query Image
     ↓
Top-K FAISS results
     ↓
Are they visually similar?
     ↓
Are they the correct product category?
     ↓
Are similarity scores properly ranked?

**********************************************************************	
Products Matching Filters
**********************************************************************	
Active Filters are displayed based on the Chosen Criteria in filters

🔎 Filters
      Category - Specify all products are the products from list
      Price Range (₹) - Price range in scrolling bar to be selected
      Availability: If not checked out of stock products will also appear

Active Filters
      Category: chair
      Price: ₹1,750 - ₹3,250
      Availability: Available only

