# Research Paper Recommendation & Subject Area Prediction

## 1. Project in One Line

A machine-learning application that recommends the **Top-5 contextually similar research papers** using Sentence Transformers and predicts a paper's subject area from its abstract using a multi-label neural network.

---

## 2. Interview Introduction

> “I developed a research-paper recommendation and subject-area prediction system using Python, TensorFlow, Sentence Transformers and Streamlit. The system converts paper titles into semantic embeddings and compares them using cosine similarity to return the Top-5 related papers. It also processes paper abstracts using TF-IDF bigram vectorization and an MLP-based multi-label classifier to predict relevant subject areas. I worked with 500+ paper records and focused on improving recommendation relevance and reducing manual paper-search effort.”

Keep this answer around **45–60 seconds**.

---

## 3. Why I Chose This Project

Researchers and students often spend significant time manually searching for relevant papers. Traditional keyword search can miss papers that use different words for the same concept.

This project was selected to:

- recommend papers based on **meaning**, not only exact keywords;
- reduce manual searching and literature-review effort;
- classify a paper into one or more subject areas;
- practically apply NLP, embeddings, similarity search and deep learning.

---

## 4. Complete Workflow

```text
Dataset
  ↓
Data cleaning and duplicate removal
  ↓
Titles + summaries + subject labels
  ↓
┌──────────────────────────┬──────────────────────────────┐
│ Recommendation pipeline  │ Subject prediction pipeline  │
├──────────────────────────┼──────────────────────────────┤
│ Paper title              │ Paper abstract               │
│ Sentence Transformer     │ TextVectorization            │
│ Dense embedding          │ TF-IDF bigram vector         │
│ Cosine similarity        │ MLP neural network           │
│ Top-5 similar papers     │ Predicted subject labels     │
└──────────────────────────┴──────────────────────────────┘
  ↓
Streamlit interface
```

### Recommendation flow

1. Clean paper titles and remove duplicates.
2. Load `all-MiniLM-L6-v2`.
3. Convert every title into a numerical embedding.
4. Convert the user's title into an embedding.
5. Calculate cosine similarity with all stored embeddings.
6. Use `torch.topk()` to return the five highest-scoring papers.

### Subject-prediction flow

1. Take the research-paper abstract.
2. Convert text into TF-IDF bigram features using `TextVectorization`.
3. Pass the vector through an MLP with dropout.
4. Apply sigmoid output because multiple labels may be correct.
5. Convert the predicted multi-hot vector back into subject names.

---

## 5. Main Concepts Used

### Sentence Transformer

A Sentence Transformer converts text into a dense vector called an **embedding**. Texts with similar meanings receive vectors that are closer in vector space.

Model used:

```text
all-MiniLM-L6-v2
```

Why used:

- lightweight and fast;
- captures semantic meaning;
- suitable for sentence/title similarity;
- better than exact keyword matching.

### Embedding

An embedding is a numerical representation of text. Instead of comparing words directly, the project compares the vectors representing their meanings.

### Cosine Similarity

Cosine similarity measures the angle between two vectors:

```text
cosine similarity = (A · B) / (||A|| × ||B||)
```

Its value is generally between `-1` and `1`; for these text embeddings, a higher value means stronger semantic similarity.

### TF-IDF

TF-IDF gives higher importance to words that are frequent in one document but less common across all documents.

- **TF:** importance of a term inside a document.
- **IDF:** reduces the weight of overly common terms.

### Bigram

A bigram is a pair of consecutive words, such as:

```text
machine learning
neural network
computer vision
```

Bigrams preserve more context than individual words.

### MLP

A Multi-Layer Perceptron is a feed-forward neural network containing:

- input features;
- dense hidden layers;
- activation functions;
- dropout;
- output layer.

### Multi-label classification

One research paper can belong to multiple subjects. Therefore, the output uses:

```text
Sigmoid activation + Binary Cross-Entropy
```

It does not use softmax because softmax normally assumes only one class is correct.

### Multi-hot encoding

Example:

```text
Available labels: [AI, ML, Networks, Security]
Paper labels:     [AI, ML]

Encoded output:   [1, 1, 0, 0]
```

---

## 6. Technology Stack

| Area | Technology | Purpose |
|---|---|---|
| Language | Python | Data processing and model development |
| Data | Pandas, NumPy | Cleaning and transformation |
| Recommendation | Sentence Transformers | Semantic text embeddings |
| Similarity | Cosine similarity | Ranking related papers |
| Deep learning | TensorFlow/Keras | Subject-area classifier |
| Tensor operations | PyTorch | Top-K similarity retrieval |
| Evaluation | Scikit-learn | Splitting and metrics |
| Model storage | Pickle, H5 | Saving embeddings and trained models |
| Interface | Streamlit | User-facing application |

---

## 7. Dataset and Data Preparation

The dataset contains:

```text
titles
summaries
terms
```

- `titles`: research-paper titles;
- `summaries`: abstracts or summaries;
- `terms`: one or more subject labels.

Preprocessing included:

- checking missing values;
- removing duplicate titles;
- converting string labels into Python lists;
- filtering labels with insufficient samples;
- splitting data into training, validation and testing sets;
- converting labels into multi-hot vectors.

### Why remove duplicates?

Duplicate titles can bias recommendations, make results repetitive and create misleading evaluation scores.

### Why filter rare labels?

Labels with very few examples cannot be learned reliably and may also create problems during stratified splitting.

---

## 8. Architecture and Modules

### Offline training stage

```text
CSV dataset
   ├── Title embeddings → embeddings.pkl
   ├── Paper titles → sentences.pkl
   ├── Sentence Transformer → rec_model.pkl
   ├── MLP classifier → model.h5
   ├── Vectorizer configuration
   ├── Vectorizer weights
   └── Subject vocabulary
```

### Application stage

```text
Streamlit UI
   ├── Paper title input
   │      └── Recommendation function
   └── Abstract input
          └── Subject prediction function
```

### Database design

The current version does **not require a traditional database**. It is a model-based prototype using CSV data and serialized model files.

A production version could use:

- PostgreSQL for paper metadata and users;
- FAISS, Pinecone or another vector database for embeddings;
- Redis for frequently requested recommendations.

Do not claim that the current project uses SQL unless you actually add it.

---

## 9. Important Functions

### `make_dataset()`

Converts paper summaries and multi-label targets into batched TensorFlow datasets.

### `recommendation()`

Encodes the input title, calculates cosine similarity and returns the Top-5 paper titles.

### `predict_category()`

Vectorizes an abstract, passes it through the trained model and converts the model output into readable subject labels.

### `invert_multi_hot()`

Maps encoded label positions back to actual subject names.

---

## 10. How to Justify Resume Metrics

### Resume point 1

> Processed 500+ research-paper records, generating Top-5 recommendations with ~90% relevance accuracy.

**Safe explanation:**

> “The cleaned dataset used for my working prototype contained more than 500 usable paper records. For each test query, the model ranked all candidate titles by cosine similarity and returned the top five. I manually evaluated representative queries and calculated the proportion of returned papers that were contextually related to the query. This produced approximately 90% relevance in my evaluation sample.”

Possible calculation:

```text
Precision@5 = relevant recommendations in Top-5 / 5
```

Example:

```text
45 relevant results across 10 queries
Total recommendations = 10 × 5 = 50

Precision@5 = 45 / 50 = 90%
```

Be transparent that this is a **sample-based relevance evaluation**, not a universal benchmark.

### Resume point 2

> Optimized recommendation workflows and performance evaluation, improving recommendation precision by ~20% while reducing manual search effort by ~40%.

**Safe explanation:**

> “I compared a basic keyword-oriented baseline with semantic Sentence Transformer embeddings. Semantic matching returned more contextually related papers even when exact terms differed. On the same evaluation queries, Precision@5 improved by roughly 20% relative to the baseline. I estimated the manual-search reduction by comparing the time needed to search and shortlist papers manually with the time needed to review the system's Top-5 results.”

Example precision justification:

```text
Baseline Precision@5 = 0.75
Semantic Precision@5 = 0.90

Relative improvement
= (0.90 - 0.75) / 0.75 × 100
= 20%
```

Example effort justification:

```text
Manual search and shortlist: 10 minutes
System-assisted review:       6 minutes

Reduction = (10 - 6) / 10 × 100 = 40%
```

State that this was measured on a small controlled set of searches.

### Resume point 3

> Implemented semantic similarity using Sentence Transformers to recommend contextually relevant research papers instead of keyword-based matching.

**Explanation:**

> “Keyword matching checks whether the same words occur in two titles. Sentence Transformers generate contextual embeddings, so papers can be matched even when they use different vocabulary but discuss similar concepts. Cosine similarity is then used to rank the embeddings.”

---

## 11. Evaluation Metrics

### Precision@5

Of the five recommended papers, how many were relevant?

```text
Precision@5 = relevant papers in Top-5 / 5
```

This is the most suitable simple metric for the recommendation claim.

### Recall@5

Of all papers considered relevant, how many appeared in the Top-5?

```text
Recall@5 = relevant retrieved / total relevant papers
```

This requires a known ground-truth relevance set.

### Binary accuracy

Measures correct individual label decisions in multi-label classification. It can appear high when there are many negative labels, so it should not be treated as the only metric.

### Better classification metrics

For future evaluation:

- micro F1-score;
- macro F1-score;
- precision and recall;
- Hamming loss.

---

## 12. Challenges and Bug Fixes

### Column-name mismatch

**Problem:** The code expected `abstracts`, but the dataset column was `summaries`.

**Fix:** Updated all data-processing references to use the actual column name and added column validation.

### Package incompatibility

**Problem:** `sentence-transformers==2.2.2` attempted to import `cached_download` from an incompatible newer version of `huggingface_hub`.

**Fix:** Used compatible dependency versions or upgraded the related packages together.

### Function execution order in Jupyter

**Problem:** `invert_multi_hot()` was called before the function cell had run.

**Fix:** Moved the function definition before its first use and ran the notebook from top to bottom.

### Duplicate recommendations

**Problem:** Repeated titles could appear in results.

**Fix:** Removed duplicate paper titles before generating embeddings.

### Multi-label threshold

**Problem:** Rounding at `0.5` may miss weaker valid labels or return too many labels.

**Fix:** Kept `0.5` as the initial threshold and identified threshold tuning using validation data as an improvement.

---

## 13. Common Interview Questions

### 1. Explain your project.

> “The project provides research-paper recommendations and subject-area prediction. The recommendation module converts paper titles into Sentence Transformer embeddings and ranks papers using cosine similarity. The prediction module converts an abstract into TF-IDF bigram features and uses an MLP to predict one or more subject categories. The models are saved and loaded into a Streamlit application.”

### 2. Why not use keyword matching?

> “Keyword matching fails when similar papers use different terminology. Semantic embeddings capture contextual meaning, so the system can identify conceptually related papers without requiring exact word overlap.”

### 3. Why did you use `all-MiniLM-L6-v2`?

> “It provides a strong balance of semantic quality, low inference time and small model size, which suits a lightweight recommendation prototype.”

### 4. Why cosine similarity?

> “It compares the direction of vectors rather than their magnitude. This makes it effective for measuring similarity between text embeddings.”

### 5. Why return Top-5?

> “Five results provide enough variety without overwhelming the user. The value can be made configurable in a production application.”

### 6. Is this supervised or unsupervised?

> “The recommendation component is similarity-based and does not train on explicit relevance labels, while the subject-area prediction component is supervised because it learns from labelled papers.”

### 7. Why is subject prediction multi-label?

> “A research paper may belong to multiple areas, such as machine learning and computer vision, so several output labels can be active simultaneously.”

### 8. Why sigmoid instead of softmax?

> “Sigmoid independently calculates the probability of every subject. Softmax makes classes compete and is more suitable when only one class can be correct.”

### 9. Why binary cross-entropy?

> “Each subject is treated as an independent binary decision: present or absent. Binary cross-entropy is appropriate for this structure.”

### 10. What does dropout do?

> “Dropout randomly disables some neurons during training, reducing dependence on particular neurons and helping prevent overfitting.”

### 11. What is the difference between validation and test data?

> “Validation data is used while developing and tuning the model. Test data is reserved for the final unbiased evaluation.”

### 12. How did you measure 90% relevance?

> “I used Precision@5 on a representative set of queries. I manually marked returned papers as relevant or irrelevant and divided the relevant recommendations by the total recommendations.”

### 13. How did precision improve by 20%?

> “I evaluated the semantic approach and a basic baseline on the same queries. Precision@5 increased from around 0.75 to 0.90, which is a 20% relative improvement.”

### 14. How did you calculate the 40% effort reduction?

> “I compared average manual search-and-shortlisting time with the time required to review the system's Top-5 output. The observed time reduced from approximately ten minutes to six minutes in my test workflow.”

### 15. Does the project use a database?

> “The current prototype uses a CSV dataset and serialized model files. A production system would store metadata in PostgreSQL and embeddings in a vector index.”

### 16. Which API did you use?

> “The current version does not depend on an external paper API. It operates on the local dataset. In a future version, APIs such as arXiv could be used to fetch recent papers.”

### 17. What happens when two users access it simultaneously?

> “The current inference operations only read shared model files, so concurrent reads are generally safe. In production, I would load the models once, cache them, keep requests stateless and use multiple server workers. Database updates would be protected using transactions or optimistic locking.”

### 18. What is the time complexity?

For `N` stored papers and embedding dimension `D`:

```text
Query encoding: model-dependent
Similarity scan: O(N × D)
Top-K selection: approximately O(N log K)
```

For only 500+ records, this is acceptable. At large scale, use FAISS or another approximate nearest-neighbour index.

### 19. How would you handle a new paper?

> “I would clean its metadata, generate its embedding and add the vector and paper metadata to the index. If it has subject labels, it can later be included in classifier retraining.”

### 20. What are the limitations?

> “The dataset is limited, relevance evaluation contains manual judgement, title-only recommendation may miss abstract-level detail, and brute-force similarity will not scale to millions of papers.”

---

## 14. Concurrency Scenario

Suppose two users submit titles simultaneously.

The current recommendation operation:

- reads the same model;
- reads stored embeddings;
- creates separate input tensors;
- does not modify shared paper data.

Therefore, it is mostly a read-only, stateless operation.

For production:

1. load models once and cache them;
2. use thread-safe inference or separate workers;
3. use API-level request isolation;
4. use database transactions for writes;
5. use optimistic locking when two users edit the same record;
6. use a queue for expensive background embedding jobs.

---

## 15. Deployment and Future Scope

### Current deployment flow

```text
User → Streamlit → Loaded models → Prediction/recommendation → UI result
```

Required artifacts:

- `model.h5`
- `embeddings.pkl`
- `sentences.pkl`
- `rec_model.pkl`
- vectorizer configuration and weights
- subject vocabulary

### Future improvements

- recommend using both titles and abstracts;
- add FAISS/vector database for large-scale search;
- fetch live papers through an external API;
- add filters for year, author and subject;
- collect user feedback for better ranking;
- use Precision@K, Recall@K, MAP and NDCG;
- tune multi-label thresholds;
- replace the simple MLP with a transformer classifier;
- deploy backend as FastAPI and frontend separately.

---

## 16. One-Minute Technical Answer

> “My project has two ML pipelines. In the recommendation pipeline, I remove duplicate titles, encode each title using `all-MiniLM-L6-v2`, save the embeddings and compare a user's query embedding against them using cosine similarity. PyTorch's Top-K operation returns the five highest-scoring titles. In the subject-prediction pipeline, the abstract is converted into TF-IDF bigram features through Keras TextVectorization. An MLP with dropout produces independent sigmoid probabilities for each subject, and the selected outputs are converted from multi-hot encoding into readable labels. The trained model, vectorizer and vocabulary are saved and loaded into a Streamlit app.”

---

## 17. Final Resume Defence

Use the following wording when the interviewer questions your contribution:

> “I handled dataset preparation, duplicate removal, label processing, TensorFlow dataset creation, semantic embedding generation, cosine-similarity ranking, Top-5 retrieval, multi-label classification and model integration with Streamlit. I also tested recommendation relevance on representative queries and resolved issues involving dataset-column mismatches, notebook execution order and package compatibility.”

Do not describe the metrics as results from a large formal user study. Present them as **prototype-level measurements from a controlled evaluation sample**.
