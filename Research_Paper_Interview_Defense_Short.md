# Research Paper Recommendation System — Interview Defense Notes

## 1. Resume Project

**Research Paper Recommendation System**  
*Python, NLP, Sentence Transformers, MLP, Streamlit*

- Processed **35,000+ research-paper records** through data cleaning and duplicate removal.
- Built a semantic recommendation pipeline using **Sentence Transformers and cosine similarity** to generate Top-5 contextually similar papers.
- Developed an **MLP-based multi-label classifier** for predicting research-paper subject areas and integrated both modules into a Streamlit application.

---

## 2. Project Explanation — 45 Seconds

> “Maine ek Research Paper Recommendation and Subject Area Prediction system banaya. Dataset ko clean karke duplicate titles aur rare labels remove kiye. Recommendation module mein paper titles ko Sentence Transformer se embeddings mein convert kiya aur cosine similarity se Top-5 similar papers retrieve kiye. Dusre module mein abstract ko TF-IDF bigram features mein convert karke MLP-based multi-label classifier se subject areas predict kiye. Dono modules ko Streamlit UI mein integrate kiya.”

---

## 3. Complete Flow

```text
CSV: titles, summaries, terms
          |
          v
Cleaning + duplicate removal + rare-label filtering
          |
          +------------------------------+
          |                              |
          v                              v
Recommendation Module              Prediction Module
Input paper title                  Input abstract
Sentence Transformer              TextVectorization
Embedding                         TF-IDF + bigrams
Cosine similarity                 MLP
Top-5 papers                      Multiple subject labels
          |                              |
          +--------------+---------------+
                         v
                    Streamlit UI
```

---

## 4. Tech Stack and Why

| Technology | Use and justification |
|---|---|
| **Python** | ML, preprocessing and UI ke liye simple ecosystem |
| **Pandas** | CSV loading, filtering and duplicate removal |
| **NumPy** | Numerical and label operations |
| **Sentence Transformers** | Text ko semantic embeddings mein convert karta hai |
| **all-MiniLM-L6-v2** | Fast, lightweight and good accuracy-speed balance |
| **Cosine Similarity** | Embeddings ki direction compare karke contextual similarity deta hai |
| **PyTorch** | `torch.topk()` se highest five similarity scores select kiye |
| **TensorFlow/Keras** | MLP subject classifier train aur save kiya |
| **TextVectorization** | Abstract ko TF-IDF bigram numerical features mein convert kiya |
| **Streamlit** | Minimum frontend code mein working ML application |
| **Pickle/H5** | Embeddings, vocabulary, vectorizer aur trained model save/load |

### API and Database

> “Current prototype mein external API aur traditional database use nahi hua. Dataset CSV mein hai aur trained artifacts local `.pkl` aur `.h5` files se load hote hain.”

Code mein koi arXiv, Semantic Scholar, REST ya third-party data API call nahi hai.

---

## 5. My Role

> “Ye mera personal major project hai. Maine dataset schema check kiya, null/duplicate records handle kiye, rare subject groups filter kiye, train-validation-test split banaya, multi-hot label encoding ki, TF-IDF bigram vectorizer adapt kiya, MLP train kiya, Sentence Transformer embeddings generate ki, cosine-similarity Top-5 retrieval implement ki aur Streamlit integration ki.”

### Actual tasks

- `titles`, `summaries`, `terms` columns validate kiye.
- 51,774 raw records load kiye.
- Duplicate titles remove karke 38,972 records rakhe.
- Rare label combinations filter karke 36,651 usable records rakhe.
- Split: 32,985 train, 1,833 validation, 1,833 test.
- Recommendation ke liye unique titles ke embeddings generate kiye.
- Subject prediction ke liye 512 and 256 neuron MLP with dropout banaya.
- Model, vocabulary, vectorizer and embeddings save/load kiye.
- Streamlit se title and abstract inputs connect kiye.

---

## 6. Resume Lines Ki Justification

### “Processed 35,000+ records”

> “Raw dataset mein 51,774 rows thi. Duplicate titles remove karne ke baad 38,972 rows bachi. Rare subject groups filter karne ke baad final usable dataset 36,651 records ka tha, isliye resume mein conservative 35,000+ likha.”

### “Data cleaning and duplicate removal”

Cleaning mein:

- required-column validation 
- null check
- duplicate-title removal
- rare label combinations remove karna
- index reset
- train/validation/test split

### “Semantic recommendation pipeline”

> “Har paper title ka dense numerical embedding create hota hai. User input ka embedding stored paper embeddings se compare hota hai. Isliye system exact keywords ke bajay contextual meaning par recommend karta hai.”

### “Top-5 contextually similar papers”

> “Cosine similarity scores calculate karke `torch.topk(..., k=5)` highest five papers return karta hai.”

**Important:** Top-5 generation fully implemented hai, lekin resume mein recommendation accuracy percentage claim nahi kiya gaya.

### “MLP-based multi-label classifier”

> “Ek paper multiple subjects ka ho sakta hai. Isliye output layer mein sigmoid activation aur binary cross-entropy use ki. MLP mein Dense 512, Dropout, Dense 256, Dropout and final sigmoid layer hai.”

### “Integrated both modules into Streamlit”

> “Same UI mein title input recommendation module ko aur abstract input subject-prediction module ko diya gaya. Button click par dono outputs show hote hain.”

---

## 7. Metrics

### Dataset Metric

| Stage | Records |
|---|---:|
| Raw dataset | 51,774 |
| After duplicate-title removal | 38,972 |
| After rare-label filtering | 36,651 |
| Training | 32,985 |
| Validation | 1,833 |
| Test | 1,833 |

### Classification Result

Notebook output:

- Test binary accuracy: **99.42%**
- Validation binary accuracy: **99.45%**

Safe answer:

> “99.42% binary accuracy subject-area classifier ki hai, recommendation module ki nahi. Multi-label problems mein class imbalance ke kaaran accuracy alone misleading ho sakti hai, isliye future evaluation mein precision, recall, F1 and per-label metrics add karunga.”

Never say **90% recommendation accuracy** unless a separate labelled or manual Precision@5 evaluation is performed.

---

## 8. Alternatives and Why

### Sentence Transformer alternatives

| Alternative | Why consider it |
|---|---|
| **TF-IDF + cosine** | Simple and fast baseline, but exact-word focused |
| **MPNet** | Potentially better semantic quality, but heavier |
| **E5/BGE embeddings** | Retrieval-focused modern models; comparison useful |
| **OpenAI embeddings/API** | Strong managed embeddings, but internet, cost and API dependency |
| **Doc2Vec** | Lightweight classical alternative, usually weaker context understanding |

Current MiniLM was chosen because it is **local, fast and lightweight**.

### Similarity-search alternatives

| Alternative | Why consider it |
|---|---|
| **FAISS** | Large datasets par faster nearest-neighbour search |
| **Chroma/Pinecone/Weaviate** | Vector storage, filtering and production retrieval |
| **Euclidean distance** | Possible, but cosine is more natural for text-vector direction |

### Classifier alternatives

| Alternative | Why consider it |
|---|---|
| **Logistic Regression** | Simple baseline and highly explainable |
| **SVM** | Strong text classification baseline |
| **LSTM** | Sequence information capture karta hai but slower |
| **BERT fine-tuning** | Better context possible, but expensive and heavier |
| **One-vs-Rest classifier** | Easy classical multi-label baseline |

MLP choose kiya because TF-IDF features ke saath **simple, fast and easy to deploy** tha.

### UI alternatives

- Flask/FastAPI + React: more flexible production architecture
- Gradio: faster ML demo
- Streamlit: quickest complete prototype

---

## 9. Challenges and Fixes

### 1. Column mismatch

**Problem:** Code `abstracts` expect kar raha tha, actual column `summaries` tha.  
**Fix:** Correct column references and schema validation add ki.

### 2. Package compatibility

**Problem:** `sentence-transformers==2.2.2` newer `huggingface_hub` ke saath `cached_download` error de raha tha.  
**Fix:** Compatible versions pin kiye or package stack upgrade ki.

### 3. Notebook execution order

**Problem:** `invert_multi_hot()` definition se pehle call ho rahi thi.  
**Fix:** Function definition first-use se pehle move ki and Run All sequence correct kiya.

### 4. Duplicate titles

**Problem:** Same title recommendation pool mein repeat ho sakta tha.  
**Fix:** Embeddings create karne se pehle title-level duplicates remove kiye.

### 5. Rare label combinations

**Problem:** Kuch subject combinations sirf ek baar the, jisse stratified split fail ya unstable ho sakta tha.  
**Fix:** Frequency one wale groups remove kiye.

### 6. Overfitting risk

**Fix:** Dropout layers and EarlyStopping use kiya.

---

## 10. Future Scope — Alternatives Ko Compare Karna

> “Future mein blindly technology replace nahi karunga. Same evaluation set par alternatives compare karke best accuracy-speed-cost trade-off select karunga.”

1. **MiniLM vs MPNet vs E5/BGE**  
   Precision@5, latency and memory compare karunga.

2. **Title-only vs title + abstract embeddings**  
   Check karunga ki additional context relevance improve karta hai ya nahi.

3. **Linear cosine search vs FAISS**  
   Dataset grow hone par search latency compare karunga.

4. **MLP vs Logistic Regression/SVM/BERT**  
   F1 score, training cost and inference time compare karunga.

5. **CSV vs PostgreSQL/vector database**  
   User history, feedback and filters add hone par database introduce karunga.

6. **Offline data vs arXiv/Semantic Scholar API**  
   Live papers useful hon toh API integrate karunga; current local model fallback maintain karunga.

7. **Streamlit vs FastAPI + frontend**  
   Prototype ke liye Streamlit; production scalability ke liye API-based architecture compare karunga.

8. **Feedback-based reranking**  
   User clicks/saves se future ranking personalize ki ja sakti hai.

---

## 11. Important Follow-up Answers

### Why semantic matching?

> “Same meaning different words mein express ho sakta hai. Embeddings contextual meaning capture karti hain; keyword matching mostly word overlap dekhta hai.”

### Why cosine similarity?

> “Text embeddings mein direction meaning represent karti hai. Cosine direction compare karta hai and vector magnitude se less affected hota hai.”

### Why multi-label?

> “Research paper AI aur Computer Vision jaise multiple subjects belong kar sakta hai.”

### Why sigmoid, not softmax?

> “Softmax normally one class select karta hai. Sigmoid har subject ki independent probability deta hai.”

### Two users at the same time?

> “Current inference mostly read-only hai, so each request independently run ho sakti hai. Future writes like ratings or feedback ke liye database transactions or optimistic locking use karunga.”

### Database design?

> “Current version file-based hai. Production version mein `papers`, `subjects`, `users` and `feedback` tables rakhunga; embeddings FAISS or vector DB mein.”

### API used?

> “No external API in the current implementation. Future mein live paper retrieval ke liye arXiv or Semantic Scholar API evaluate karunga.”

---

## 12. Final Revision

```text
Raw records: 51,774
Final usable: 36,651
Recommendation: title → MiniLM embedding → cosine similarity → Top-5
Prediction: abstract → TF-IDF bigrams → MLP → multiple subjects
UI: Streamlit
Storage: CSV + local model files
API/database: not used currently
Future: compare MPNet/E5, FAISS, BERT, APIs and production architecture
```
