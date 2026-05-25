### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 25-05-2026
## name:R mounish vamsi kumar
## reg no : 212224100033
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```
!pip install nltk scikit-learn
```
```
  from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        token for token in tokens
        if token not in stopwords.words("english") and token not in string.punctuation
    ]
    return " ".join(tokens)

preprocessed_docs = {
    doc_id: preprocess_text(doc)
    for doc_id, doc in documents.items()
}

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

def search(query):
    query_vec = tfidf_vectorizer.transform([preprocess_text(query)])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results = []
    for doc_id, score in zip(documents.keys(), cosine_sim):
        results.append((doc_id, documents[doc_id], score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

query = input("Enter your query: ")

search_results = search(query)

print("Query:", query)

for i, result in enumerate(search_results, start=1):
    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")

highest_rank_score = max(result[2] for result in search_results)
print("The highest rank cosine score is:", highest_rank_score)
```
### Output:
<img width="678" height="625" alt="image" src="https://github.com/user-attachments/assets/ad81ccc0-ea89-477a-ab67-061956b94512" />

### Result:
Thus, the Information Retrieval using Vector Space Model in Python was executed successfully.
