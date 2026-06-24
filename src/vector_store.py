import numpy as np
import json
import os
import ollama

def cosine_similarity(v1, v2):
    """
    Computes the cosine similarity between two vectors.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

class LocalVectorStore:
    """
    An in-memory vector database with disk-based persistence.
    """
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.db = []  # List of dicts: {"text": str, "vector": list[float]}

    def add_chunks(self, chunks, batch_size=16):
        """
        Generates embeddings for chunks in batches and adds them to the vector store.
        """
        print(f"Generating embeddings for {len(chunks)} chunks using '{self.model_name}'...")
        for idx in range(0, len(chunks), batch_size):
            batch = chunks[idx:idx + batch_size]
            try:
                # Generate embeddings for the batch
                response = ollama.embed(model=self.model_name, input=batch)
                embeddings = response.embeddings
                for j, text in enumerate(batch):
                    self.db.append({
                        "text": text,
                        "vector": embeddings[j]
                    })
            except Exception as e:
                # Fallback to single chunk processing if batch fails
                print(f"Batch generation failed at index {idx}, falling back to item-by-item: {e}")
                for text in batch:
                    try:
                        res = ollama.embed(model=self.model_name, input=text)
                        self.db.append({
                            "text": text,
                            "vector": res.embeddings[0]
                        })
                    except Exception as inner_e:
                        print(f"Failed to embed chunk: {inner_e}")
                        
            if idx > 0 and idx % (batch_size * 5) == 0:
                print(f"Embedded {idx}/{len(chunks)} chunks...")
                
        print(f"Successfully indexed {len(self.db)} chunks.")

    def save(self, filepath):
        """
        Saves the database to a JSON file on disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": self.model_name,
                "db": self.db
            }, f, ensure_ascii=False)
        print(f"Vector database saved to {filepath}")

    def load(self, filepath):
        """
        Loads the database from a JSON file.
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.model_name = data.get("model_name", self.model_name)
            self.db = data.get("db", [])
        print(f"Vector database loaded from {filepath} with {len(self.db)} records.")
        return True

    def retrieve(self, query, top_k=3):
        """
        Embeds the query and retrieves the top-k most semantically similar chunks.
        """
        if not self.db:
            print("Vector store is empty.")
            return []
            
        try:
            res = ollama.embed(model=self.model_name, input=query)
            query_vector = res.embeddings[0]
        except Exception as e:
            print(f"Failed to embed query: {e}")
            return []
            
        scored_chunks = []
        for item in self.db:
            score = cosine_similarity(query_vector, item["vector"])
            scored_chunks.append((score, item["text"]))
            
        # Sort in descending order of similarity score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return scored_chunks[:top_k]
