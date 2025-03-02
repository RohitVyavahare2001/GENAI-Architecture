import os
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from datasets import load_dataset
# Load the Wikipedia dataset (first 1000 samples)
dataset = load_dataset("wikipedia", "20220301.en", split="train[:1000]", trust_remote_code=True)

# Extract text data
documents = [doc["text"] for doc in dataset]

# Print a sample document
print("Sample Document:", documents[0][:500])


query = "What is the capital of France?"

# Load a sentence transformer model to generate document embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert each document to an embedding
document_embeddings = embedding_model.encode(documents, convert_to_tensor=True)
def retrieve_relevant_docs(query, top_k=3):
    # Convert query to embedding
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()

    # Search FAISS index
    distances, indices = index.search(query_embedding_np.reshape(1, -1), top_k)

    # Retrieve the top matching documents
    retrieved_docs = [documents[i] for i in indices[0]]
    
    return retrieved_docs

retrieved_docs = retrieve_relevant_docs(query)


# Load GPT-2 for generation
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def generate_response(query, retrieved_docs):
    # Combine retrieved documents as context
    context = "\n".join(retrieved_docs)
    
    # Format prompt for LLM
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    output = model.generate(**inputs, max_length=150)

    # Decode response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Generate answer using RAG approach
response = generate_response(query, retrieved_docs)

# Print response
print("\nGenerated Response:\n", response)
