import os
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ==============================================
# ✅ STEP 1: LOAD DATASET AND DOCUMENTS
# ==============================================

# Load the Wikipedia dataset (first 1000 samples)
dataset = load_dataset("wikipedia", "20220301.en", split="train[:1000]", trust_remote_code=True)

# Extract text data
documents = [doc["text"] for doc in dataset if "text" in doc]

# Ensure no empty documents
documents = [doc.strip() for doc in documents if len(doc.strip()) > 0]

print("Sample Document:", documents[0][:500])


# ==============================================
# ✅ STEP 2: CREATE OR LOAD FAISS INDEX
# ==============================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

FAISS_INDEX_PATH = "faiss_index.idx"

if os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing FAISS index...")
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    print("Creating new FAISS index...")

    document_embeddings = embedding_model.encode(documents, convert_to_tensor=True)
    document_embeddings_np = document_embeddings.cpu().detach().numpy()

    dimension = document_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(document_embeddings_np)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("FAISS index created and saved!")


# ==============================================
# ✅ STEP 3: RETRIEVE RELEVANT DOCUMENTS
# ==============================================

def retrieve_relevant_docs(query, top_k=2):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()

    if index.ntotal == 0:
        raise ValueError("FAISS index is empty. Ensure documents were properly added.")

    distances, indices = index.search(query_embedding_np.reshape(1, -1), top_k)

    retrieved_docs = [documents[i] for i in indices[0] if i < len(documents)]

    if not retrieved_docs:
        retrieved_docs = ["No relevant information found."]

    return retrieved_docs


query = "What is the capital of France?"
retrieved_docs = retrieve_relevant_docs(query, top_k=2)

for i, doc in enumerate(retrieved_docs):
    print(f"\nTop {i+1} Retrieved Document:\n", doc[:500])


# ==============================================
# ✅ STEP 4: GENERATE RESPONSE USING GPT-2 (FINAL FIX)
# ==============================================

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

def generate_response(query, retrieved_docs):
    context = "\n".join(retrieved_docs)

    # ✅ Dynamically truncate context to fit within 1024 tokens
    tokenized_context = tokenizer(context, truncation=True, max_length=750, return_tensors="pt")
    tokenized_query = tokenizer(query, truncation=True, max_length=250, return_tensors="pt")

    # Combine context and query
    input_ids = torch.cat([tokenized_context["input_ids"], tokenized_query["input_ids"]], dim=-1)
    attention_mask = torch.cat([tokenized_context["attention_mask"], tokenized_query["attention_mask"]], dim=-1)

    # ✅ Ensure total input does not exceed 1024 tokens
    input_ids = input_ids[:, :1024]
    attention_mask = attention_mask[:, :1024]

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

query = "What is anarchism?"
retrieved_docs = retrieve_relevant_docs(query, top_k=2)

response = generate_response(query, retrieved_docs)

print("\nGenerated Response:\n", response)
