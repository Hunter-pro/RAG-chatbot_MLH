from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

def read_doc(directory: str) -> list[str]:
    # Initialize a PyPDFDirectoryLoader object with the given directory
    file_loader = PyPDFDirectoryLoader(directory)

    # Load PDF documents from the directory
    documents = file_loader.load()

    # Extract only the page content from each document
    page_contents = [doc.page_content for doc in documents]

    return page_contents

def chunk_text_for_list(docs: list[str], max_chunk_size: int = 1000) -> list[list[str]]:
    def chunk_text(text: str, max_chunk_size: int) -> list[str]:
        # Ensure each text ends with a double newline to correctly split paragraphs
        if not text.endswith("\n\n"):
            text += "\n\n"
        # Split text into paragraphs
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        # Iterate over paragraphs and assemble chunks
        for paragraph in paragraphs:
            # Check if adding the current paragraph exceeds the maximum chunk size
            if (
                len(current_chunk) + len(paragraph) + 2 > max_chunk_size
                and current_chunk
            ):
                # If so, add the current chunk to the list and start a new chunk
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Add the current paragraph to the current chunk
            current_chunk += paragraph.strip() + "\n\n"
        # Add any remaining text as the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # Apply the chunk_text function to each document in the list
    return [chunk_text(doc, max_chunk_size) for doc in docs]

# Replace OpenAI embeddings with sentence-transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(documents: list[any]) -> list[list[float]]:
    # Flatten the nested list if present
    flattened_docs = [item for sublist in documents for item in (sublist if isinstance(sublist, list) else [sublist])]
    # Generate embeddings
    embeddings = model.encode(flattened_docs)
    # Return as list of lists
    return [[emb.tolist()] for emb in embeddings]

import hashlib

def generate_short_id(content: str) -> str:
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()


def combine_vector_and_text(
    documents: list[any], doc_embeddings: list[list[float]]
) -> list[dict[str, any]]:
    data_with_metadata = []

    for doc_text, embedding in zip(documents, doc_embeddings):
        # Convert doc_text to string if it's not already a string
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        # Generate a unique ID based on the text content
        doc_id = generate_short_id(doc_text)

        # Create a data item dictionary
        data_item = {
            "id": doc_id,
            "values": embedding[0],
            "metadata": {"text": doc_text},  # Include the text as metadata
        }

        # Append the data item to the list
        data_with_metadata.append(data_item)

    return data_with_metadata


try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        st.error("Pinecone API key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, if not create it
if "mlh-rag-chatbot" not in pc.list_indexes().names():
    pc.create_index(
        name="mlh-rag-chatbot",
        dimension=384,  # all-MiniLM-L6-v2 dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index("mlh-rag-chatbot")

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]]) -> None:
    index.upsert(vectors=data_with_metadata)

#Call the function
#upsert_data_to_pinecone(data_with_metadata= data_with_meta_data)

def get_query_embeddings(query: str) -> list[float]:
    # Generate embeddings for the query
    query_embedding = model.encode([query])[0]
    return query_embedding.tolist()

# Call the function

def query_pinecone_index(
    query_embeddings: list, top_k: int = 2, include_metadata: bool = True
) -> dict[str, any]:
    query_response = index.query(
        vector=query_embeddings, top_k=top_k, include_metadata=include_metadata
    )
    return query_response

# Call the function

from openai import OpenAI
def generate_answer(answers: dict[str, any], prompt) -> str:
    # Check if we have any matches
    if not answers or not answers.get('matches') or len(answers['matches']) == 0:
        st.warning("No relevant information found in the context. Please try a different question.")
        return "No relevant information found in the context. Please try a different question."

    try:
        client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed"
        )
        text_content = answers['matches'][0]['metadata']['text']

        # Create a placeholder for streaming output
        message_placeholder = st.empty()
        full_response = ""

        completion = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Analyze the provided context carefully and answer questions accurately. If the answer cannot be found in the context, say so."},
                {"role": "user", "content": f"""Context: {text_content}

Question: {prompt}

Instructions: 
1. Use the context above to answer the question
2. If the answer isn't in the context, say "I cannot find the answer in the provided context"
3. Keep your response focused and relevant to the question
4. Use clear, concise language

Answer:"""}
            ],
            temperature=0.3,
            max_tokens=512,
            stream=True
        )
        
        # Stream the response with better formatting
        for chunk in completion:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                # Update the placeholder with the accumulated response
                message_placeholder.markdown(full_response)
        
        return full_response
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "Sorry, there was an error. Please ensure LM Studio is running and try again."