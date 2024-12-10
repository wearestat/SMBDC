import os
import json
import sys
import requests
import pandas as pd
from pathlib import Path
from supabase import create_client
from PyPDF2 import PdfReader
import numpy as np
from openai import OpenAI
import time
import openpyxl
import logging

# Load environment variables for local testing
if os.getenv("GITHUB_ACTIONS") is None:  # Detect if running locally
    from dotenv import load_dotenv
    load_dotenv()

# Initialize OpenAI and Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SERVICE_ROLE")
supabase = create_client(supabase_url, supabase_key)
MAX_TOKENS = 8191
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI()

# Function to download file
def download_file(uri, destination="downloads"):
    os.makedirs(destination, exist_ok=True)
    file_name = Path(uri).name
    file_path = os.path.join(destination, file_name)

    # Convert GitHub blob URL to raw URL if needed
    if "github.com" in uri:
        uri = uri.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    response = requests.get(uri)
    response.raise_for_status()
    with open(file_path, "wb") as file:
        file.write(response.content)
    return file_path

# Generate embedding for a single input
def generate_embedding(content):
    response = client.embeddings.create(
        input=content,
        model=OPENAI_EMBEDDING_MODEL
    )
    return response.data[0].embedding

# Aggregate embeddings by averaging
def aggregate_embeddings(embeddings):
    return np.mean(embeddings, axis=0).tolist()

# Generate embeddings for chunks
def generate_embeddings_for_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        if len(chunk) > MAX_TOKENS:
            chunk = chunk[:MAX_TOKENS]  # Truncate to avoid exceeding token limit
        try:
            response = client.embeddings.create(input=chunk, model=OPENAI_EMBEDDING_MODEL)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
            raise
    return embeddings


def generate_embeddings_with_rate_limit(chunks, batch_size, model, tpm_limit):
    """
    Generate embeddings with rate limiting to respect OpenAI TPM constraints.
    """
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            # Prepare input for embedding API
            batch_contents = [chunk["content"] for chunk in batch]
            token_count = sum(len(content.split()) for content in batch_contents)

            # Ensure we don’t exceed TPM
            if token_count > tpm_limit:
                wait_time = token_count / tpm_limit * 60  # Calculate wait time in seconds
                print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)

            response = client.embeddings.create(input=batch_contents, model=model)
            batch_embeddings = [data.embedding for data in response.data]

            # Attach embeddings to chunks
            for j, embedding in enumerate(batch_embeddings):
                batch[j]["embedding"] = embedding
                embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embeddings for batch {i}-{i+batch_size}: {e}")
            raise

    return embeddings

def process_csv_with_batching(file_path, dataset_id, chunk_size=50, batch_size=50, tpm_limit=1000000):
    """
    Process a large CSV file with batching, chunking, and rate-limiting.
    """
    dataframe = pd.read_csv(file_path)
    
    # Extract schema
    schema = {"fields": [{"name": col, "type": str(dataframe[col].dtype)} for col in dataframe.columns]}
    tags = [{"name": col} for col in dataframe.columns]
    
    chunks = []  # List to store chunks
    embeddings = []  # List to store all embeddings

    # Step 1: Create chunks of rows
    for i in range(0, len(dataframe), chunk_size):
        chunk = dataframe.iloc[i:i + chunk_size]
        chunk_content = "\n".join([
            " ".join([f"{col}: {row[col]}" for col in chunk.columns if pd.notna(row[col])])
            for _, row in chunk.iterrows()
        ])
        chunks.append({
            "dataset_id": dataset_id,
            "content": chunk_content,
            "metadata": {"chunk_start": i, "chunk_end": min(i + chunk_size, len(dataframe))}
        })
    
    # Step 2: Generate embeddings with rate limiting
    embeddings = generate_embeddings_with_rate_limit(
        chunks=chunks,
        batch_size=batch_size,
        model=OPENAI_EMBEDDING_MODEL,
        tpm_limit=tpm_limit
    )

    # Step 3: Compute aggregated embedding for the entire dataset
    aggregated_embedding = aggregate_embeddings(embeddings)

    return chunks, aggregated_embedding, schema, tags


# Process CSV files
def process_csv(file_path, dataset_id, chunk_size=1000):
    dataframe = pd.read_csv(file_path)
    schema = {"fields": [{"name": col, "type": str(dataframe[col].dtype)} for col in dataframe.columns]}
    tags = [{"name": col} for col in dataframe.columns]
    
    rows = []
    embeddings = []
    
    for index, row in dataframe.iterrows():
        content = " ".join([f"{col}: {row[col]}" for col in dataframe.columns if pd.notna(row[col])])
        embedding = generate_embedding(content)
        rows.append({
            "dataset_id": dataset_id,
            "content": content,
            "embedding": embedding,
            "metadata": row.to_dict()
        })
        embeddings.append(embedding)

    # Calculate aggregated embedding
    aggregated_embedding = aggregate_embeddings(embeddings)
    return rows, aggregated_embedding, schema, tags

# Process XLS/XLSX files
def process_xsl(file_path, dataset_id, chunk_size=1000):
    """
    Process XLS/XLSX files by reading the data, chunking, generating embeddings, and preparing rows.
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        rows = []
        for index, row in df.iterrows():
            content = row.to_json()
            embedding = generate_embedding(content)
            rows.append({
                "dataset_id": dataset_id,
                "content": content,
                "embedding": embedding,
                "metadata": row.to_dict()
            })
        aggregated_embedding = aggregate_embeddings([row["embedding"] for row in rows])
        schema = df.columns.tolist()
        tags = {"file_type": "xlsx"}
        return rows, aggregated_embedding, schema, tags
    
    except Exception as e:
        print(f"Error processing XLS file: {e}")
        raise

# Process XLS/XLSX files with batching and chunking
def process_xsl_with_batching(file_path, dataset_id, chunk_size=1000, batch_size=50, tpm_limit=1000000):
    """
    Process XLS/XLSX files with chunking and batching to handle large datasets efficiently.
    """
    try:
        for chunk_number, dataframe in enumerate(pd.read_excel(file_path, chunksize=chunk_size, engine='openpyxl')):
            print(f"Processing chunk {chunk_number + 1}")
            
            schema = {"fields": [{"name": col, "type": str(dataframe[col].dtype)} for col in dataframe.columns]}
            tags = [{"name": col} for col in dataframe.columns]
            
            chunks = []
            for index, row in dataframe.iterrows():
                content = " ".join([f"{col}: {row[col]}" for col in dataframe.columns if pd.notna(row[col])])
                chunks.append({
                    "dataset_id": dataset_id,
                    "content": content,
                    "metadata": row.to_dict()
                })
            
            # Generate embeddings with batching and rate limiting
            embeddings = generate_embeddings_with_rate_limit(
                chunks=chunks,
                batch_size=batch_size,
                model=OPENAI_EMBEDDING_MODEL,
                tpm_limit=tpm_limit
            )
    
            # Attach embeddings to chunks
            rows = []
            for i, chunk in enumerate(chunks):
                rows.append({
                    "dataset_id": chunk["dataset_id"],
                    "content": chunk["content"],
                    "embedding": embeddings[i],
                    "metadata": chunk["metadata"]
                })
    
            # Calculate aggregated embedding for the chunk
            aggregated_embedding = aggregate_embeddings(embeddings)
    
       
        return rows, aggregated_embedding, schema, tags
    
    except Exception as e:
        print(f"Error processing XLS/XLSX file: {e}")
        raise

# Process PDF files
def process_pdf(file_path, dataset_id, chunk_size=1000):
    reader = PdfReader(file_path)
    content = " ".join(page.extract_text() for page in reader.pages)
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    embeddings = generate_embeddings_for_chunks(chunks)

    rows = []
    for i, chunk in enumerate(chunks):
        rows.append({
            "dataset_id": dataset_id,
            "content": chunk,
            "embedding": embeddings[i],
            "metadata": {}
        })

    aggregated_embedding = aggregate_embeddings(embeddings)
    schema = None
    tags = []
    return rows, aggregated_embedding, schema, tags

# Process text or Markdown files
def process_text_or_markdown(file_path, dataset_id, chunk_size=1000):
    with open(file_path, "r") as file:
        content = file.read()
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    embeddings = generate_embeddings_for_chunks(chunks)

    rows = []
    for i, chunk in enumerate(chunks):
        rows.append({
            "dataset_id": dataset_id,
            "content": chunk,
            "embedding": embeddings[i],
            "metadata": {}
        })

    aggregated_embedding = aggregate_embeddings(embeddings)
    schema = None
    tags = []
    return rows, aggregated_embedding, schema, tags

# Update dataset metadata in Supabase
def update_supabase_dataset(dataset_id, schema, tags, embedding):
    response = supabase.table("datasets").update({
        "schema": json.dumps(schema),
        "tags": json.dumps(tags),
        "embedding": embedding
    }).eq("id", dataset_id).execute()

    if not response.data:
        raise Exception(f"Error updating dataset: {response}")
    print("Supabase dataset update successful!")

# Insert rows into `dataset_rows` table in Supabase
def insert_rows_into_supabase(rows):

    response = supabase.table("dataset_rows").upsert(rows).execute()
    print("Rows successfully inserted into dataset_rows!" + response.count)

# Main function to process datasets
def process_dataset(payload):
    try:
        # Parse payload
        print("Parsing JSON payload")
        dataset_id = payload["id"]
        uri = payload["URI"]

        print(f"Processing dataset {dataset_id}")
        file_path = download_file(uri)
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".csv": 
            # Process CSV with batching and chunking
            rows, aggregated_embedding, schema, tags = process_csv_with_batching(
                file_path=file_path,
                dataset_id=dataset_id,
                chunk_size=50, 
                batch_size=50, 
                tpm_limit=1000000
            )
        elif file_ext == ".pdf":
            rows, aggregated_embedding, schema, tags = process_pdf(file_path, dataset_id)
        elif file_ext in [".md", ".txt"]:
            rows, aggregated_embedding, schema, tags = process_text_or_markdown(file_path, dataset_id)
        elif file_ext in [".xls", ".xlsx"]:
            rows, aggregated_embedding, schema, tags = process_xsl_with_batching(file_path=file_path,
                dataset_id=dataset_id,
                chunk_size=50, 
                batch_size=50, 
                tpm_limit=1000000)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Update dataset in `datasets` table
        update_supabase_dataset(dataset_id, schema, tags, aggregated_embedding)

        # Insert rows into `dataset_rows` table
        insert_rows_into_supabase(rows)

        print(f"Successfully processed dataset {dataset_id}")
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_dataset.py <payload.json>")
        sys.exit(1)
    payload_file = sys.argv[1]
    try:
        with open(payload_file, 'r') as f:
            payload = json.load(f)
        process_dataset(payload)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON input: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Payload file not found: {payload_file}")
        sys.exit(1)