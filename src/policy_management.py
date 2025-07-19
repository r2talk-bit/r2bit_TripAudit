"""
Policy Loader Module for TripAudit

This module provides functionality to load policy documents from PDF files into a ChromaDB vector database,
and retrieve relevant policies via semantic search. The implementation uses an in-memory ChromaDB instance
and supports operations based on user ID.

Functions:
- load_pdf: Loads a PDF file and extracts its text content
- add_policy: Adds a policy document to the vector database
- search_policies: Retrieves relevant policies based on semantic search
- delete_user_policies: Deletes all policies associated with a specific user
- get_all_user_policies: Gets all policies associated with a specific user
"""

# Standard library imports for file operations and unique ID generation
import os
import uuid
from typing import Dict, List, Optional, Union, Any  # Type hints for better code documentation

# ChromaDB: Vector database for storing and searching embeddings
import chromadb

# LangChain components for document processing and embeddings
from langchain_core.documents import Document  # Document object to represent text chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_openai import OpenAIEmbeddings  # For generating embeddings using OpenAI's models

# PyPDF for extracting text from PDF files
import pypdf

# Initialize the ChromaDB client with in-memory storage
# Using the new client configuration format as per ChromaDB documentation
client = chromadb.EphemeralClient()  # This creates an in-memory client that doesn't persist data

# Initialize the OpenAI embeddings model with fallback mechanism
# This will convert text into high-dimensional vectors that capture semantic meaning
try:
    embeddings = OpenAIEmbeddings()  # Uses OpenAI's API to generate embeddings
    use_openai_embeddings = True
except Exception as e:
    print(f"Warning: Could not initialize OpenAI embeddings: {str(e)}")
    print("Using fallback embedding method instead.")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    try:
        # Try to use a local HuggingFace model if available
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        use_openai_embeddings = False
    except Exception as e2:
        print(f"Warning: Could not initialize HuggingFace embeddings: {str(e2)}")
        print("Using simple keyword-based matching instead of embeddings.")
        # Define a simple embedding function that just counts keyword occurrences
        # This is a very basic fallback that will work without any external API
        use_openai_embeddings = False
        
        class SimpleKeywordEmbeddings:
            def embed_query(self, text):
                # Very simple embedding: just a count of characters
                # This is just a placeholder and won't provide good semantic search
                # but will allow the system to function without API access
                return [ord(c) % 10 for c in text[:100]] + [0] * 1436
                
        embeddings = SimpleKeywordEmbeddings()

# Create a collection named 'policies' to store our policy documents
# A collection in ChromaDB is like a table in a traditional database
policy_collection = client.get_or_create_collection(
    name="policies",  # Name of our collection
    metadata={"hnsw:space": "cosine"}  # Using cosine similarity for comparing vectors
)

def load_pdf(file_path: str) -> str:
    """
    Load a PDF file and extract its text content.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content from the PDF
    """
    # First, check if the file exists to provide a clear error message if it doesn't
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    try:
        # Open the PDF file in binary mode ('rb') since PDF is a binary format
        with open(file_path, "rb") as file:
            # Create a PDF reader object from PyPDF library
            pdf_reader = pypdf.PdfReader(file)
            
            # Initialize an empty string to store the extracted text
            text = ""
            
            # Loop through each page in the PDF and extract its text content
            for page in pdf_reader.pages:
                # Add the page's text plus a newline character to separate pages
                text += page.extract_text() + "\n"
            
            return text
    except Exception as e:
        # If any error occurs during processing, raise a more informative exception
        raise Exception(f"Error loading PDF file: {str(e)}")

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split text into chunks for better semantic search.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of Document objects containing the text chunks
    """
    # Why do we chunk text?
    # 1. Vector databases work better with smaller text segments
    # 2. Semantic search is more precise with focused content
    # 3. Helps avoid token limits in embedding models
    
    # Create a text splitter with recursive character splitting strategy
    # This is more intelligent than simple splitting as it tries to preserve semantic units
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Maximum size of each chunk (in characters)
        chunk_overlap=chunk_overlap,  # Overlap between chunks to maintain context
        length_function=len,  # Function to measure text length
    )
    
    # Split the text into chunks and convert them to Document objects
    # Document objects contain the text content and can hold metadata
    chunks = text_splitter.create_documents([text])
    
    return chunks

def add_policy(
    policy_text: str, 
    policy_name: str, 
    user_id: str, 
    policy_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a policy document to the vector database.
    
    Args:
        policy_text: Text content of the policy
        policy_name: Name of the policy document
        user_id: ID of the user adding the policy
        policy_id: Optional ID for the policy (generated if not provided)
        metadata: Optional metadata for the policy
        
    Returns:
        ID of the added policy
    """
    # Generate a unique policy ID if not provided by the caller
    # Using UUID (Universally Unique Identifier) to ensure uniqueness
    if policy_id is None:
        policy_id = str(uuid.uuid4())  # Generate a random UUID and convert to string
    
    # Split the policy text into smaller, manageable chunks for better retrieval
    # This helps with more precise semantic search and avoids token limits
    chunks = split_text_into_chunks(policy_text)
    
    # Initialize metadata dictionary if none was provided
    if metadata is None:
        metadata = {}
    
    # Add essential information to metadata that will be stored with each chunk
    # This allows filtering and organizing policies by user and policy name
    metadata["user_id"] = user_id      # Track which user this policy belongs to
    metadata["policy_name"] = policy_name  # Store the name for better organization
    
    # Process each chunk of the policy document separately
    for i, chunk in enumerate(chunks):
        # Create a unique ID for each chunk by combining policy ID and chunk index
        chunk_id = f"{policy_id}_{i}"  # Format: policy_uuid_chunkNumber
        
        # Create chunk-specific metadata by copying the base metadata
        # and adding chunk-specific information
        chunk_metadata = metadata.copy()  # Create a copy to avoid modifying the original
        
        # Ensure all metadata values are primitive types (str, int, float, bool, None)
        # ChromaDB doesn't accept lists or dictionaries as metadata values
        for key, value in list(chunk_metadata.items()):
            if isinstance(value, (list, dict)):
                # Convert lists and dictionaries to strings
                chunk_metadata[key] = str(value)
        
        # Add chunk-specific information
        chunk_metadata["chunk_index"] = i  # Position of this chunk in the document
        chunk_metadata["total_chunks"] = len(chunks)  # Total number of chunks in this policy
        chunk_metadata["policy_id"] = policy_id  # Link back to the parent policy
        
        # Generate vector embeddings for this chunk
        # These embeddings capture the semantic meaning of the text
        try:
            embedding = embeddings.embed_query(chunk.page_content)
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            # Use a simple fallback embedding if the API call fails
            # This won't provide good semantic search but allows the system to function
            embedding = [0.1] * 1536  # Standard OpenAI embedding dimension
        
        # Add the chunk to the ChromaDB collection with its metadata and embedding
        policy_collection.add(
            ids=[chunk_id],  # Unique identifier for this chunk
            embeddings=[embedding],  # Vector representation of the text
            metadatas=[chunk_metadata],  # Associated metadata for filtering and organization
            documents=[chunk.page_content]  # The actual text content
        )
    
    # Return the policy ID so the caller can reference this policy later
    return policy_id

def search_policies(
    query: str, 
    user_id: str, 
    n_results: int = 5,
    policy_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for relevant policies based on a query.
    
    Args:
        query: Search query
        user_id: ID of the user searching for policies
        n_results: Number of results to return
        policy_name: Optional filter by policy name
        
    Returns:
        List of relevant policy chunks with their metadata
    """
    # STEP 1: Convert the text query into a vector embedding
    # This is the key to semantic search - we convert the query to the same vector space as our stored policies
    try:
        print(f"query for policies = {query}")
        query_embedding = embeddings.embed_query(query)  # Uses OpenAI's model to create the embedding
    except Exception as e:
        print(f"Error generating query embedding: {str(e)}")
        # Use a simple fallback embedding if the API call fails
        # This won't provide good semantic search but allows the system to function
        query_embedding = [0.1] * 1536  # Standard OpenAI embedding dimension
    
    # STEP 2: Set up filters to narrow down the search scope
    # We always filter by user_id to ensure data isolation between users
    # ChromaDB requires using operators like $eq for equality comparisons
    where_filter = {"user_id": {"$eq": user_id}}  # Basic filter to only search within this user's policies
    
    # Optionally filter by policy name if specified
    if policy_name:
        # Use $and operator to combine multiple conditions
        where_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"policy_name": {"$eq": policy_name}}
            ]
        }
    
    # STEP 3: Perform the semantic search against the vector database
    # This finds the most similar vectors (policy chunks) to our query vector
    results = policy_collection.query(
        query_embeddings=[query_embedding],  # The vector representation of our search query
        n_results=n_results,  # How many results to return
        where=where_filter  # Metadata-based filtering criteria with proper operators
    )
    
    # STEP 4: Format the results into a more user-friendly structure
    formatted_results = []
    
    # Check if we got any results back
    if results["documents"]:
        # Process each result and create a structured dictionary with all relevant information
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append({
                "content": doc,  # The actual text content of the policy chunk
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},  # Associated metadata
                "id": results["ids"][0][i] if results["ids"] else None,  # Unique ID of the chunk
                "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None  # Similarity score (lower is better)
            })
    
    # Return the formatted results
    return formatted_results

def delete_user_policies(user_id: str, policy_name: Optional[str] = None) -> int:
    """
    Delete all policies associated with a user.
    
    Args:
        user_id: ID of the user whose policies should be deleted
        policy_name: Optional filter to delete only policies with this name
        
    Returns:
        Number of deleted policy chunks
    """
    # STEP 1: Create a filter to identify which policies to delete
    # We always filter by user_id to ensure we only delete this user's policies
    # ChromaDB requires using operators like $eq for equality comparisons
    where_filter = {"user_id": {"$eq": user_id}}  # Basic filter for user isolation
    
    # Optionally narrow down to a specific policy name
    if policy_name:
        # Use $and operator to combine multiple conditions
        where_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"policy_name": {"$eq": policy_name}}
            ]
        }
    
    # STEP 2: Retrieve all policy chunks that match our filter criteria
    # This gives us the IDs we need to delete
    results = policy_collection.get(where=where_filter)  # Get all matching chunks
    
    # STEP 3: Delete the identified chunks from the collection
    if results["ids"]:
        # If we found matching chunks, delete them all at once
        policy_collection.delete(ids=results["ids"])  # Bulk delete operation
        return len(results["ids"])  # Return how many chunks were deleted
    
    # If no matching policies were found, return 0
    return 0

def get_all_user_policies(user_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all policies associated with a user, grouped by policy_id.
    
    Args:
        user_id: ID of the user
        
    Returns:
        Dictionary mapping policy IDs to lists of policy chunks
    """
    # STEP 1: Retrieve all policy chunks for this specific user
    # This ensures data isolation between users
    # ChromaDB requires using operators like $eq for equality comparisons
    results = policy_collection.get(where={"user_id": {"$eq": user_id}})  # Get all chunks for this user
    
    # STEP 2: Organize the chunks by policy_id to reconstruct complete policies
    # This helps present the data in a more structured way
    policies = {}  # Dictionary to hold policies grouped by policy_id
    
    if results["ids"]:
        # Process each chunk we retrieved
        for i, chunk_id in enumerate(results["ids"]):
            # Get the metadata for this chunk
            metadata = results["metadatas"][i]
            
            # Extract the policy_id this chunk belongs to
            policy_id = metadata.get("policy_id")
            
            # If this is the first chunk we've seen for this policy_id, initialize a list
            if policy_id not in policies:
                policies[policy_id] = []  # Create a new list for this policy
            
            # Add this chunk to the appropriate policy list
            policies[policy_id].append({
                "content": results["documents"][i],  # The actual text content
                "metadata": metadata,  # All associated metadata
                "id": chunk_id  # The unique ID of this chunk
            })
    
    # Return the organized policy data
    return policies

def load_policy_from_pdf(
    pdf_path: str, 
    policy_name: str, 
    user_id: str,
    policy_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Load a policy from a PDF file and add it to the vector database.
    
    Args:
        pdf_path: Path to the PDF file
        policy_name: Name of the policy document
        user_id: ID of the user adding the policy
        policy_id: Optional ID for the policy (generated if not provided)
        metadata: Optional metadata for the policy
        
    Returns:
        ID of the added policy
    """
    # STEP 1: Extract text content from the PDF file
    # This uses our load_pdf helper function to handle the PDF parsing
    policy_text = load_pdf(pdf_path)  # Convert PDF to plain text
    
    # STEP 2: Add the extracted text to our vector database
    # This delegates to our add_policy function which handles chunking and embedding
    return add_policy(policy_text, policy_name, user_id, policy_id, metadata)  # Pass all parameters to add_policy


def load_policy_from_uploaded_file(
    uploaded_file, 
    policy_name: str, 
    user_id: str,
    policy_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Load a policy from a Streamlit uploaded file and add it to the vector database.
    
    Args:
        uploaded_file: Streamlit UploadedFile object containing the PDF data
        policy_name: Name of the policy document
        user_id: ID of the user adding the policy
        policy_id: Optional ID for the policy (generated if not provided)
        metadata: Optional metadata for the policy
        
    Returns:
        ID of the added policy
    """
    # STEP 1: Extract text content from the uploaded PDF file
    # Create a PDF reader from the uploaded file's bytes
    pdf_reader = pypdf.PdfReader(uploaded_file)
    
    # Extract text from all pages
    policy_text = ""
    for page in pdf_reader.pages:
        policy_text += page.extract_text() + "\n"
    
    # STEP 2: Add the extracted text to our vector database
    # This delegates to our add_policy function which handles chunking and embedding
    return add_policy(policy_text, policy_name, user_id, policy_id, metadata)  # Pass all parameters to add_policy

def add_policy_without_embeddings(policy_text: str, policy_name: str, user_id: str, policy_id: str) -> str:
    """
    Add a policy document to the vector database without using embeddings.
    This is a fallback method when the OpenAI API is unavailable.
    
    Args:
        policy_text: Text content of the policy
        policy_name: Name of the policy document
        user_id: ID of the user adding the policy
        policy_id: ID for the policy
        
    Returns:
        ID of the added policy
    """
    try:
        # Create a unique ID for this policy chunk
        chunk_id = f"{policy_id}_0"  # Just one chunk
        
        # Create metadata
        metadata = {
            "user_id": user_id,
            "policy_name": policy_name,
            "policy_id": policy_id,
            "chunk_index": 0,
            "total_chunks": 1,
            "category": "General" if "reimbursement" in policy_text.lower() else "Meals",
            "applies_to": "All expenses" if "reimbursement" in policy_text.lower() else "Meals"
        }
        
        # Create a simple embedding (this won't be good for semantic search
        # but will allow the system to function without API access)
        simple_embedding = [0.1] * 1536  # Standard OpenAI embedding dimension
        
        # Add the chunk to the ChromaDB collection
        policy_collection.add(
            ids=[chunk_id],
            embeddings=[simple_embedding],
            metadatas=[metadata],
            documents=[policy_text]
        )
        
        return policy_id
    except Exception as e:
        return policy_id

def delete_user_policies(user_id: str) -> int:
    """
    Delete all policies associated with a specific user.
    
    Args:
        user_id: ID of the user whose policies should be deleted
        
    Returns:
        Number of policies deleted
    """
    # Define a filter to match all documents with the specified user_id
    user_filter = {"user_id": {"$eq": user_id}}
    
    # Get the IDs of all documents that match the filter
    matching_docs = policy_collection.get(where=user_filter)
    matching_ids = matching_docs.get("ids", [])
    
    # If there are matching documents, delete them
    if matching_ids:
        policy_collection.delete(ids=matching_ids)
        
    return len(matching_ids)


def delete_specific_policies(user_id: str, policy_names: List[str]) -> int:
    """
    Delete specific policies by name for a user.
    
    Args:
        user_id: ID of the user whose policies should be deleted
        policy_names: List of policy names to delete
        
    Returns:
        Number of policies deleted
    """
    deleted_count = 0
    
    for policy_name in policy_names:
        # Define a filter to match documents with the specified user_id and policy_name
        policy_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"policy_name": {"$eq": policy_name}}
            ]
        }
        
        # Get the IDs of all documents that match the filter
        matching_docs = policy_collection.get(where=policy_filter)
        matching_ids = matching_docs.get("ids", [])
        
        # If there are matching documents, delete them
        if matching_ids:
            policy_collection.delete(ids=matching_ids)
            deleted_count += len(matching_ids)
            print(f"Deleted {len(matching_ids)} chunks of policy '{policy_name}' for user {user_id}")
    
    return deleted_count

def get_relevant_policies(expense_data: Dict[str, Any], user_id: str, n_results: int = 20) -> List[Dict[str, Any]]:
    """
    Get policies relevant to an expense report.
    
    Args:
        expense_data: Structured expense report data from WorkflowState
        user_id: ID of the user
        n_results: Number of results to return
        
    Returns:
        List of relevant policy chunks
    """
    try:
        # Initialize query with an instructional prompt that emphasizes key policy areas
        query = "Find policy rules that specifically address and regulate the following expense items and their compliance requirements. Pay special attention to policies about MEAL EXPENSES and MAXIMUM REIMBURSEMENT LIMITS: "
        
        # Simple approach: directly convert expense_data dictionary to a string representation
        # Handle top-level fields first
        top_level_items = []
        for key, value in expense_data.items():
            # Skip expense_items as we'll process them separately
            if key == 'expense_items':
                continue
                
            # Add the key-value pair to our query
            if value is not None:
                top_level_items.append(f"{key}: {value}")
        
        # Add all top-level items to the query
        if top_level_items:
            query += ", ".join(top_level_items)
        
        # Process expense items if they exist
        if 'expense_items' in expense_data and expense_data['expense_items']:
            query += ". Expense items (pay special attention to meal expenses): "
            
            # Process each expense item
            item_descriptions = []
            for i, item in enumerate(expense_data['expense_items']):
                item_parts = []
                for item_key, item_value in item.items():
                    if item_value is not None:
                        item_parts.append(f"{item_key}: {item_value}")
                
                # Join all parts of this item
                if item_parts:
                    item_str = f"Item {i+1}: {', '.join(item_parts)}"
                    item_descriptions.append(item_str)
            
            # Join all item descriptions
            if item_descriptions:
                query += "; ".join(item_descriptions)
        
        # Special handling for USD currency if present
        if 'currency' in expense_data and expense_data['currency'] == 'USD':
            query += ". This expense report includes USD currency transactions. Find relevant foreign currency and international travel expense policies."
        
        # Check for USD in expense items
        has_usd_items = False
        if 'expense_items' in expense_data:
            for item in expense_data['expense_items']:
                if 'currency_code' in item and item['currency_code'] == 'USD':
                    has_usd_items = True
                    break
        
        if has_usd_items:
            query += ". Some expense items are in USD currency. Find relevant foreign currency policies."
        
        query += "Consider policy items that address value limits and types of expenses covered. IMPORTANT: Specifically identify any policies that restrict meal expenses or set maximum reimbursement limits."
        
        # STEP 3: Use the query to search for relevant policies
        # Make sure we're calling search_policies with the correct parameters
        # The function signature is: search_policies(query, user_id, n_results, policy_name)
        results = search_policies(query=query, user_id=user_id, n_results=n_results)
        
        # Return the results
        return results
    except Exception as e:
        print(f"Error in get_relevant_policies: {str(e)}")
        return []
