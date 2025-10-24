from pymongo import MongoClient
from typing import List, Dict

class CosmosVectorDB:
    """
    CosmosVectorDB handles all vector database operations.

    Key responsibilities:
    - Connect to Azure Cosmos DB for MongoDB vCore
    - Create and manage vector indexes
    - Store document embeddings
    - Perform similarity search
    """

    def __init__(self, connection_string: str, database_name: str, 
                 collection_name: str, embedding_dimensions: int = 1536,
                 vector_index_type: str = "ivf"):
        """
        Initialize connection to Cosmos DB for MongoDB vCore.

        Args:
            connection_string (str): MongoDB connection string from Azure Portal
            database_name (str): Name of the database (e.g., "hr_knowledge_base")
            collection_name (str): Name of collection (e.g., "hr_policies")
            embedding_dimensions (int): Embedding vector size (1536 for ada-002)
            vector_index_type (str): Index type (ivf, hnsw, etc.) - default: ivf
        """
        print(f" Connecting to Cosmos DB...")

        # Connect to MongoDB
        self.client = MongoClient(connection_string)
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

        self.embedding_dimensions = embedding_dimensions
        self.vector_index_type = vector_index_type

        # Create vector index for efficient similarity search
        self._create_vector_index()

        print(f"✓ Connected to database: {database_name}")
        print(f"✓ Using collection: {collection_name}")
        print(f"✓ Vector index type: {vector_index_type.upper()}")

    def _create_vector_index(self):
        """
        Create a vector index for efficient similarity search.

        Vector Index Types:
        - ivf: Inverted File Index (compatible with all cluster tiers)
        - hnsw: Hierarchical Navigable Small World (requires higher tier, faster)
        - diskann: Disk-based approximate nearest neighbor

        This is like creating an index in a database for faster queries.
        Without it, searching would be very slow!
        """
        try:
            # Check if index already exists
            existing_indexes = list(self.collection.list_indexes())

            # Look for vector index
            vector_index_exists = any(
                'embedding' in idx.get('key', {}) 
                for idx in existing_indexes
            )

            if not vector_index_exists:
                print(f" Creating vector index ({self.vector_index_type})...")

                # Prepare index kind with proper prefix
                index_kind = self.vector_index_type
                if not index_kind.startswith('vector-'):
                    index_kind = f'vector-{index_kind}'

                # Configure index options based on type
                cosmos_search_options = {
                    'kind': index_kind,
                    'dimensions': self.embedding_dimensions,
                    'similarity': 'COS'  # Cosine similarity
                }

                # Add type-specific options
                if 'ivf' in index_kind:
                    cosmos_search_options['numLists'] = 100  # Number of clusters
                elif 'hnsw' in index_kind:
                    cosmos_search_options['m'] = 16  # Bi-directional links
                    cosmos_search_options['efConstruction'] = 64  # Construction parameter

                # Create vector search index
                self.database.command({
                    'createIndexes': self.collection.name,
                    'indexes': [
                        {
                            'name': 'vector_index',
                            'key': {
                                "embedding": "cosmosSearch"
                            },
                            'cosmosSearchOptions': cosmos_search_options
                        }
                    ]
                })
                print("✓ Vector index created successfully")
            else:
                print("✓ Vector index already exists")

        except Exception as e:
            error_msg = str(e)

            # Handle HNSW not supported error - fallback to IVF
            if "hnsw index is not supported" in error_msg:
                print(f"⚠️  HNSW index not supported on this cluster tier")
                print(f"   Falling back to IVF index...")

                try:
                    # Retry with IVF index
                    self.database.command({
                        'createIndexes': self.collection.name,
                        'indexes': [
                            {
                                'name': 'vector_index',
                                'key': {
                                    "embedding": "cosmosSearch"
                                },
                                'cosmosSearchOptions': {
                                    'kind': 'vector-ivf',
                                    'numLists': 100,
                                    'dimensions': self.embedding_dimensions,
                                    'similarity': 'COS'
                                }
                            }
                        ]
                    })
                    self.vector_index_type = 'ivf'
                    print("✓ IVF vector index created successfully")
                except Exception as e2:
                    print(f"⚠️  Warning: Could not create IVF index: {e2}")
                    print("   Index might already exist or will be created automatically")

            # Handle index already exists
            elif "already exists" in error_msg or "IndexAlreadyExists" in error_msg:
                print("✓ Vector index already exists")

            # Other errors
            else:
                print(f"⚠️  Warning: Could not create vector index: {e}")
                print("   Index might already exist or will be created automatically")

    def insert_documents(self, documents: List[Dict]) -> int:
        """
        Insert documents with embeddings into Cosmos DB.

        Args:
            documents (List[Dict]): List of documents to insert
                Each document should have:
                - content: The text content
                - embedding: The vector embedding
                - source: Source filename
                - page: Page number
                - chunk_index: Chunk identifier

        Returns:
            int: Number of documents successfully inserted

        Example:
            documents = [
                {
                    "content": "Vacation policy text...",
                    "embedding": [0.123, -0.456, ...],
                    "source": "hr_policy.pdf",
                    "page": 5,
                    "chunk_index": 0
                }
            ]
        """
        if not documents:
            print("⚠️  No documents to insert")
            return 0

        try:
            # Insert all documents at once (batch operation)
            result = self.collection.insert_many(documents)
            inserted_count = len(result.inserted_ids)

            print(f"✓ Inserted {inserted_count} documents into Cosmos DB")
            return inserted_count

        except Exception as e:
            print(f"❌ Error inserting documents: {e}")
            return 0

    def vector_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Perform vector similarity search to find relevant documents.

        How it works:
        1. Takes a query embedding (vector of numbers)
        2. Compares it with all document embeddings in the database
        3. Returns the most similar documents

        This is the CORE of retrieval - finding relevant information!

        Args:
            query_embedding (List[float]): Embedding vector of the query
            top_k (int): Number of top results to return (default: 5)

        Returns:
            List[Dict]: List of most relevant documents with similarity scores

        Example:
            Query: "What is the vacation policy?"
            Returns: Top 5 most relevant document chunks about vacation
        """
        try:
            # MongoDB aggregation pipeline for vector search
            # This is a special query that Cosmos DB understands
            pipeline = [
                {
                    # Vector search stage
                    "$search": {
                        "cosmosSearch": {
                            "vector": query_embedding,  # The query vector
                            "path": "embedding",         # Field containing embeddings
                            "k": top_k                   # Number of results
                        },
                        "returnStoredSource": True      # Return full documents
                    }
                },
                {
                    # Project stage - select which fields to return
                    "$project": {
                        "_id": 0,                    # Don't return MongoDB ID
                        "content": {"$ifNull": ["$content", "$text_chunk"]},  # Support both field names
                        "source": 1,                 # Return source filename
                        "page": 1,                   # Return page number
                        "chunk_index": 1,            # Return chunk index
                        "similarity_score": {        # Calculate similarity score
                            "$meta": "searchScore"
                        }
                    }
                }
            ]

            # Execute the search
            results = list(self.collection.aggregate(pipeline))

            print(f"✓ Retrieved {len(results)} relevant documents")
            return results

        except Exception as e:
            print(f"❌ Error during vector search: {e}")
            return []

    def count_documents(self) -> int:
        """
        Count total documents in the collection.

        Returns:
            int: Total number of documents stored
        """
        try:
            count = self.collection.count_documents({})
            return count
        except Exception as e:
            print(f"❌ Error counting documents: {e}")
            return 0

    def delete_all_documents(self):
        """
        Delete all documents from the collection.

        WARNING: This will remove all stored data!
        Use carefully, typically only for testing or resetting.
        """
        try:
            result = self.collection.delete_many({})
            print(f"️  Deleted {result.deleted_count} documents")
            return result.deleted_count
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            return 0

    def close(self):
        """
        Close the database connection.

        Good practice: Always close connections when done!
        """
        self.client.close()
        print("✓ Database connection closed")
 