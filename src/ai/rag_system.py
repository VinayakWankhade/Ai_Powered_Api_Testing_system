"""
Retrieval-Augmented Generation (RAG) system for API testing.
"""

import os
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from ..database.models import DocumentationStore, APISpecification
from ..database.connection import get_db_session
from ..utils.logger import get_logger

logger = get_logger(__name__)

class RAGSystem:
    """
    Retrieval-Augmented Generation system for retrieving relevant documentation
    and examples for test generation.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.db = get_db_session()
        
        # Initialize ChromaDB
        chromadb_path = os.getenv("CHROMADB_PATH", "./data/chromadb")
        self.chroma_client = chromadb.PersistentClient(
            path=chromadb_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="api_documentation",
            metadata={"description": "API documentation and examples for test generation"}
        )
        
        logger.info(f"RAG system initialized with model: {embedding_model_name}")

    def add_documents_to_vector_store(
        self,
        api_spec_id: int,
        force_reindex: bool = False
    ) -> int:
        """
        Add documents from database to the vector store.
        
        Args:
            api_spec_id: ID of the API specification
            force_reindex: Whether to force reindexing of existing documents
            
        Returns:
            Number of documents added/updated
        """
        try:
            # Get all documentation for the API specification
            docs = self.db.query(DocumentationStore).filter(
                DocumentationStore.api_spec_id == api_spec_id
            ).all()
            
            documents_added = 0
            
            for doc in docs:
                doc_id = f"api_{api_spec_id}_doc_{doc.id}"
                
                # Check if document already exists in vector store
                existing_docs = self.collection.get(ids=[doc_id])
                if existing_docs['ids'] and not force_reindex:
                    continue
                
                # Generate embedding
                embedding = self.embedding_model.encode(doc.content).tolist()
                
                # Prepare metadata
                metadata = {
                    "api_spec_id": api_spec_id,
                    "doc_id": doc.id,
                    "doc_type": doc.doc_type,
                    "title": doc.title,
                    "source": doc.source or "unknown",
                    "created_at": doc.created_at.isoformat() if doc.created_at else ""
                }
                
                # Add structured content if available
                if doc.structured_content:
                    metadata["structured_content"] = json.dumps(doc.structured_content)
                
                # Add or update in ChromaDB
                self.collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[doc.content],
                    metadatas=[metadata]
                )
                
                # Update embedding in database
                doc.embedding = embedding
                doc.embedding_model = self.embedding_model_name
                
                documents_added += 1
            
            self.db.commit()
            logger.info(f"Added/updated {documents_added} documents to vector store for API {api_spec_id}")
            return documents_added
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            self.db.rollback()
            raise

    def retrieve_relevant_docs(
        self,
        query: str,
        api_spec_id: Optional[int] = None,
        doc_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: Search query
            api_spec_id: Optional API specification ID to filter by
            doc_types: Optional list of document types to filter by
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with metadata and similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare filters
            where_filters = {}
            if api_spec_id:
                where_filters["api_spec_id"] = api_spec_id
            
            if doc_types:
                where_filters["doc_type"] = {"$in": doc_types}
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filters if where_filters else None
            )
            
            # Format results
            relevant_docs = []
            for i in range(len(results['ids'][0])):
                doc_data = {
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": 1.0 - results['distances'][0][i]  # Convert distance to similarity
                }
                relevant_docs.append(doc_data)
            
            # Update retrieval count in database
            self._update_retrieval_counts([doc['metadata']['doc_id'] for doc in relevant_docs])
            
            logger.debug(f"Retrieved {len(relevant_docs)} relevant documents for query: {query[:100]}...")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant documents: {str(e)}")
            return []

    def retrieve_endpoint_examples(
        self,
        endpoint_path: str,
        method: str,
        api_spec_id: int,
        example_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve specific examples for an endpoint.
        
        Args:
            endpoint_path: API endpoint path
            method: HTTP method
            api_spec_id: API specification ID
            example_types: Types of examples to retrieve
            
        Returns:
            List of relevant examples
        """
        try:
            query = f"{method} {endpoint_path}"
            
            # Search for endpoint-specific documentation
            endpoint_docs = self.retrieve_relevant_docs(
                query=query,
                api_spec_id=api_spec_id,
                doc_types=["endpoint_doc", "example", "test_case"],
                top_k=10
            )
            
            # Filter and enhance results
            examples = []
            for doc in endpoint_docs:
                metadata = doc['metadata']
                
                # Check if this document is specifically about our endpoint
                if 'structured_content' in metadata:
                    try:
                        structured = json.loads(metadata['structured_content'])
                        if (structured.get('path') == endpoint_path and 
                            structured.get('method', '').upper() == method.upper()):
                            examples.append({
                                **doc,
                                "relevance": "exact_match",
                                "structured_data": structured
                            })
                        elif structured.get('path') == endpoint_path:
                            examples.append({
                                **doc,
                                "relevance": "path_match",
                                "structured_data": structured
                            })
                    except json.JSONDecodeError:
                        pass
                
                # Include high similarity docs even if no structured match
                if doc['similarity_score'] > 0.8:
                    examples.append({
                        **doc,
                        "relevance": "high_similarity"
                    })
            
            logger.debug(f"Retrieved {len(examples)} examples for {method} {endpoint_path}")
            return examples[:5]  # Return top 5 most relevant
            
        except Exception as e:
            logger.error(f"Failed to retrieve endpoint examples: {str(e)}")
            return []

    def retrieve_error_handling_docs(
        self,
        error_type: str,
        api_spec_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documentation related to error handling.
        
        Args:
            error_type: Type of error (e.g., "authentication", "validation", "server_error")
            api_spec_id: Optional API specification ID
            
        Returns:
            List of relevant error handling documents
        """
        try:
            query = f"error handling {error_type} troubleshooting debug"
            
            return self.retrieve_relevant_docs(
                query=query,
                api_spec_id=api_spec_id,
                doc_types=["error_guide", "endpoint_doc", "troubleshooting"],
                top_k=3
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve error handling docs: {str(e)}")
            return []

    def retrieve_similar_test_patterns(
        self,
        test_description: str,
        api_spec_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar test patterns and examples.
        
        Args:
            test_description: Description of the test being generated
            api_spec_id: Optional API specification ID
            
        Returns:
            List of similar test patterns
        """
        try:
            return self.retrieve_relevant_docs(
                query=test_description,
                api_spec_id=api_spec_id,
                doc_types=["test_case", "example", "test_pattern"],
                top_k=5
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar test patterns: {str(e)}")
            return []

    def add_custom_document(
        self,
        content: str,
        title: str,
        doc_type: str,
        api_spec_id: Optional[int] = None,
        source: str = "manual",
        structured_content: Optional[Dict[str, Any]] = None
    ) -> DocumentationStore:
        """
        Add a custom document to the knowledge base.
        
        Args:
            content: Document content
            title: Document title
            doc_type: Type of document
            api_spec_id: Optional API specification ID
            source: Source of the document
            structured_content: Optional structured data
            
        Returns:
            Created DocumentationStore record
        """
        try:
            # Create database record
            doc = DocumentationStore(
                api_spec_id=api_spec_id,
                title=title,
                doc_type=doc_type,
                source=source,
                content=content,
                structured_content=structured_content
            )
            
            self.db.add(doc)
            self.db.commit()
            self.db.refresh(doc)
            
            # Add to vector store
            if api_spec_id:
                self.add_documents_to_vector_store(api_spec_id, force_reindex=False)
            else:
                # Add individual document
                doc_id = f"custom_doc_{doc.id}"
                embedding = self.embedding_model.encode(content).tolist()
                
                metadata = {
                    "doc_id": doc.id,
                    "doc_type": doc_type,
                    "title": title,
                    "source": source,
                    "api_spec_id": api_spec_id or -1
                }
                
                if structured_content:
                    metadata["structured_content"] = json.dumps(structured_content)
                
                self.collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata]
                )
                
                doc.embedding = embedding
                doc.embedding_model = self.embedding_model_name
                self.db.commit()
            
            logger.info(f"Added custom document: {title}")
            return doc
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to add custom document: {str(e)}")
            raise

    def search_by_keywords(
        self,
        keywords: List[str],
        api_spec_id: Optional[int] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents by keywords.
        
        Args:
            keywords: List of keywords to search for
            api_spec_id: Optional API specification ID
            top_k: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            query = " ".join(keywords)
            return self.retrieve_relevant_docs(
                query=query,
                api_spec_id=api_spec_id,
                top_k=top_k
            )
            
        except Exception as e:
            logger.error(f"Failed to search by keywords: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze types
            sample_docs = self.collection.peek(limit=100)
            doc_types = {}
            api_specs = set()
            
            for metadata in sample_docs['metadatas']:
                doc_type = metadata.get('doc_type', 'unknown')
                api_spec_id = metadata.get('api_spec_id', -1)
                
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                if api_spec_id != -1:
                    api_specs.add(api_spec_id)
            
            return {
                "total_documents": count,
                "document_types": doc_types,
                "api_specs_count": len(api_specs),
                "embedding_model": self.embedding_model_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}

    def _update_retrieval_counts(self, doc_ids: List[int]):
        """Update retrieval counts for documents."""
        try:
            if not doc_ids:
                return
            
            # Update retrieval count and last retrieved timestamp
            docs = self.db.query(DocumentationStore).filter(
                DocumentationStore.id.in_(doc_ids)
            ).all()
            
            from datetime import datetime
            for doc in docs:
                doc.retrieval_count += 1
                doc.last_retrieved_at = datetime.utcnow()
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update retrieval counts: {str(e)}")

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'db'):
            self.db.close()
