#!/usr/bin/env python3
"""
Complete RAG pipeline for Vietnamese legal documents.

Components:
1. Embedding: Vietnamese bi-encoder
2. Vector Store: ChromaDB (local, persistent)
3. LLM: Gemma 4 E2B (via llama.cpp/Ollama)
4. Orchestration: LangChain

Usage:
    uv run python rag/pipeline.py
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import click
from tqdm import tqdm

# Optional imports with nice error messages
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Installing ChromaDB...")
    os.system("uv pip install chromadb")
    import chromadb
    from chromadb.config import Settings

try:
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import LlamaCpp
    from langchain.chains import RetrievalQA
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    print("Installing LangChain...")
    os.system("uv pip install langchain langchain-community")
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import LlamaCpp
    from langchain.chains import RetrievalQA


class LegalRAGPipeline:
    """Vietnamese Legal RAG Pipeline."""

    def __init__(
        self,
        embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder",
        llm_path: str = None,
        persist_directory: str = "./chroma_db",
    ):
        """Initialize RAG pipeline components."""
        self.persist_directory = persist_directory

        # Vietnamese embeddings
        print(f"Loading embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},  # Use CPU for compatibility
            encode_kwargs={"normalize_embeddings": True},
        )

        # Vector store
        self.vectorstore = None
        self.qa_chain = None
        self.llm = None

    def index_documents(self, data_path: str):
        """Index documents into vector store."""
        print(f"\n=== Indexing Documents ===")
        print(f"Source: {data_path}")

        data_path = Path(data_path)

        # Load documents based on format
        documents = []
        if data_path.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(data_path)

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
                text = row.get("content_markdown") or row.get("content_text", "")
                if text:
                    documents.append({
                        "text": text,
                        "metadata": {
                            "doc_id": row.get("doc_id", ""),
                            "title": row.get("title", ""),
                            "url": row.get("url", ""),
                            "doc_type": row.get("doc_type", ""),
                            "doc_number": row.get("doc_number", ""),
                        }
                    })
        elif data_path.suffix == ".jsonl":
            import json
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    text = doc.get("content_markdown") or doc.get("content_text", "")
                    if text:
                        documents.append({
                            "text": text,
                            "metadata": {
                                "doc_id": doc.get("doc_id", ""),
                                "title": doc.get("title", ""),
                                "url": doc.get("url", ""),
                            }
                        })

        print(f"Loaded {len(documents)} documents")

        # Create vector store
        print("Creating vector store (this may take a while)...")
        self.vectorstore = Chroma.from_texts(
            texts=[d["text"] for d in documents],
            embedding=self.embeddings,
            metadatas=[d["metadata"] for d in documents],
            persist_directory=self.persist_directory,
        )
        self.vectorstore.persist()

        print(f"✅ Indexed {len(documents)} documents")
        print(f"   Persisted to: {self.persist_directory}")

    def load_vectorstore(self):
        """Load existing vector store."""
        print(f"\n=== Loading Vector Store ===")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        print(f"✅ Loaded from: {self.persist_directory}")

        # Get collection info
        collection = self.vectorstore._collection
        print(f"   Documents: {collection.count()}")

    def setup_llm(self, model_path: str, n_ctx: int = 4096):
        """Setup LLM for generation."""
        print(f"\n=== Setting up LLM ===")
        print(f"Model: {model_path}")

        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                n_ctx=n_ctx,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                verbose=False,
            )
            print("✅ LLM loaded successfully")
        except Exception as e:
            print(f"⚠️  LLM loading failed: {e}")
            print("   Make sure GGUF model is downloaded and llama.cpp is installed")
            self.llm = None

    def setup_qa_chain(self):
        """Setup RAG question-answering chain."""
        if not self.vectorstore:
            raise ValueError("Vector store not loaded. Call load_vectorstore() first.")
        if not self.llm:
            raise ValueError("LLM not setup. Call setup_llm() first.")

        print("\n=== Setting up QA Chain ===")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 passages
            ),
            return_source_documents=True,
        )
        print("✅ QA chain ready")

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline."""
        if not self.qa_chain:
            raise ValueError("QA chain not setup. Call setup_qa_chain() first.")

        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")

        result = self.qa_chain({"query": question})

        # Print answer
        print("Answer:")
        print(result["result"])

        # Print sources
        print("\nSources:")
        for i, doc in enumerate(result["source_documents"], 1):
            metadata = doc.metadata
            print(f"\n[{i}] {metadata.get('title', 'N/A')}")
            print(f"    Type: {metadata.get('doc_type', 'N/A')}")
            print(f"    URL: {metadata.get('url', 'N/A')}")
            print(f"    Preview: {doc.page_content[:150]}...")

        return result

    def interactive(self):
        """Run interactive Q&A session."""
        print("\n" + "="*60)
        print("Vietnamese Legal RAG - Interactive Mode")
        print("="*60)
        print("\nAsk questions about Vietnamese law (or 'quit' to exit):\n")

        while True:
            try:
                question = input("\n❓ Question: ").strip()
                if not question:
                    continue
                if question.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                self.query(question)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


@click.command()
@click.option("--data", "-d", default="data/processed/documents.parquet", help="Path to documents")
@click.option("--model", "-m", help="Path to GGUF model file")
@click.option("--persist-dir", "-p", default="./chroma_db", help="Vector store directory")
@click.option("--rebuild", is_flag=True, help="Rebuild vector store")
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
@click.option("--query", "-q", help="Single query to run")
def main(data: str, model: str, persist_dir: str, rebuild: bool, interactive: bool, query: str):
    """Run RAG pipeline."""
    rag = LegalRAGPipeline(persist_directory=persist_dir)

    # Build or load vector store
    if rebuild or not Path(persist_dir).exists():
        rag.index_documents(data)
    else:
        rag.load_vectorstore()

    # Setup LLM if model provided
    if model:
        rag.setup_llm(model)
        rag.setup_qa_chain()

        # Run queries
        if interactive:
            rag.interactive()
        elif query:
            rag.query(query)
    else:
        print("\n⚠️  No model provided. Use --model to specify GGUF path.")
        print("   Vector store is ready for retrieval-only mode.")

        # Simple retrieval test
        if query:
            print("\n=== Retrieval-only mode ===")
            docs = rag.vectorstore.similarity_search(query, k=3)
            for i, doc in enumerate(docs, 1):
                print(f"\n[{i}] {doc.metadata.get('title', 'N/A')}")
                print(f"    {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
