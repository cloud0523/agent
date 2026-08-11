import os

import chromadb
class VectorStore:
      def __init__(self, persist_dir: str, collection_name: str = "rag_documents"):
          """初始化 ChromaDB 客户端，创建或打开集合"""
          os.makedirs(persist_dir, exist_ok=True)
          self.client = chromadb.PersistentClient(path=persist_dir)
          # 获取或创建集合，指定 cosine 距离度量
          self.collection = self.client.get_or_create_collection(
              name=collection_name,
              metadata={"hnsw:space": "cosine"}
          )
          print(f"向量数据库已完成初始化，数据存储在：{persist_dir}")
      def add_chunks(self, doc_id: str, chunks: list, embeddings: list):
          """将文档的文本块和向量一并存入 ChromaDB"""
          # 每个 chunk 需要: id, embedding, document (文本), metadata (doc_id, chunk_index)
          chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
          metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]  # 默认使用文档来源元数据
          self.collection.add(
              ids=chunk_ids,
              embeddings=embeddings,
              documents=chunks,
              metadatas=metadatas
          )
          print(f"成功写入{len(chunks)}个文本块，文档ID: {doc_id}")

      def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
          """用查询向量搜索最相似的 top_k 个文本块"""
          # 返回格式: [{"chunk_id": ..., "text": ..., "doc_id": ..., "score": ...}, ...]
          results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
          #解析并格式化返回结果
          formatted_results = []
          if results['ids'] and len(results['ids']) > 0:
              for i in range(len(results['ids'][0])):
                  chunk_id = results['ids'][0][i]
                  metadata = results['metadatas'][0][i] or {}
                  safe_doc_id = metadata.get("doc_id", "unknown")
                  distance = results['distances'][0][i]
                  score = 1.0 - distance / 2.0

                  formatted_results.append({
                      "chunk_id": chunk_id,
                      "text": results['documents'][0][i],
                      "doc_id": safe_doc_id,
                      "score": score,
                      "distance": distance,
                      "chunk_index": metadata.get("chunk_index"),
                  })
          return formatted_results

      def delete_document(self, doc_id: str):
          """删除某个文档的所有向量"""
          try:
              self.collection.delete(where={"doc_id": doc_id})
              print(f"已删除文档ID: {doc_id} 关联的所有向量")
          except Exception as e:
              print(f"删除失败: {e}")