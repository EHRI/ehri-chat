from sentence_transformers import SentenceTransformer
from ehri_graph_rag.rag.sparql_manager import RagSPARQLManager, GraphRagSPARQLManager, Entity
import faiss
import numpy as np
import json
import os
import time
import logging
logger = logging.getLogger("ehri_graph_rag")

class EmbeddingsManager:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

class RagEmbeddingsManager(EmbeddingsManager):
    def __init__(self):
        super().__init__()
        self.sparql_manager = RagSPARQLManager()
        self.faiss_index_file = "conf/faiss/rag_index_ehri.faiss"
        self.chunks_file = "conf/faiss/rag_chunks.json"
        if(os.path.isfile(self.faiss_index_file) and os.path.isfile(self.chunks_file)):
            self.index, self.chunks = self.retrieve_faiss_index()

    def encode_chunks_and_persist_to_index(self, chunks, index, chunks_list):
        chunks_list.extend(chunks)    
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        index.add(np.array(embeddings))

    def retrieve_relevant_chunks(self, query, top_k=6):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        logger.debug(f"Distances for RAG retrieval: {distances}")
        return [self.chunks[indices[0][i]] for i in range(len(indices[0]))] #if distances[0][i] < 0.94]

    def retrieve_contents(self, index, chunks_list):
        self.encode_chunks_and_persist_to_index(self.sparql_manager.load_countries_chunks(), index, chunks_list)
        self.encode_chunks_and_persist_to_index(list(self.sparql_manager.load_institutions_chunks()), index, chunks_list)

        for step in range(40):
            self.retrieve_archival_descriptions_contents(step, index, chunks_list)
    
    def retrieve_archival_descriptions_contents(self, step, index, chunks_list):
        completed = False
        while not completed:
            try:
                partial_contents = list(self.sparql_manager.load_archival_descriptions_chunks(step=step, step_size=10000))
                self.encode_chunks_and_persist_to_index(partial_contents, index, chunks_list)
                completed = True
            except Exception as e:
                logger.error(f"Error retrieving archival descriptions chunks for step {step}: {e}. Waiting 1 min and retrying...")
                time.sleep(60)

    def retrieve_faiss_index(self):
        index = faiss.read_index(self.faiss_index_file)
        with open(self.chunks_file, "r") as f:
            chunks = json.load(f)
        return index, chunks

    def create_faiss_index(self):
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            f.write("")

        chunks_list = []
        index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())

        self.retrieve_contents(index, chunks_list)

        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_list, f)

        faiss.write_index(index, self.faiss_index_file)

class GraphRagEmbeddingsManager(EmbeddingsManager):
    def __init__(self):
        super().__init__()
        self.sparql_manager = GraphRagSPARQLManager()
        self.conf_files = {
            "countries": ("conf/faiss/graphrag_index_ehri_countries.faiss", "conf/faiss/graphrag_chunks_countries.json"),
            "institutions": ("conf/faiss/graphrag_index_ehri_institutions.faiss", "conf/faiss/graphrag_chunks_institutions.json"),
            "archival_descriptions": ("conf/faiss/graphrag_index_ehri_archival_descriptions.faiss", "conf/faiss/graphrag_chunks_archival_descriptions.json")
        }
        files_statuses = [os.path.isfile(a) and os.path.isfile(b) for a, b in self.conf_files.values()]
        if all(files_statuses):
            self.loaded_files = dict([(k, self.retrieve_faiss_index(k)) for k in self.conf_files.keys()])

    def encode_chunks_and_persist_to_index(self, chunks, index, chunks_list):
        chunks_list.extend([chunk.__dict__ for chunk in chunks])
        chunks_to_embed = [self.to_entity_representation(chunk) for chunk in chunks]
        embeddings = self.model.encode(chunks_to_embed, show_progress_bar=True)
        index.add(np.array(embeddings))

    def retrieve_contents(self, index, chunks_list, type):
        self.encode_chunks_and_persist_to_index(self.sparql_manager.load_entities_chunks(type), index, chunks_list)

    def retrieve_relevant_chunks(self, query, type, top_k=6):
        query_vec = self.model.encode([query])
        index, chunks = self.loaded_files[type]
        distances, indices = index.search(query_vec, top_k)
        logger.debug(f"Distances for GraphRag retrieval of type {type}: {distances}")
        return [self.to_entity(chunks[indices[0][i]]) for i in range(len(indices[0])) if distances[0][i] < 1]

    def retrieve_faiss_index(self, type):
        faiss_index_file, chunks_file = self.conf_files[type]
        index = faiss.read_index(faiss_index_file)
        with open(chunks_file, "r") as f:
            chunks = json.load(f)
        return index, chunks

    def create_faiss_index(self):
        for k, v in self.conf_files.items():
            faiss_file, chunks_file = v
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write("")
            chunks_list = []
            index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
            self.retrieve_contents(index, chunks_list, k)
            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump(chunks_list, f)

            faiss.write_index(index, faiss_file)

    def to_entity(self, entity_dict):
        return Entity(entity_dict["id"], entity_dict["type"], entity_dict["name"], entity_dict["description"])

    def to_entity_representation(self, entity):
        return f"""Name: {entity.name}
Type: {entity.type}
URI: {entity.id}
Description: {entity.description}"""
    
if __name__ == "__main__":
    manager = RagEmbeddingsManager()
    manager.create_faiss_index()
    manager = GraphRagEmbeddingsManager()
    manager.create_faiss_index()