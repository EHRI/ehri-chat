from ehri_chat.rag.sparql_manager import GraphRagSPARQLManager
from ehri_chat.embeddings.embeddings_manager import GraphRagEmbeddingsManager
import logging
logger = logging.getLogger("ehri_chat")

class GraphRagContextManager:
    def __init__(self):
        self.sparql_manager = GraphRagSPARQLManager()
        self.embeddings_manager = GraphRagEmbeddingsManager()

    def retrieve_relevant_context(self, user_prompt):
        for type, top_k in [("countries", 2), ("institutions", 6), ("archival_descriptions", 10)]:
            relevant_entities = self.embeddings_manager.retrieve_relevant_chunks(user_prompt, type, top_k=top_k)
            logger.debug("Retrieved entities for GraphRag:")
            for entity in relevant_entities:
                logger.debug(f"{entity.type} {entity.id}")
                match entity.type:
                    case "http://lod.ehri-project-test.eu/ontology#Country":
                        yield self.retrieve_relevant_context_country(entity)
                    case "http://lod.ehri-project-test.eu/ontology#Institution":
                        yield self.retrieve_relevant_context_institution(entity)
                    case "http://lod.ehri-project-test.eu/ontology#RecordSet":
                        yield self.retrieve_relevant_context_archival_description(entity)
                    case _:
                        yield ""
    
    def retrieve_relevant_context_country(self, entity):
        return self.sparql_manager.load_relevant_data_country(entity.id)

    def retrieve_relevant_context_institution(self, entity):
        return self.sparql_manager.load_relevant_data_institution(entity.id)

    def retrieve_relevant_context_archival_description(self, entity):
        return self.sparql_manager.load_relevant_data_archival_description(entity.id)