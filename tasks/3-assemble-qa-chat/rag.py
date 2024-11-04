from embedding_db import EmbeddingDatabase
from llm import LargeLanguageModel


class QuestionAnsweringRAG:
    def __init__(self, llm: LargeLanguageModel, embedding_db: EmbeddingDatabase):
        self.llm = llm
        self.embedding_db = embedding_db

    def _create_prompt(self, context: str, message: str) -> str:
        # TODO: Add your RAG prompt
        raise NotImplementedError()


    def query(self, query: str) -> str:
        documents = self.embedding_db.retrieve(query)
        context = "\n".join(documents)
        prompt = self._create_prompt(context, query)
        
        return self.llm.call(prompt)
