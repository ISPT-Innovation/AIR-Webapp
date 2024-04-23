from typing import List
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from llama_index.core import QueryBundle, StorageContext, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine, Accumulate
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.azureaisearch import IndexManagement, AzureAISearchVectorStore, MetadataIndexFieldType
from llama_index_client import MessageRole, ChatMessage

load_dotenv()

import os

azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_api_version = os.environ["AZURE_OPENAI_VERSION"]
search_service_endpoint = os.environ["AI_SEARCH_ENDPOINT"]
search_service_api_key = os.environ["AI_SEARCH_KEY"]

llm = AzureOpenAI(
    model=os.environ["LLM_MODEL_NAME"],
    deployment_name=os.environ["AZURE_LLM_MODEL_DEPLOYMENT_NAME"],
    api_key=azure_openai_key,
    azure_endpoint=azure_openai_endpoint,
    api_version=aoai_api_version,
)

embed_model = AzureOpenAIEmbedding(
    model=os.environ["EMBEDDING_MODEL_NAME"],
    deployment_name=os.environ["AZURE_EMBEDDING_MODEL_DEPLOYMENT_NAME"],
    api_key=azure_openai_key,
    azure_endpoint=azure_openai_endpoint,
    api_version=aoai_api_version,
)
Settings.embed_model = embed_model
Settings.llm = llm


class CustomVectorIndexRetriever(BaseRetriever):

    def __init__(
            self,
            index_name,
            llm,
            filter_threshold=0.015,
            reranker=False,
            reranker_top_n=5,
            vector_top_k=10
    ) -> None:
        """Init params."""

        self.search_index = self.get_search_index(index_name)
        self.filter_threshold = filter_threshold
        self.reranker = reranker
        self.llm = llm
        self.vector_top_k = vector_top_k
        self.filter_threshold = filter_threshold
        self.reranker_top_n = reranker_top_n

        super().__init__()

    def get_vector_store(self, search_index_name):
        credential = AzureKeyCredential(search_service_api_key)
        index_client = SearchIndexClient(
            endpoint=search_service_endpoint,
            credential=credential,
        )

        metadata_fields = {
            "filename": "filename",
            "level": ("level", MetadataIndexFieldType.INT32),
            "parent_id": ("parent_id", MetadataIndexFieldType.STRING),
            "url": ("url", MetadataIndexFieldType.STRING)
        }
        vector_store = AzureAISearchVectorStore(
            search_or_index_client=index_client,
            filterable_metadata_field_keys=metadata_fields,
            index_name=search_index_name,
            index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
            id_field_key="id",
            chunk_field_key="chunk",
            embedding_field_key="embedding",
            embedding_dimensionality=1536,
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn",
        )
        return vector_store

    def get_search_index(self, SEARCH_INDEX_NAME):
        vector_store = self.get_vector_store(SEARCH_INDEX_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        search_index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context
        )
        return search_index

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        retriever = VectorIndexRetriever(
            index=self.search_index,
            similarity_top_k=self.vector_top_k,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID
        )

        retrieved_nodes = retriever.retrieve(query_bundle)

        retrieved_nodes = [node for node in retrieved_nodes if node.score > self.filter_threshold]

        if self.reranker:
            reranker = RankGPTRerank(
                llm=self.llm,
                top_n=self.reranker_top_n,
                verbose=True,
            )
            retrieved_nodes = reranker.postprocess_nodes(
                retrieved_nodes, query_bundle
            )

        context_nodes = []
        current_size = 0
        for node in retrieved_nodes:
            if current_size + node.metadata['size'] > 10000:
                break
            context_nodes.append(node)
            current_size += node.metadata['size']

        print(len(context_nodes), current_size)

        return retrieved_nodes


def get_query_engine(retriever, response_synthesizer):
    custom_query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    return custom_query_engine


def get_chat_engine(custom_query_engine, custom_chat_history):
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=custom_query_engine,
        chat_history=custom_chat_history,
        verbose=True,
    )
    return chat_engine


async def get_answer_directly_from_openai(query):
    retriever = CustomVectorIndexRetriever("ispt-air-dev-esg-llamaindex-3", Settings.llm)
    nodes = retriever.retrieve(query)
    context = "\n\n".join([node.get_content() for node in nodes])
    #print(context)
    import openai
    llm = openai.AzureOpenAI(
        api_key=azure_openai_key,
        azure_endpoint=azure_openai_endpoint,
        api_version=aoai_api_version,
    )

    response = llm.chat.completions.create(
        model=os.environ["AZURE_LLM_MODEL_DEPLOYMENT_NAME"],  # model = "deployment_name".
        messages=[
            {"role": "system",
             "content": "You are a representative of ISPT, a property fund company. You need to respond to questions with the context provided as a first person."},
            {"role": "system", "content": "What is the context?"},
            {"role": "user", "content": f"The context is {context} "},
            {"role": "system", "content": "What is your query?"},
            {"role": "system", "content": query},
        ],
        stream=True
    )
    return response


if __name__ == "__main__":
    import asyncio

    asyncio.run(get_answer_directly_from_openai("What is the NLA for IRAPT in 2022?"))
