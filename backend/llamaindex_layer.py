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
from openai.types.chat.chat_completion_chunk import ChoiceDelta, Choice, ChatCompletionChunk
from llama_index.core.chat_engine import CondenseQuestionChatEngine

load_dotenv()

import os

azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_api_version = os.environ["AZURE_OPENAI_VERSION"]
search_service_endpoint = os.environ["AI_SEARCH_ENDPOINT"]
search_service_api_key = os.environ["AI_SEARCH_KEY"]
AZURE_OPENAI_TEMPERATURE = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", 0))
AZURE_OPENAI_TOP_P = float(os.environ.get("AZURE_OPENAI_TOP_P", 1.0))

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
            vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID
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

        #print(len(context_nodes), current_size)

        return retrieved_nodes


class CustomVectorIndexRetriever2(BaseRetriever):

    def __init__(
            self,
            index_names,
            llm,
            filter_threshold=0.015,
            reranker=False,
            reranker_top_n=10,
            vector_top_k=20
    ) -> None:
        """Init params."""

        self.search_indexes = [self.get_search_index(index_name) for index_name in index_names]
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
        all_retrieved_nodes = []
        for search_index in self.search_indexes:
            retriever = VectorIndexRetriever(
                index=search_index,
                similarity_top_k=self.vector_top_k,
                vector_store_query_mode=VectorStoreQueryMode.HYBRID
            )
            retrieved_nodes = retriever.retrieve(query_bundle)
            all_retrieved_nodes.extend(retrieved_nodes)


        retrieved_nodes = [node for node in all_retrieved_nodes if node.score > self.filter_threshold]

        retrieved_nodes = sorted(retrieved_nodes, key=lambda x: x.score, reverse=True)

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

        #print(len(context_nodes), current_size)

        return context_nodes


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


async def get_answer_directly_from_openai(query, indexes):
    retriever = CustomVectorIndexRetriever2(indexes, Settings.llm)
    nodes = retriever.retrieve(query)
    context = "\n\n".join([f"doc{index + 1}:" + '\n' + node.get_content() for index, node in enumerate(nodes)])
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
            {"role": "system",
             "content": "Whenever the answer is numerical, make sure that it is exactly as it is in source documents.   If a table contains information about two dates, extract information for each date and then answer the question.     If the user requests data for a specific date, Check if exact date match is found, if not, use exact month or exact year. To give clarity for your answer, please provide your chain of thought, clearly explaining step by step how you arrived at the answer."},
            {"role": "system",
             "content": "Always cite the sources using the doc number inside square brackets. For example [doc1], [doc2]"},
            {"role": "assistant", "content": "What is the context?"},
            {"role": "user", "content": f"The context is {context} "},
            {"role": "assistant", "content": "What is your query?"},
            {"role": "user", "content": query},
        ],
        stream=True,
        top_p=AZURE_OPENAI_TOP_P,
        temperature=AZURE_OPENAI_TEMPERATURE
    )

    citationsChunk = get_citations(nodes)
    #responseText = ""
    # for k in response:
    #     try:
    #         responseText += k.choices[0].delta.content
    #     except:
    #         pass
    # print(responseText)
    return response, citationsChunk


def get_citations(nodes):

    citation_choices = []

    for k in nodes:
        x = {
            'content': k.text,
            'title': k.metadata['filename'],
            'url': k.metadata['url'],
            'filepath': k.id_,
            'chunk_id': '0',
            'metadata': k.metadata
        }
        citation_choices.append(x)
    import json
    context = {
        'messages': [
            {
                'role': 'tool',
                'end_turn': False,
                'content': json.dumps({'citations': citation_choices, 'strategy': "STRATEGY1"})
            }
        ]
    }
    c = Choice(delta=ChoiceDelta(content=None,
                                 function_call=None,
                                 role='assistant',
                                 tool_calls=None,
                                 context=context),
               finish_reason=None,
               index=0,
               logprobs=None
               )
    completionChunk = ChatCompletionChunk(id='chatcmpl-8ZB9m2Ubv8FJs3CIb84WvYwqZCHST',
                                          choices=[c],
                                          created=1703395058,
                                          model='gpt-3.5-turbo-0613',
                                          object='chat.completion.chunk',
                                          system_fingerprint=None)

    return completionChunk


custom_prompt = PromptTemplate(
    """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)


def get_final_question_based_on_history(history, latest_message):
    if len(history) == 0:
        return latest_message
    string_messages = []
    for k in history:
        if k['role'] in ['user', 'assistant']:
            string_message = f"{k['role']}: {k['content']}"
            string_messages.append(string_message)
    history_str = "\n".join(string_messages)

    custom_prompt = f"""\
    Given a conversation (between Human and Assistant) and a follow up message from Human, \
    rewrite the message to be a standalone question that captures all relevant context \
    from the conversation.

    <Chat History>
    {history_str}

    <Follow Up Message>
    {latest_message}

    <Standalone question>
    """

    import openai
    llm = openai.AzureOpenAI(
        api_key=azure_openai_key,
        azure_endpoint=azure_openai_endpoint,
        api_version=aoai_api_version,
    )
    response = llm.chat.completions.create(
        model=os.environ["AZURE_LLM_MODEL_DEPLOYMENT_NAME"],  # model = "deployment_name".
        messages=[

            {"role": "user", "content": f"{custom_prompt} "},
        ],
    )

    final_query = response.choices[0].message.content
    #print(f"FINAL QUERY : {final_query}")
    return final_query


if __name__ == "__main__":
    import asyncio

    asyncio.run(get_answer_directly_from_openai("Who is the tenant of Green Square - North Tower, Brisbane QLD as of December 2019 and give details about the tenancy?", ["ispt-air-dev-ir-llamaindex-3"]))
