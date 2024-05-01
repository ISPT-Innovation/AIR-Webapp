from typing import List
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents._generated.models import VectorizedQuery, QueryType, QueryCaptionType, QueryAnswerType
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from llama_index.core import QueryBundle, StorageContext, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import Refine, Accumulate
from llama_index.core.schema import NodeWithScore, TextNode
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


class CustomVectorIndexRetriever3(BaseRetriever):

    def __init__(
            self,
            index_names,
            llm,
            filter_threshold=2,
            reranker=False,
            reranker_top_n=10,
            vector_top_k=10
    ) -> None:
        """Init params."""
        self.index_names = index_names
        self.filter_threshold = filter_threshold
        self.reranker = reranker
        self.llm = llm
        self.vector_top_k = vector_top_k
        self.filter_threshold = filter_threshold
        self.reranker_top_n = reranker_top_n
        self.credential = AzureKeyCredential(search_service_api_key)

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        all_retrieved_nodes = []

        for search_index_name in self.index_names:
            print(search_index_name)
            search_client = SearchClient(endpoint=search_service_endpoint, index_name=search_index_name,
                                         credential=self.credential)

            embedding = embed_model.get_text_embedding(query_bundle.query_str)
            print(embedding)
            vector_query = VectorizedQuery(vector=embedding,
                                           k_nearest_neighbors=self.vector_top_k,
                                           fields="contentVector",
                                           exhaustive=True)

            results = search_client.search(
                search_text=query_bundle.query_str,
                vector_queries=[vector_query],
                select=["metadata", "content", "filename", "url", "id"],
                query_type=QueryType.SEMANTIC, semantic_configuration_name='my-semantic-config',
                query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
                top=20
            )
            for result in results:
                # print(result['filename'])
                # print(result['id'])
                # print(result['content'])
                # print(result['metadata'])
                # print(result['@search.reranker_score'])
                import json
                node1 = TextNode(text=result['content'], id_=result['id'], metadata=json.loads(result['metadata']))
                # print(node1)
                node1_score = NodeWithScore(node=node1, score=result['@search.reranker_score'])
                all_retrieved_nodes.append(node1_score)




        retrieved_nodes = [node for node in all_retrieved_nodes if node.score > self.filter_threshold]

        # print(retrieved_nodes)
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

        print(len(context_nodes), current_size)

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
    retriever = CustomVectorIndexRetriever3(indexes, Settings.llm)
    nodes = retriever.retrieve(query)
    context = "\n\n".join([f"doc{index + 1}:" + '\n' + node.get_content() for index, node in enumerate(nodes)])
    query = f"Always cite the sources using the doc number.\n{query}"
    # print(context)
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
             "content": "Always cite the sources using the doc number. For example [doc1], [doc2]"},
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
    # responseText = ""
    # for k in response:
    #     try:
    #         responseText += k.choices[0].delta.content
    #     except:
    #         pass
    # print(responseText)
    return response, citationsChunk


def get_citations(nodes):
    citation_choices = []

    import urllib.parse


    for k in nodes:
        base_url = "https://airstoragestg2.blob.core.windows.net/ispt-air-dev-3-container/DocumentLibrary/IR"
        new_filename = k.metadata['filename']

        # Replace spaces with %20 to make the filename URL-friendly
        url_friendly_filename = urllib.parse.quote(new_filename)

        # Concatenate the base URL with the URL-friendly filename
        final_url = f"{base_url}/{url_friendly_filename}"

        x = {
            'content': k.text,
            'title': k.metadata['filename'],
            'url': final_url,
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
    # print(f"FINAL QUERY : {final_query}")
    return final_query


if __name__ == "__main__":
    import asyncio

    asyncio.run(get_answer_directly_from_openai(
        "what is the Gross assets for Core Fund in 2022",
        ["ispt-air-dev-ir-llamaindex-4"]))
