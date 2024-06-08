import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
# from langchain.chains.conversational_retrieval im

load_dotenv()
embeddings = OpenAIEmbeddings()
#
prompt="""What is LangChain"""
llm = ChatOpenAI(temperature=0)


# chain = PromptTemplate.from_template(template=prompt) | llm
# result = chain.invoke(input={})
# print(result.content)


vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'],embedding=embeddings)
retrival_qa_chat = hub.pull("varies/retrieval-qa-chat")
docs_chain = create_stuff_documents_chain(llm,retrival_qa_chat)

retrival_chain= create_retrieval_chain(retriever=vectorstore.as_retriever(),combine_docs_chain=docs_chain)
result = retrival_chain.invoke(input={"input":prompt})
print(result)