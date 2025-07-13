import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError(" Hugging Face token not found. Set it in .env or environment variables.")


MODEL_REPO_ID = "google/gemma-2b-it"

def load_llm():
   
    base_llm = HuggingFaceEndpoint(
        repo_id=MODEL_REPO_ID,
        temperature=0.5,
        max_new_tokens=512, 
        huggingfacehub_api_token=HF_TOKEN
    )
   
    return ChatHuggingFace(llm=base_llm) 


system_template = """Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know.
Only answer based on the context provided.

Context:
{context}"""

human_template = "{question}"

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template),
])

DB_PATH = "vectorstore/db_chroma"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Combine everything into RAG pipeline
llm = load_llm()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": chat_prompt},
    return_source_documents=True
)

user_query = input("Write query here:")
response = qa_chain.invoke({"query":user_query})
print("Result:" , response["result"])
print("SOURCE DOCUMENTS:",response["source_documents"])
