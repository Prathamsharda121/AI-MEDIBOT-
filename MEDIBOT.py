import os
import streamlit as st
from dotenv import load_dotenv

# Import necessary components from LangChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    st.error("❌ Hugging Face token not found. Set it in your .env file or as an environment variable.")
    st.stop() 

MODEL_REPO_ID = "google/gemma-2b-it" 

@st.cache_resource 
def get_vectorstore():
    DB_PATH = "vectorstore/db_chroma"
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

@st.cache_resource 
def load_llm_and_chain():
   
    base_llm = HuggingFaceEndpoint(
        repo_id=MODEL_REPO_ID,
        temperature=0.7,
        max_new_tokens=2048,
        huggingfacehub_api_token=HF_TOKEN
    )
    
   
    chat_llm = ChatHuggingFace(llm=base_llm)

    
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

    retriever = get_vectorstore()
    if retriever is None:
        return None 

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": chat_prompt},
        return_source_documents=True
    )
    return qa_chain



def main():
    st.title("Ask Chatbot...")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
   
    qa_chain = load_llm_and_chain()
    if qa_chain is None:
        st.warning("RAG pipeline could not be initialized. Please check error messages above.")
        return 
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

   
    prompt = st.chat_input("Pass your prompt here")

    if prompt:
       
        with st.chat_message("user"):
            st.markdown(prompt)
        
      
        st.session_state.messages.append({'role': 'user', 'content': prompt})

       
        with st.spinner("Thinking..."): 
            try:
                response = qa_chain.invoke({"query": prompt})
                result = response["result"]
                source_documents = response.get("source_documents", []) 
               
                full_response_text = result
                if source_documents:
                    full_response_text += "\n\n**Sources:**\n"
                    for doc in source_documents:
                        source_info = doc.metadata.get("source", "Unknown")
                        page_content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                        full_response_text += f"- *Source:* `{source_info}`\n"
                        
                with st.chat_message("assistant"):
                    st.markdown(full_response_text)
                
                
                st.session_state.messages.append({'role': 'assistant', 'content': full_response_text})

            except Exception as e:
                st.error(f"Error processing your query: {e}")
                
                st.session_state.messages.append({'role': 'assistant', 'content': f"Sorry, an error occurred: {e}"})


if __name__ == "__main__":
    main()