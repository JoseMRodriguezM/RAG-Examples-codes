# Prueba de una RAG hechos con ollama, para ejecutar los modelos localmente.

from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def ollama_chat(question):
    llm = Ollama(model='llama3')

    # Load, chunk and index the contents of the pdf.
    file_path = "Your PDF path"
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # ------ Chunk text ------ #
    text_splitter = CharacterTextSplitter(chunk_size=2047, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    # Retrieve and generate using the relevant snippets.
    retriever = vectorstore.as_retriever()

    # ------ Create PromptTemplate and RAGChain ------ #
    template = """
        ### [INST] Instrucción:
        Eres un profesor univesitario, responda en español la pregunta según sus conocimientos del siguiente pdf. 
        Aquí hay contexto para ayudar:


        {context}

        ### PREGUNTA:
        {question} (responde en castellano) [/INST]

        """
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


if __name__ == "__main__":
    pregunta = input("Mensaje: ")
    respuesta = ollama_chat(pregunta)
    print(respuesta)
