# About this repository. 
This repository contains differents codes examples to make a RAG. The idea of this repository is to show that there are many ways to code a RAG.
The examples are made with Langchain, Ollama, HuggingFace, Unstructured DB and FAISS vectorstore db. If you are intrersted in use some of them you can use others tecnolgies, such us other vectorsotore db like Chroma DB o use them without Unstructured, etc, feel free to modify it.

# How to use them.
+ First install the requirementes, using ``pip install -q .\requirements.txt``
+ Then just run one of them, using `` python .\explame-you-want.py``
+ If you gonna use the third example you have to insert your Unstrucured API key in the field ``os.environ["UNSTRUCTURED_API_KEY"] = "Your API KEY"``. YOu can get your api key for free in this page: https://unstructured.io/api-key-free 
