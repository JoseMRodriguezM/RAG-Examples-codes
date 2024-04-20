import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements

from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["UNSTRUCTURED_API_KEY"] = "Your API KEY"

# ----- Indexing aritcle ------ #
path_to_pdf = "Your PDF path"
unstructured_api_key = os.environ.get("UNSTRUCTURED_API_KEY")

client = UnstructuredClient(
    api_key_auth=unstructured_api_key,
    # if using paid API, provide your unique API URL:
    # server_url=“YOUR_API_URL”,
)
with open(path_to_pdf, "rb") as f:
    files = shared.Files(
      content=f.read(),
      file_name=path_to_pdf,
      )
req = shared.PartitionParameters(
        files=files,
        chunking_strategy="by_title",
        max_characters=512,
)
try:
    resp = client.general.partition(req)
except SDKError as e:
    print(e)

elements = dict_to_elements(resp.elements)
documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    documents.append(Document(page_content=element.text, metadata=metadata))

# Load chunked documents into the FAISS index
db = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ------ Charging the model and tokenizing ------ #

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""
Promopt ejample that Meta brings or recommend.

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ model_answer_1 }}<|eot_id|>

"""
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=200,
    eos_token_id=terminators,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions, in Spanish, about IPM.
You are given the extracted parts of a long document and a question.
Provide a conversational answer.
If you don't know the answer, just say "I do not know."
Don't make up an answer.

Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
result = rag_chain.invoke("Question")
print(result)
