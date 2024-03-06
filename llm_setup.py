from llama_index.callbacks import CallbackManager
from llama_index import ServiceContext
from llama_index.llms import OpenAI
import os
from llama_index import StorageContext, VectorStoreIndex, download_loader, load_index_from_storage
from pathlib import Path
import pandas as pd
from llama_index import Document

# system_prompt = """You are an expert in due diligence processes, specializing in financial, technical, and market analysis. 
# Your role is to act as a financial and legal advisor and provide insightful and accurate information and advise related to user queries. 
# Do not ask to consult anyone else and give your views according to the data supplied.
# Please answer questions based on your expertise in these areas. Do not hallucinate features."""

system_prompt = """
You are given a dataset of companies which has 50 companies, the dataset has sector, the Index which the companies are listed on.
The dataset also has financial numbers of the company which help us identify about the company and its performance.
You need to act like Junior Financial Analyst and give your views and recommendation to the team about accquiring a suitable company for the given company.
Answer should be from within the dataset. Answer should be kept short
"""

PERSIST_DIR = "./storage"
DATA_FILE_PATH = "MandAData.xlsx"

def read_and_get_data():
    df = pd.read_excel(DATA_FILE_PATH)

    # Get column names
    column_names = df.columns.tolist()
    
    # Initialize an empty list to store documents
    documents = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Convert each row into a document format
        id = f"{row['index']}_{row['sector']}_{index}"  # Example: "12345_Tech_0"
        metadata = {
            "id": id,
            "sector": row["sector"],  # Example: Extracting author metadata from the Excel file
            "index": row["index"],  # Example: Extracting category metadata from the Excel file
            "Company name" : row["Company Name"]# Add more metadata fields as needed
        }
         
        document_text = " ".join(f"{column_name}: {value}" for column_name, value in zip(column_names, row))
        
        # Create a Document object with ID and text
        document = Document(doc_id=str(id), text=document_text, metadata= metadata)
        
        # Append the document to the list of documents
        documents.append(document)
   
    return documents

def get_vector_index(service_context):
    if not os.path.exists(PERSIST_DIR):
        documents = read_and_get_data()
        print(documents)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:  # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, service_context=service_context)
    return index

def get_service_context(token_counter):
    callback_manager = CallbackManager([token_counter])

    llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", system_prompt=system_prompt, max_tokens=256)
    service_context = ServiceContext.from_defaults(llm=llm, callback_manager=callback_manager)

    return service_context

def get_query_engine(service_context):
    index = get_vector_index(service_context)
    return index.as_query_engine(vector_store_query_mode="svm", similarity_top_k=10)
