## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
The goal is to build a chatbot that can accurately extract and provide answers based on the text from a PDF document, allowing users to interact and retrieve specific information from the document without manually reading it.
### DESIGN STEPS:

#### STEP 1:
Before starting the implementation, ensure that all necessary libraries and dependencies are installed. This includes LangChain for processing the text, PyPDF2 (or similar) for reading PDF files, and an LLM like OpenAI for question-answering functionality.Install Necessary Libraries
#### STEP 2:
Use libraries like PyPDF2 to extract the text from the provided PDF document. The PDF extraction process should handle multiple pages and ensure that the text is clean and usable for further processing.
#### STEP 3:
Once the PDF text is extracted, it needs to be processed using LangChain’s tools, such as the TextSplitter and QuestionAnsweringChain, to handle large documents and provide accurate answers based on the content.
#### STEP 4:
Allow the user to input questions and receive responses based on the content extracted from the PDF document. The user will interact with the chatbot by entering questions, and the bot will provide answers based on the document’s content.
### PROGRAM:
```
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()

    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # define embedding
    embeddings = OpenAIEmbeddings()

    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/03257d90-141f-4ea1-b5b2-a5dff0279e92)
![image](https://github.com/user-attachments/assets/c821be8e-73e6-4cae-8ddd-5ba4d11d9bf7)
![image](https://github.com/user-attachments/assets/bed3790c-6795-475c-b51b-782111e9445c)


### RESULT:
The chatbot successfully extracts content from the provided PDF document and answers user queries based on the text. The results can vary depending on the complexity and clarity of the document, but the chatbot aims to provide accurate and relevant answers. The system can be further enhanced with more advanced features like document summarization or handling more complex question-answering scenarios.
