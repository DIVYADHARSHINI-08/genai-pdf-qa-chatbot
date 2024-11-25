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
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import QuestionAnsweringChain
from langchain.llms import OpenAI

# Extract PDF text
def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Initialize LLM (OpenAI, or other LLMs)
llm = OpenAI(temperature=0.7)

# Initialize TextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Create Q&A Chain
qa_chain = QuestionAnsweringChain.from_llm(llm)

def answer_question(question, chunks):
    context = " ".join(chunks)
    return qa_chain.run({"input_document": context, "question": question})

def main():
    pdf_path = "document.pdf"  # Provide the path to your PDF file
    extracted_text = extract_pdf_text(pdf_path)
    chunks = splitter.split_text(extracted_text)
    
    print("PDF-based Question Answering Chatbot")
    
    while True:
        question = input("Ask a question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        answer = answer_question(question, chunks)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/6c19b4d3-38c4-46aa-b868-7149113d7a17)

### RESULT:
The chatbot successfully extracts content from the provided PDF document and answers user queries based on the text. The results can vary depending on the complexity and clarity of the document, but the chatbot aims to provide accurate and relevant answers. The system can be further enhanced with more advanced features like document summarization or handling more complex question-answering scenarios.
