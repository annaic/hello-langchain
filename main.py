import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from operator import itemgetter


load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")



model = ChatOpenAI(openai_api_key=OPEN_API_KEY, model="gpt-3.5-turbo")
str_parser = StrOutputParser()


template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know"
Context: {context}

Question: {question}
"""
qa_prompt = ChatPromptTemplate.from_template(template)
# qa_prompt.format(context="Mary's sister is Susana", question="Who is Mary's sister?")
qa_chain = qa_prompt| model | str_parser
# message = qa_chain.invoke({
#     "context": "Mary's sister is Susana",
#     "question": "Who is Mary's sister?"
# })
translation_prompt = ChatPromptTemplate.from_template(
    "Translate {answer} to {language}"
)

translation_chain = (
    {"answer": qa_chain, "language": itemgetter("language")} | translation_prompt | model | str_parser
)

message = translation_chain.invoke(
    {
        "context": "Mary's sister is Susana. She doesn't have any more siblings.",
        "question": "How many sisters does Mary have?",
        "language": "Tamil",
    }
)
print(message)

loader = TextLoader("transcription.txt")
text_documents = loader.load()


# print(text_documents)

# with open("transcription.txt", "rt") as file:
#     transcription = file.read()
#
# try:
#     qa_chain.invoke({
#         "context": transcription,
#         "question": "whoami"
#     })
# except Exception as e:
#     print(e)
