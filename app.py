### Google Docs API (Fetching Text)

import os.path, json
from typing import List
from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough



SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
]

### Google Docs File IDs:
INITIAL_FILEID = "1xWRgZ4c6BhBV97WniRY5vIWTyGlQSMljjXggKt3jfIY"
TECHNICAL_SEO_FILEID = "1HGt1K9AbFz1GwY6jzQiVGmqZwgP6zHPwDotGD1d8bHU"
CONTENT_WRITING_FILEID = "1IdSXZwKeMo4su80s3sn4iSEQ18pvXDBsFW_uobj4zBQ"
CONTENT_MARKETING_FILEID = "11QfPGe2XY57RL764FoeN68lI1LvjOdwOM5pDaxyC3O8"
LOCAL_SEO_FILEID = "1uc4qH5roh6_xzv5x4osZG7nrPHpPUqRz9qPn31Y1azY"

creds = None


### Google Drive Authentication with Service Account(Google Cloud)
def auth_google_docs():
    service_account_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))
    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES
    )
    return creds


### Previous Logic (Local Development)
# def auth_google_docs():
#     global creds
#     if DOCS_TOKEN:
#         try:
#             token_json = json.loads(DOCS_TOKEN)
#             creds = Credentials.from_authorized_user_file(token_json, SCOPES)
#         except Exception:
#             creds = None

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             if DOCS_CREDENTIALS:
#                 client_config_json = json.loads(DOCS_CREDENTIALS)
#                 flow = InstalledAppFlow.run_local_server(
#                     client_config_json, SCOPES, port=0
#                 )

#                 creds = flow.run_console()

#     return creds


def get_docs_text() -> List[Document]:
    creds = auth_google_docs()
    ### Build Google Docs Service
    service = build("docs", "v1", credentials=creds)

    doc_ids = [
        INITIAL_FILEID, TECHNICAL_SEO_FILEID, CONTENT_WRITING_FILEID,
        CONTENT_MARKETING_FILEID, LOCAL_SEO_FILEID
    ]

    docs = []

    for doc_id in doc_ids:
        try:
            doc = service.documents().get(documentId=doc_id).execute()
            title = doc["title"]
            elements = doc["body"]["content"]

            text = ""
            for elem in elements:
                if "paragraph" in elem:
                    for run in elem["paragraph"]["elements"]:
                        if "textRun" in run:
                            text += run["textRun"]["content"]

            ### Pydantic Document for each doc
            docs.append(
                Document(
                    page_content=text,
                    metadata={"title": title, "id": doc_id}
                )
            )

        except Exception as e:
            print(f"{doc_id} not found: {e}")

    return docs


### Langchain LCEL (RAG) Implementation

OPENAI_MODEL_ID = "gpt-5-mini"
db_path = "vector_db"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_docs() -> List[Document]:
    docs: List[Document] = get_docs_text()
    return docs


# print("Number of Docs: ", len(load_docs()))


def split_text():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(load_docs())


### Sanity check:
# chunks = split_text()
# print(f"Total Chunks: {len(chunks)}")

# for c in chunks[:10]:
#     print(f"Meta: {c.metadata} | LENGTH: {len(c.page_content)}")


def vectorize_text(batch_size: int = 100):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    all_docs: List[Document] = split_text()

    vector_store = None
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=db_path
            )
        else:
            vector_store.add_documents(batch)
    return vector_store


# print("Embedded Text: ", vectorize_text())


def get_prompt() -> ChatPromptTemplate:
    system_template = """You are a top-tier SEO strategist helping with small business SEO. 
    You are best recommended to use the provided context(retrieved documents) for your responses 
    unless you need extra resources outside for more comprehensive or insightful advice.
    You should be concise and professional.

    Make sure to format headings and sub headings in bold and approppriate heading element for each in your markdown responses:
    example: heading and sub headings in **bold** and H2/H3/H4 in an appropriate format.

    Below is the context: 
    <context>
    {context}
    <context>
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ('human', "{question}")
    ])

    return prompt


def chain_rag_elements(vector_store, prompt):
    llm = ChatOpenAI(model=OPENAI_MODEL_ID, streaming=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    parser = StrOutputParser()

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | parser
    )

    return rag_chain

### RAG Workflow
def build_rag_workflow(embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)):
    ### Instantiate vector store that vectorizes text
    ### from Google Docs(as the external resource for RAG vector store)
    if os.path.exists(db_path) and os.listdir(path=db_path):
        print("Loading existing vectorstore...")
        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    else:
        vector_db = vectorize_text()
    # print(vector_store._collection.count())

    ### Chain all necessary elements into the RAG chain
    template = get_prompt()
    rag_chain = chain_rag_elements(vector_db, template)

    return rag_chain


### UI (Gradio)
import gradio as gr

chain = build_rag_workflow()


def chat(query, history):
    response = ""
    for chunk in chain.stream(query):
        response += chunk
        yield response


demo = gr.ChatInterface(
    fn=chat, type="messages", title="SEO Expert Bot",
    description="SEO specialist chatbot, built in a RAG system using 1000 pages volume of SEO hack documents as an external data resource(vector db)"
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)