import os
import requests
import nest_asyncio
import asyncio
import bs4
from typing import Literal
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


load_dotenv()

nest_asyncio.apply()


async def load_mcp():
    mcp_client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@theo.foobar/mcp-time"],
            }
        },
    )

    mcp_tools = await mcp_client.get_tools()
    print(f"Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")
    return mcp_tools


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]


class LLMError(Exception):
    pass


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pc = Pinecone(PINECONE_API_KEY)

indexes = pc.list_indexes()

try:
    index = pc.Index("langchain-agent")
    print("Znalazłem indeks")
except Exception as e:
    print(f"Nie udało się połączyć do indeksu w bazie wektorowej: {e}")
    print("==== Tworzę nowy indeks w bazie wektorowej ====")
    index = pc.create_index(
        name="langchain-agent",
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        metric="cosine",
    )
    print("Utworzyłem indeks")


vector_store = PineconeVectorStore(index=index, embedding=embeddings)

bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)


all_splits = text_splitter.split_documents(docs)

document_ids = vector_store.add_documents(documents=all_splits)


@tool
def retrive_context(query: str):
    """Przetwarza informacje z bazy wektorowej w celu udzielenia dokładniejszej odpowiedzi"""
    retrived_docs = vector_store.similarity_search(query=query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrived_docs
    )
    return serialized, retrived_docs


@tool(
    description=(
        "Bardzo prosty kalkulator przyjmujacy dwie liczby i wykonujacy na nich akcje zgodnie z podanym znakiem"
    ),
    parse_docstring=True,
)
def liczenie(
    a: int | float, b: int | float, znak: Literal["+", "-", "*", "/"]
) -> float:
    """Pozwala obliczyć dwie liczby

    Args:
        a: pierwsza liczba w działaniu
        b: druga liczba w działaniu
        znak: jest to argument, który przyjmuje:
            "-" czyli dodawanie a - b
            "+" czyli odejmowanie a + b
            "*" czyli mnożenie a * b
            "/" czyli dzielenie a / b
    """
    print("Zaczynam liczenie 🧮")
    if znak == "+":
        return a + b
    elif znak == "-":
        return a - b
    elif znak == "*":
        return a * b
    elif znak == "/":
        if b == 0:
            raise ZeroDivisionError("Nie można dzielić przez zero")
        return a / b
    else:
        raise ValueError("Nie właściwy znak został podany")


@tool(
    "kantor",
    description=("Sluzy do wyszukiwania kursu waluty po jej kodzie zgodnie z ISO 4217"),
    parse_docstring=True,
)
def sprawdz_kurs_waluty(kod_waluty: str) -> dict:
    """Sprawdza kursy walut z polskiej na waluty obce

    Args:
        kod_waluty: jest to kod waluty, który jest zgodny z oznaczeniem ISO 4217
    """

    r = requests.get(
        f"http://api.nbp.pl/api/exchangerates/rates/A/{kod_waluty}/?format=json"
    )
    if r.status_code == 200:
        response = r.json()
        return response
    else:
        raise Exception(f"Jest błędny status usługi, a dokładnie {r.status_code}")


provider = ChatOpenAI(
    model="gpt-5-mini", temperature=0.4, max_retries=2, api_key=OPENAI_API_KEY
)

ollama_provider = ChatOllama(model="llama3.2:3b", temperature=0.4)

system_msg = SystemMessage(
    """Jesteś moim osobistym pomocnikiem o imieniu Orion, a ja mam na imię Kuba. 
        Obecnie masz bardzo prostego toola podłączonego, który potrafi liczyć proste rzeczy dodawanie, odejmowanie, mnożenie i dzielenie.
        Drugi tool to wyszukiwanie uśrednionego kursu walut wedle kursu NBP - nalezy podawać kod waluty zgodny z oznaczeniem ISO 4217
        Trzeci tool to dostęp do bazy wektorowej, która ma za zadanie pogłębić Twoją wiedzę na zadawane pytanie odnośnie tematyki Autonomicznych Agentów zasilanych LLM-ami - używaj go aby udzielać lepszych odpowiedzi użytkownikowi
        Czwarty tool to serwer mcp, który jest od sprawdzania czasu
    """
)
human_msg = HumanMessage(
    "Sprawdź kurs waluty dolara amerykańskiego, a następnie oblicz ile by go było za 100 złotych oraz sprawdź godzinę w Nowym Jorku"
)

config = {"configurable": {"thread_id": "1"}}
checkpointer = InMemorySaver()


async def build_agent():
    mcp_tools = await load_mcp()
    agent = create_agent(
        model=provider,
        tools=[liczenie, sprawdz_kurs_waluty, retrive_context, *mcp_tools],
        system_prompt=system_msg,
        checkpointer=checkpointer,
    )

    return agent


steps = []


async def chat():
    agent = await build_agent()
    while True:
        try:
            text = await asyncio.to_thread(input, "Cześć, w czym Ci dzisiaj pomóc?\n")
        except (EOFError, KeyboardInterrupt):
            break

        text = text.strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            break

        prompt = HumanMessage(text)
        async for response in agent.astream(
            {"messages": [prompt]}, config=config, stream_mode="values"
        ):
            response["messages"][-1].pretty_print()
            steps.append(response)


if __name__ == "__main__":
    asyncio.run(chat())
