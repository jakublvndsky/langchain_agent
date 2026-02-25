import os
import requests
from typing import Literal
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


class LLMError(Exception):
    pass


@tool(
    description=(
        "Bardzo prosty kalkulator przyjmujacy dwie liczby i wykonujacy na nich akcje zgodnie z podanym znakiem"
    ),
    parse_docstring=True,
)
def liczenie(
    a: int | float, b: int | float, znak: Literal["+", "-", "*", "/"]
) -> float:
    """Pozwala obliczyć dwie liczby dodawanie lub odejmowanie

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
def sprawdz_kurs_waluty(kod_waluty: str) -> str:
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
        Obecnie masz bardzo prostego toola podłączonego, który potrafi liczyć proste rzeczy dodawanie, odejmowanie, mnożenie i dzielenie."""
)
human_msg = HumanMessage(
    "Sprawdź kurs waluty dolara amerykańskiego, a następnie oblicz ile by go było za 100 złotych"
)


agent = create_agent(
    model=provider, tools=[liczenie, sprawdz_kurs_waluty], system_prompt=system_msg
)

for response in agent.stream({"messages": [human_msg]}, stream_mode="values"):
    response["messages"][-1].pretty_print()
    # print(response.content, end="")
