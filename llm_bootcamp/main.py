import ast
from openai import OpenAI
import pandas as pd
import tiktoken
from scipy import spatial

client = OpenAI(
    api_key="mabeleda",
    base_url="http://openai-api-proxy.discovery:8888/v1",
)

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


def main():
    query = "Which athletes won the gold medal in curling at the 2022 Winter Olympics?"

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You answer questions about the 2022 Winter Olympics.",
            },
            {"role": "user", "content": query},
        ],
        model=GPT_MODEL,
        temperature=0,
    )

    print(completion.choices[0].message)


if __name__ == "__main__":
    main()
