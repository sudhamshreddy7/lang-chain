import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    # print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))

    query = "What is LangChain?"
    query_template = "Give me short 3 to 4 lines answer for the question: {query}"
    query_prompt = PromptTemplate.from_template(query_template)
    # query_prompt.format(query=query)
    llm = ChatOpenAI(
        model="gpt-4.1-nano", temperature=0, max_retries=3, timeout=60, max_tokens=500
    )
    chain = query_prompt | llm
    ## output of left component is passed as input to right component.
    ## In out case query_prompt's output is passed as input to llm
    ## This returns a runnable chain object
    response = chain.invoke({"query": query})
    print("Response:", response.content)


if __name__ == "__main__":
    main()
