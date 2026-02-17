from rag.chain import create_rag_chain

chain = create_rag_chain()

response = chain.invoke("What services do you provide?")

print(response)
