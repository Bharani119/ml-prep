## Questions
- what are the steps that we take to deploy Azure OpenAI
- How to Access the endpoints without keeping the key in the script of the REST API
- Which database is usually used to store the documents for the RAG application - AI search
  - And where is the database is located (is it Azure database)
  - How does the REST API have access to the database
- How are the texts stored (I know it's stored as vectors of chunks that are created from the text)
- What is the flow of control that happens in a RAG application
  - from user input
  - to relevant text extraction
  - to passing the top text to the Azure OpenAI endpoint
  - The API returns json, which we parse and store the relevant output to the database history - cosmos db
- How does an update to the application looks like
  - We push the updated REST API with docker to the repo
  - It is merged
  - a trigger is used to create a new docker instance in a vm 


Post hook
A new docker image is build when a new branch is merged
used to reload app services


- What are embeddings used for
  - to create vectors from text
- can models work only with vectors created with their own embeddings?

------------------
### RAG evaluation

- context relevancy
- groundedness
- answer relevancy
- similarity score - human answer vs gen ans