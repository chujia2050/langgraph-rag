# Corrective RAG

Corrective-RAG (CRAG) is a strategy for Retrieval-Augmented Generation (RAG) 
that incorporates self-reflection and self-grading on retrieved documents. 
This approach aims to enhance the relevance and accuracy of generated responses 
by evaluating and refining the information used during the generation process.

## Key Concepts

In the paper [here](https://arxiv.org/pdf/2401.15884), several steps are outlined for implementing 
Corrective-RAG:

1. **Relevance Threshold**:
   - If at least one document exceeds the threshold for relevance, the process 
   proceeds to generation.
   - Before generation, it performs knowledge refinement by partitioning the 
   document into "knowledge strips."
   - Each strip is graded, and irrelevant ones are filtered out.

2. **Supplemental Retrieval**:
   - If all documents fall below the relevance threshold or if the grader is 
   unsure, the framework seeks an additional data source.
   - Web search is used to supplement retrieval, ensuring that the most 
   relevant information is available for generation.

## Implementation with LangGraph

We will implement some of these ideas from scratch using [LangGraph](https://langchain-ai.github.io/langgraph/), focusing 
on the following aspects:

- **Skipping Knowledge Refinement**:
  - As a first pass, we will skip the knowledge refinement phase. This can be 
  added back as a node if desired.

- **Supplemental Retrieval with Web Search**:
  - If all documents are irrelevant, we will opt to supplement retrieval with 
  web search.
  - [Tavily Search](https://python.langchain.com/v0.2/docs/integrations/tools/tavily_search/) will be used for web search, providing a robust mechanism 
  for finding additional relevant information.

- **Query Re-Writing**:
  - We will use query re-writing to optimize the query for web search, 
  enhancing the quality and relevance of the retrieved documents.

## Future Enhancements

- **Knowledge Refinement**:
  - Implementing the knowledge refinement phase as a separate node in the 
  LangGraph workflow can further enhance the accuracy of the generated responses.

- **Dynamic Thresholds**:
  - Exploring dynamic thresholds for relevance can provide more flexibility 
  and adaptability in different contexts.

By leveraging these strategies, Corrective-RAG aims to improve the quality of 
RAG systems, making them more reliable and effective in generating accurate 
and contextually relevant responses.

<img src="img/CRAG.png" alt="Diagram of Corrective RAG" width="600"/>


# Agentic RAG

[Retrieval Agents](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#agents) are powerful tools designed to make informed decisions about 
whether to retrieve data from an index. By leveraging the capabilities of a 
Language Model (LLM) and a retriever tool, these agents can efficiently access 
and utilize relevant information.

## Implementation

To implement a Retrieval Agent, we simply need to provide an LLM with access 
to a retriever tool. This integration allows the LLM to query the retriever 
and make decisions based on the retrieved data.

### Steps to Implement a Retrieval Agent

1. **Set Up the LLM**:
   - Choose a suitable Language Model that can process and understand the 
   queries and data.

2. **Integrate the Retriever Tool**:
   - Connect the LLM to a retriever tool that can access the desired index or 
   database.

3. **Decision-Making Logic**:
   - Implement logic within the LLM to decide when and what to retrieve based 
   on the context and requirements.

4. **Incorporate into LangGraph**:
   - Use LangGraph to structure and manage the workflow of the Retrieval Agent, 
   ensuring seamless integration and operation.

## Incorporating into LangGraph

LangGraph provides a flexible framework to incorporate Retrieval Agents into 
your workflows. By using LangGraph, you can:

- **Define Nodes and Edges**: Structure the retrieval process with clear nodes 
and decision-making edges.
- **Manage State**: Keep track of the state and context throughout the 
retrieval process.
- **Optimize Workflows**: Streamline the retrieval and decision-making process 
for efficiency and accuracy.


## Future Enhancements

- **Advanced Decision-Making**: Implement more sophisticated decision-making 
algorithms to enhance retrieval accuracy.
- **Dynamic Indexing**: Explore dynamic indexing techniques to improve retrieval 
speed and relevance.
- **Integration with Other Tools**: Expand the capabilities of Retrieval Agents 
by integrating with additional tools and data sources.

By implementing Retrieval Agents within LangGraph, you can optimize your data 
retrieval processes, making them more efficient, accurate, and scalable.
<img src="img/Agentic_RAG.png" alt="Diagram of Agentic RAG" width="600"/>