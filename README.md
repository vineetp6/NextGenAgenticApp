# NextGenAgenticApp
Wikipedia Talk Network Analyzer
1. Background and Motivation
Background: I am a data scientist/engineer with expertise in graph analytics, natural language processing (NLP), and machine learning. I have experience working with graph databases like Neo4j and ArangoDB, as well as building agentic applications using frameworks like LangChain and LangGraph.

Why I Participated: I participated to explore the integration of GraphRAG, cuGraph, and ArangoDB for scalable graph analytics and natural language querying. The goal was to build an agentic application that can dynamically process natural language queries and retrieve insights from large-scale graph datasets.

2. Dataset and Business Problem
Dataset: I chose the Wikipedia Talk Network dataset (wiki-Talk.txt.gz) from the Stanford Network Analysis Project (SNAP). This dataset represents user interactions on Wikipedia talk pages, where edges indicate that one user replied to another user's comment.

Business Problem:

Problem: Understanding user interactions and influence in online communities is critical for improving engagement and moderation.

Solution: By analyzing the Wikipedia Talk Network, we can identify influential users, detect communities, and understand communication patterns. This can help Wikipedia moderators improve community engagement and resolve conflicts.

3. Data Conversion to a Graph
Data Format: The dataset is a text file where each line represents an edge in the format source target, indicating that user source replied to user target.

Conversion:

The dataset was loaded into a Pandas DataFrame.

A directed graph was created using NetworkX, where nodes represent users and edges represent replies.

The graph was then wrapped in a NetworkxEntityGraph for compatibility with LangChain's GraphQAChain.

import pandas as pd
import networkx as nx

# Load the dataset into a Pandas DataFrame
wiki_talk = pd.read_csv(
    "./wiki-Talk.txt.gz",
    compression="gzip",
    sep="\t",
    names=["source", "target"],
    skiprows=4  # Skip the header lines
)

# Create a directed graph in NetworkX
G = nx.from_pandas_edgelist(wiki_talk, "source", "target", create_using=nx.DiGraph)
4. Persisting/Loading Data into ArangoDB
Step 1: Connect to ArangoDB using the python-arango library.

Step 2: Create a graph in ArangoDB and define the edge and vertex collections.

Step 3: Load the NetworkX graph into ArangoDB in batches to handle large datasets efficiently.


from arango import ArangoClient

# Connect to ArangoDB
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("wiki_talk", username="root", password="password")

# Create a graph in ArangoDB
graph = db.create_graph("wiki_talk_graph")
users = graph.create_vertex_collection("users")
replies = graph.create_edge_definition(
    edge_collection="replies",
    from_vertex_collections=["users"],
    to_vertex_collections=["users"]
)

# Load the graph into ArangoDB in batches
batch_size = 10000
for i in range(0, len(wiki_talk), batch_size):
    batch = wiki_talk[i:i + batch_size]
    for _, row in batch.iterrows():
        users.insert({"_key": str(row["source"])})
        users.insert({"_key": str(row["target"])})
        replies.insert({"_from": f"users/{row['source']}", "_to": f"users/{row['target']}"})
5. Visualizing a Sample of the Graph
A subgraph of the Wikipedia Talk Network was visualized using NetworkX and Matplotlib.

The visualization shows a small subset of nodes and edges to demonstrate the structure of the graph.


import matplotlib.pyplot as plt

# Visualize a subgraph
subgraph = nx.subgraph(G, list(G.nodes())[:1000])
pos = nx.spring_layout(subgraph, seed=42)
plt.figure(figsize=(10, 8))
nx.draw(subgraph, pos, node_size=20, with_labels=False, width=0.5)
plt.title("Wikipedia Talk Network Subgraph")
plt.show()
6. Agentic App: Dynamic Query Processing
The agentic application uses LangChain and LangGraph to process natural language queries. Here's how it works:

Components of the Jupyter Notebook
Tools:

graph_qa_tool: Answers questions about the graph using GraphQAChain.

cugraph_analysis_tool: Performs GPU-accelerated graph analytics using cuGraph.

Agent:

The agent uses a ReAct (Reasoning and Acting) framework to dynamically select and execute tools based on the query.

It uses a custom ReAct prompt to ensure structured outputs.

Gradio Interface:

A web interface is provided using Gradio for users to interact with the agent.

Walkthrough of the Notebook
Load the Dataset:

The dataset is loaded into a NetworkX graph and persisted in ArangoDB.

Define Tools:

Tools are defined to handle specific types of queries (e.g., graph queries, analytics).

Initialize the Agent:

The agent is initialized with the tools and a custom ReAct prompt.

Launch the Interface:

The Gradio interface is launched to allow users to input queries and view results.


# Define tools
@tool
def graph_qa_tool(query: str):
    """Use this tool to answer questions about the Wikipedia Talk Network."""
    if "how many edges" in query.lower():
        return f"Observation: The graph has {G.number_of_edges()} edges."
    elif "how many nodes" in query.lower():
        return f"Observation: The graph has {G.number_of_nodes()} nodes."
    else:
        return f"Observation: {graph_qa_chain.run(query)}"

@tool
def cugraph_analysis_tool(query: str):
    """Use this tool for GPU-accelerated graph analytics using cuGraph."""
    if "pagerank" in query.lower():
        import cugraph as cg
        import cudf

        # Convert NetworkX graph to cuGraph
        edges = cudf.DataFrame(list(G.edges()), columns=["source", "target"])
        G_cugraph = cg.Graph()
        G_cugraph.from_cudf_edgelist(edges, source="source", destination="target")

        # Perform PageRank
        pagerank = cg.pagerank(G_cugraph)
        return f"Observation: PageRank results: {pagerank.to_pandas().head()}"
    else:
        return "Observation: Invalid query for cuGraph analysis."

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"prompt": react_prompt}
)

# Launch the Gradio interface
interface = gr.Interface(
    fn=query_graph,
    inputs="text",
    outputs="text",
    title="Wikipedia Talk Network Analyzer",
    description="Ask questions about the Wikipedia Talk Network or perform GPU-accelerated graph analytics."
)
interface.launch(inline=True)
7. Agent Tools
graph_qa_tool:

Answers questions about the graph (e.g., "How many edges are in the graph?").

Uses GraphQAChain to process natural language queries.

cugraph_analysis_tool:

Performs GPU-accelerated graph analytics (e.g., "Perform PageRank on the graph.").

Uses cuGraph for fast computation on large graphs.

Example Queries and Outputs
Query:


How many edges are in the graph?
Output:
The graph has 5021410 edges.



# Graph Visualization
![download](https://github.com/user-attachments/assets/ee1ee19b-dd59-4aff-a50b-e629c7e43f7f)
