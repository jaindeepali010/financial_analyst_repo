from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import json

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o", temperature = 0) 

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

pdf_path = "Stock_Market_Performance_2024.pdf" #any stock market related PDF

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages) 

persist_directory = "chroma_db"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise
 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} 
)

web_search = TavilySearchResults(
    max_results=5,  
    search_depth="advanced",  
    include_answer=True,  
    include_raw_content=False,  
    include_images=False  
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "NO_RELEVANT_INFORMATION_FOUND"
    
    # Check if the retrieved documents contain meaningful information
    meaningful_results = []
    for i, doc in enumerate(docs):
        # Basic check for meaningful content
        if len(doc.page_content.strip()) > 50:  # Minimum content length
            meaningful_results.append(f"Document {i+1}:\n{doc.page_content}")
    
    if not meaningful_results:
        return "NO_RELEVANT_INFORMATION_FOUND"
    
    return "\n\n".join(meaningful_results)

@tool
def web_search_tool(query: str) -> str:
    """
    This tool searches the web using Tavily for information when the document doesn't contain relevant data.
    Tavily provides high-quality, structured search results optimized for AI applications.
    """
    try:
        search_results = web_search.invoke({"query": query})
        if search_results:
            formatted_results = []
            for i, result in enumerate(search_results, 1):
                formatted_result = f"Result {i}:\n"
                if 'title' in result:
                    formatted_result += f"Title: {result['title']}\n"
                if 'content' in result:
                    formatted_result += f"Content: {result['content']}\n"
                if 'url' in result:
                    formatted_result += f"Source: {result['url']}\n"
                formatted_results.append(formatted_result)
            
            return f"Web search results for '{query}':\n\n" + "\n---\n".join(formatted_results)
        else:
            return f"No web search results found for '{query}'"
            
    except Exception as e:
        return f"Error performing web search: {str(e)}"

retriever_tools = [retriever_tool]
fallback_tools = [web_search_tool]

# Create LLM instances with different tool bindings
retriever_llm = llm.bind_tools(retriever_tools)
fallback_llm = llm.bind_tools(fallback_tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str
    reformulated_query: str
    retrieved_content: str
    relevance_score: str
    needs_fallback: bool
    retrieval_attempted: bool

def should_continue_retrieval(state: AgentState):
    """Check if the last message contains tool calls for retrieval."""
    if not state.get('retrieval_attempted', False):
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
    return False

def should_evaluate_relevance(state: AgentState):
    """Check if we need to evaluate relevance of retrieved content."""
    return (state.get('retrieval_attempted', False) and 
            state.get('retrieved_content', '') != "NO_RELEVANT_INFORMATION_FOUND" and
            state.get('retrieved_content', '') != "")

def should_use_fallback(state: AgentState):
    """Check if we need to use fallback based on relevance evaluation."""
    return state.get('needs_fallback', False)

def should_continue_fallback(state: AgentState):
    """Check if the fallback agent needs to make tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

query_reformulation_prompt = """
You are a query reformulation expert specializing in stock market and financial data queries. 
Your task is to reformulate user queries to make them more effective for information retrieval from a Stock Market Performance 2024 document.

Apply these reformulation techniques:
1. Query Expansion: Add relevant synonyms and related terms
2. Specific Financial Terms: Include proper financial terminology
3. Context Enhancement: Add context that might be relevant for stock market data
4. Multiple Perspectives: Consider different aspects of the query

For example:
- "stock performance" → "stock performance equity returns market gains losses price movements trading volume"
- "tech companies" → "technology companies tech stocks FAANG Microsoft Apple Google Amazon Tesla semiconductor companies"
- "inflation impact" → "inflation impact monetary policy interest rates consumer price index CPI federal reserve economic indicators"

Original Query: {query}

Provide a reformulated query that includes:
- Original terms
- Synonyms and related terms
- Relevant financial terminology
- Context that might help retrieve better information

Reformulated Query:"""

relevance_evaluation_prompt = """
You are a relevance evaluation expert. Your task is to determine whether the retrieved information adequately answers the user's query.

Evaluation Criteria:
1. DIRECT_ANSWER: The retrieved content directly answers the user's question
2. PARTIAL_ANSWER: The retrieved content partially answers the question but lacks some important details
3. TANGENTIALLY_RELATED: The content is related to the topic but doesn't answer the specific question
4. NOT_RELEVANT: The content is not relevant to the user's question

User Query: {query}

Retrieved Content: {content}

Analyze the relevance and respond with ONLY one of these options:
- RELEVANT (if DIRECT_ANSWER or PARTIAL_ANSWER)
- NOT_RELEVANT (if TANGENTIALLY_RELATED or NOT_RELEVANT)

Your assessment:"""

retrieval_system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to search for information in the document. You can make multiple calls if needed.

IMPORTANT: If the retriever tool returns "NO_RELEVANT_INFORMATION_FOUND", you should inform the user that the information is not available in the document and suggest that they might need to search elsewhere for this information.

Please always cite the specific parts of the documents you use in your answers.
"""

fallback_system_prompt = """
You are a fallback AI assistant with Tavily web search capabilities. You are activated when the primary document doesn't contain relevant information for the user's query.

Use the Tavily web search tool to find current and relevant information to answer the user's question.
Tavily provides high-quality, structured search results that are optimized for AI applications.

Always mention that this information comes from web search since it wasn't found or wasn't relevant in the original document.
When citing web sources, include the source URLs when available.
Provide accurate, well-sourced information and be helpful to the user.
"""

final_answer_prompt = """
You are an AI assistant providing final answers based on retrieved information from a Stock Market Performance 2024 document.

The retrieved information has been evaluated as relevant to the user's query. 
Provide a comprehensive answer based on this information.

Always cite the specific parts of the documents you use in your answers.
Be accurate and helpful in your response.
"""

# Query Reformulation Agent
def reformulate_query(state: AgentState) -> AgentState:
    """Reformulate the user query to improve retrieval effectiveness."""
    original_query = ""
    for message in reversed(state['messages']):
        if isinstance(message, HumanMessage):
            original_query = message.content
            break
    
    print(f"Original Query: {original_query}")
    reformulation_message = [
        SystemMessage(content=query_reformulation_prompt.format(query=original_query))
    ]
    
    reformulation_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)  
    reformulated_response = reformulation_llm.invoke(reformulation_message)
    reformulated_query = reformulated_response.content.strip()
    
    print(f"Reformulated Query: {reformulated_query}")
    return {
        'messages': [],
        'original_query': original_query,
        'reformulated_query': reformulated_query,
        'retrieved_content': "",
        'relevance_score': "",
        'needs_fallback': False,
        'retrieval_attempted': False
    }

def call_retrieval_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with retrieval capabilities."""
    messages = list(state['messages'])
    if state.get('reformulated_query'):
        enhanced_prompt = f"""
{retrieval_system_prompt}

Note: The user's original query was: "{state.get('original_query', '')}"
This has been reformulated for better retrieval as: "{state.get('reformulated_query', '')}"

Use the reformulated query when making tool calls, but answer based on the user's original intent.
"""
        messages = [SystemMessage(content=enhanced_prompt)] + messages
    else:
        messages = [SystemMessage(content=retrieval_system_prompt)] + messages
    
    message = retriever_llm.invoke(messages)
    return {'messages': [message]}

# Relevance Evaluation Agent
def evaluate_relevance(state: AgentState) -> AgentState:
    """Evaluate whether the retrieved content is relevant to the user's query."""
    
    original_query = state.get('original_query', '')
    retrieved_content = state.get('retrieved_content', '')
    
    print(f"Evaluating relevance for query: {original_query}")
    print(f"Retrieved content length: {len(retrieved_content)}")
    
    # Create relevance evaluation prompt
    evaluation_message = [
        SystemMessage(content=relevance_evaluation_prompt.format(
            query=original_query, 
            content=retrieved_content[:2000]  # Limit content length for evaluation
        ))
    ]
    
    # Get relevance evaluation from LLM
    evaluation_llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Deterministic evaluation
    relevance_response = evaluation_llm.invoke(evaluation_message)
    relevance_assessment = relevance_response.content.strip()
    
    print(f"Relevance Assessment: {relevance_assessment}")
    
    # Determine if fallback is needed
    needs_fallback = "NOT_RELEVANT" in relevance_assessment.upper()
    
    if needs_fallback:
        print("Content deemed not relevant. Activating fallback agent.")
    else:
        print("Content deemed relevant. Proceeding with document-based answer.")
    
    return {
        'messages': [],
        'relevance_score': relevance_assessment,
        'needs_fallback': needs_fallback
    }

# Final Answer Agent (for relevant retrieved content)
def provide_final_answer(state: AgentState) -> AgentState:
    """Provide final answer based on relevant retrieved content."""
    
    original_query = state.get('original_query', '')
    retrieved_content = state.get('retrieved_content', '')
    
    # Create final answer prompt with retrieved content
    final_message = [
        SystemMessage(content=f"""
{final_answer_prompt}

User Query: {original_query}

Retrieved Content: {retrieved_content}

Provide a comprehensive answer based on this relevant information.
"""),
        HumanMessage(content=original_query)
    ]
    
    # Get final answer from LLM
    final_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    final_response = final_llm.invoke(final_message)
    
    return {'messages': [final_response]}

# Fallback LLM Agent (for web search)
def call_fallback_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with web search capabilities."""
    messages = list(state['messages'])
    
    # Add context about why we're using fallback
    fallback_context = f"""
{fallback_system_prompt}

The user's original query was: "{state.get('original_query', '')}"
This information was either not found or not relevant in the Stock Market Performance 2024 document, so you need to search the web for current information.
Relevance evaluation result: {state.get('relevance_score', 'Content not relevant')}
"""
    
    messages = [SystemMessage(content=fallback_context)] + messages
    message = fallback_llm.invoke(messages)
    return {'messages': [message]}

# Retriever Agent
def take_retrieval_action(state: AgentState) -> AgentState:
    """Execute retrieval tool calls from the LLM's response."""
    
    tool_calls = state['messages'][-1].tool_calls
    results = []
    retrieved_content = ""
    
    for t in tool_calls:
        query_to_use = t['args'].get('query', '')
        
        # If we have a reformulated query and the tool query seems to be the original query,
        # enhance it with the reformulated version
        if state.get('reformulated_query') and query_to_use:
            enhanced_query = f"{query_to_use} {state['reformulated_query']}"
            print(f"Calling Retrieval Tool: {t['name']} with enhanced query: {enhanced_query}")
        else:
            enhanced_query = query_to_use
            print(f"Calling Retrieval Tool: {t['name']} with query: {enhanced_query}")
        
        if t['name'] == 'retriever_tool':
            result = retriever_tool.invoke(enhanced_query)
            retrieved_content = result
            print(f"Retrieval result length: {len(str(result))}")
        else:
            result = "Invalid tool call for retrieval agent."
            
        # Append the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Retrieval Tools Execution Complete.")
    
    return {
        'messages': results,
        'retrieved_content': retrieved_content,
        'retrieval_attempted': True
    }

# Fallback Agent
def take_fallback_action(state: AgentState) -> AgentState:
    """Execute web search tool calls from the fallback LLM's response."""
    
    tool_calls = state['messages'][-1].tool_calls
    results = []
    
    for t in tool_calls:
        query_to_use = t['args'].get('query', '')
        print(f"Calling Fallback Tool: {t['name']} with query: {query_to_use}")
        
        if t['name'] == 'web_search_tool':
            # Use original query for web search (more natural)
            search_query = state.get('original_query', query_to_use)
            result = web_search_tool.invoke(search_query)
            print(f"Web search result length: {len(str(result))}")
        else:
            result = "Invalid tool call for fallback agent."
            
        # Append the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Fallback Tools Execution Complete.")
    return {'messages': results}

# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("query_reformulation", reformulate_query)
graph.add_node("retrieval_llm", call_retrieval_llm)
graph.add_node("retrieval_agent", take_retrieval_action)
graph.add_node("relevance_evaluation", evaluate_relevance)
graph.add_node("final_answer", provide_final_answer)
graph.add_node("fallback_llm", call_fallback_llm)
graph.add_node("fallback_agent", take_fallback_action)

# Add edges for the main retrieval flow
graph.add_edge("query_reformulation", "retrieval_llm")
graph.add_conditional_edges(
    "retrieval_llm",
    should_continue_retrieval,
    {True: "retrieval_agent", False: END}
)

# Add relevance evaluation after retrieval
graph.add_conditional_edges(
    "retrieval_agent",
    should_evaluate_relevance,
    {True: "relevance_evaluation", False: "fallback_llm"}
)

# Add decision point after relevance evaluation
graph.add_conditional_edges(
    "relevance_evaluation",
    should_use_fallback,
    {True: "fallback_llm", False: "final_answer"}
)

# Final answer leads to END
graph.add_edge("final_answer", END)

# Add fallback flow
graph.add_conditional_edges(
    "fallback_llm",
    should_continue_fallback,
    {True: "fallback_agent", False: END}
)
graph.add_edge("fallback_agent", "fallback_llm")

# Set entry point
graph.set_entry_point("query_reformulation")

rag_agent = graph.compile()

def running_agent():
    # print("\n=== ENHANCED RAG AGENT WITH QUERY REFORMULATION, RELEVANCE EVALUATION, AND TAVILY SEARCH ===")
    # print("This agent:")
    # print("1. Reformulates queries for better retrieval")
    # print("2. Searches the Stock Market Performance 2024 document")
    # print("3. Evaluates if retrieved content is relevant to the query")
    # print("4. Uses Tavily web search only if content is not relevant or not found")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({
            "messages": messages,
            "original_query": "",
            "reformulated_query": "",
            "retrieved_content": "",
            "relevance_score": "",
            "needs_fallback": False,
            "retrieval_attempted": False
        })
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

running_agent()