import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
load_dotenv()
# search_tool = SerperDevTool()
"""
from langchain_community.llms import Ollama

llm = Ollama(model="gemma", temperature=0.1)
"""
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"), model="llama3-70b-8192"
)  # mixtra1-8x7b-32768

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover groundbreaking insights about it",
    backstory="""You are a highly skilled researcher with a talent for delving into complex problems and extracting valuable knowledge. 
    You have a broad research background and a knack for identifying crucial trends across various disciplines.""",
    verbose=True,
    allow_delegation=False,
    tools=[search],
    llm=llm,
    max_iter=10,
)


writer = Agent(
    role="Content Strategist",
    goal="Craft compelling content on it",
    backstory="""You are a content strategist known for 
  making complex  topics interesting and easy to understand.""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    max_iter=10,
)

# Create tasks for your agents
task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
    expected_output="Full analysis report in bullet points",
    agent=researcher,
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
    expected_output="Full blog post of at least 4 paragraphs",
    agent=writer,
)
# Instantiate your crew with a sequential process
crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=2)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
