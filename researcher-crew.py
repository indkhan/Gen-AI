import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You are an expert at a technology research group, 
  skilled in identifying trends and analyzing complex data.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a content strategist known for 
  making complex tech topics interesting and easy to understand.""",
  verbose=True,
  allow_delegation=True,
)

# Create tasks for your agents
task1 = Task(
  description="""Analyze 2024's AI advancements. 
  Find major trends, new technologies, and their effects. 
  Provide a detailed report.""",
  agent=researcher
)

task2 = Task(
  description="""Create a blog post about major AI advancements using your insights. 
  Make it interesting, clear, and suited for tech enthusiasts. 
  It should be at least 4 paragraphs long.""",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)