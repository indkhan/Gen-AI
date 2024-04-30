import os

from crewai import Agent, Task, Process, Crew
from langchain_groq import ChatGroq

# To Load Local models through groq

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"), model="llama3-70b-8192"
)  # mixtra1-8x7b-32768
marketer = Agent(
    role="Market Research Analyst",
    goal="Find out how big is the demand for my products and suggest how to reach the widest possible customer base",
    backstory="""You are an expert at understanding the market demand, target audience, and competition. This is crucial for 
		validating whether an idea fulfills a market need and has the potential to attract a wide audience. You are good at coming up
		with ideas on how to appeal to widest possible audience.
		""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
    #   llm=llm # to load gemini
)

technologist = Agent(
    role="Technology Expert",
    goal="Make assessment on how technologically feasable the company is and what type of technologies the company needs to adopt in order to succeed",
    backstory="""You are a visionary in the realm of technology, with a deep understanding of both current and emerging technological trends. Your 
		expertise lies not just in knowing the technology but in foreseeing how it can be leveraged to solve real-world problems and drive business innovation.
		You have a knack for identifying which technological solutions best fit different business models and needs, ensuring that companies stay ahead of 
		the curve. Your insights are crucial in aligning technology with business strategies, ensuring that the technological adoption not only enhances 
		operational efficiency but also provides a competitive edge in the market.""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
    #   llm=llm # to load gemini
)

business_consultant = Agent(
    role="Business Development Consultant",
    goal="Evaluate and advise on the business model, scalability, and potential revenue streams to ensure long-term sustainability and profitability",
    backstory="""You are a seasoned professional with expertise in shaping business strategies. Your insight is essential for turning innovative ideas 
		into viable business models. You have a keen understanding of various industries and are adept at identifying and developing potential revenue streams. 
		Your experience in scalability ensures that a business can grow without compromising its values or operational efficiency. Your advice is not just
		about immediate gains but about building a resilient and adaptable business that can thrive in a changing market.""",
    verbose=True,  # enable more detailed or extensive output
    allow_delegation=True,  # enable collaboration between agent
    #   llm=llm # to load gemini
)

task1 = Task(
    description="""Analyze what the market demand for plugs for holes in crocs (shoes) so that this iconic footware looks less like swiss cheese. 
		Write a detailed report with description of what the ideal customer might look like, and how to reach the widest possible audience. The report has to 
		be concise with at least 10 bullet points and it has to address the most important areas when it comes to marketing this type of business.
    """,
    agent=marketer,
)

task2 = Task(
    description="""Analyze how to produce plugs for crocs (shoes) so that this iconic footware looks less like swiss cheese.. Write a detailed report 
		with description of which technologies the business needs to use in order to make High Quality T shirts. The report has to be concise with 
		at least 10  bullet points and it has to address the most important areas when it comes to manufacturing this type of business. 
    """,
    agent=technologist,
)

task3 = Task(
    description="""Analyze and summarize marketing and technological report and write a detailed business plan with 
		description of how to make a sustainable and profitable "plugs for crocs (shoes) so that this iconic footware looks less like swiss cheese" business. 
		The business plan has to be concise with 
		at least 10  bullet points, 5 goals and it has to contain a time schedule for which goal should be achieved and when.
    """,
    agent=business_consultant,
)

crew = Crew(
    agents=[marketer, technologist, business_consultant],
    tasks=[task1, task2, task3],
    verbose=2,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

result = crew.kickoff()

print("######################")
print(result)
