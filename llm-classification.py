from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

chat = ChatGroq(
    model="mixtral-8x7b-32768",  # llama3-70b-8192
    temperature=0,
)

msg = chat.invoke("what is 2 + 2?")
print(msg.content)

# --------------------------------------------------------------
# Customer Support Ticket Classification System
# --------------------------------------------------------------

from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from enum import Enum
from typing import List

# Sample customer support tickets
ticket1 = """
I ordered a laptop from your store last week (Order #12345), but I received a tablet instead.
This is unacceptable! I need the laptop for work urgently. Please resolve this immediately or I'll have to dispute the charge.
"""

ticket2 = """
Hello, I'm having trouble logging into my account. I've tried resetting my password, but I'm not receiving the reset email.
Can you please help me regain access to my account? I've been a loyal customer for years and have several pending orders.
"""

# --------------------------------------------------------------
# Regular Completion using OpenAI (with drawbacks)
# --------------------------------------------------------------

client = OpenAI()


def classify_ticket_simple(ticket_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Classify the following customer support ticket into a category.",
            },
            {"role": "user", "content": ticket_text},
        ],
    )
    return response.choices[0].message.content


system = "Classify the following customer support ticket into a category.And give only category name without any other text."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
chain.invoke({"text": ticket1})


result = classify_ticket_simple(ticket1)
print(result)

"""
Drawbacks of this approach:
1. No structured output, making it difficult to integrate into automated systems
2. No validation of the output, potentially leading to inconsistent categorizations
3. Limited information extracted, missing important details for prioritization
4. No confidence score, making it hard to flag uncertain classifications for human review
"""


# --------------------------------------------------------------
# Step 1: Get clear on your objectives
# --------------------------------------------------------------




"""
Objective: Develop an AI-powered ticket classification system that:
- Accurately categorizes customer support tickets
- Assesses the urgency and sentiment of each ticket
- Extracts key information for quick resolution
- Provides confidence scores to flag uncertain cases for human review
Business impact:
- Reduce average response time by routing tickets to the right department
- Improve customer satisfaction by prioritizing urgent and negative sentiment tickets
- Increase efficiency by providing agents with key information upfront
- Optimize workforce allocation by automating routine classifications
"""

# --------------------------------------------------------------
# Step 2: Patch your LLM with instructor
# --------------------------------------------------------------


# Instructor makes it easy to get structured data like JSON from LLMs
import os
import instructor

from groq import Groq
from pydantic import BaseModel

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter
client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)


# --------------------------------------------------------------
# Step 3: Define Pydantic data models
# --------------------------------------------------------------

"""
This code defines a structured data model for classifying customer support tickets using Pydantic and Python's Enum class.
It specifies categories, urgency levels, customer sentiments, and other relevant information as predefined options or constrained fields.
This structure ensures data consistency, enables automatic validation, and facilitates easy integration with AI models and other parts of a support ticket system.
"""


class TicketCategory(str, Enum):
    ORDER_ISSUE = "order_issue"
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_INQUIRY = "product_inquiry"
    TECHNICAL_SUPPORT = "technical_support"
    BILLING = "billing"
    OTHER = "other"


class CustomerSentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"


class TicketUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketClassification(BaseModel):
    category: TicketCategory
    urgency: TicketUrgency
    sentiment: CustomerSentiment
    confidence: float = Field(
        ge=0, le=1, description="Confidence score for the classification"
    )
    key_information: List[str] = Field(
        description="List of key points extracted from the ticket"
    )
    suggested_action: str = Field(
        description="Brief suggestion for handling the ticket"
    )


ticket_classification = TicketClassification(
    category=TicketCategory.ORDER_ISSUE,
    urgency=TicketUrgency.HIGH,
    sentiment=CustomerSentiment.ANGRY,
    confidence=0.9,
    key_information=["Order #12345", "Received tablet instead of laptop"],
    suggested_action="Contact customer to arrange laptop delivery",
)

# --------------------------------------------------------------
# Step 4: Bring everything together in a single function
# --------------------------------------------------------------


def classify_ticket(ticket_text: str) -> TicketClassification:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=TicketClassification,
        temperature=0,
        max_retries=3,
        messages=[
            {
                "role": "system",
                "content": "Analyze the following customer support ticket and extract the requested information.",
            },
            {"role": "user", "content": ticket_text},
        ],
    )
    return response


result1 = classify_ticket(ticket1)
result2 = classify_ticket(ticket2)

print(result1.model_dump_json(indent=2))
print(result2.model_dump_json(indent=2))


def classify_ticket(ticket_text: str) -> TicketClassification:
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        response_model=TicketClassification,
        temperature=0,
        max_retries=3,

        messages=[
            {"role": "system", "content": "Analyze the following customer support ticket and extract the requested information."},
            {"role": "user", "content": ticket_text}
        ],
    )
    return response

# --------------------------------------------------------------
# Step 5: Optimize your prompts and experiment
# --------------------------------------------------------------
# To optimize:
# 1. Refine the system message to provide more context about your business
# 2. Experiment with different models (e.g., gpt-3.5-turbo vs gpt-4)
# 3. Fine-tune the model on your specific ticket data if available
# 4. Adjust the TicketClassification model based on business needs

SYSTEM_PROMPT = """
You are an AI assistant for a large e-commerce platform's customer support team.
Your role is to analyze incoming customer support tickets and provide structured information to help our team respond quickly and effectively.
Business Context:
- We handle thousands of tickets daily across various categories (orders, accounts, products, technical issues, billing).
- Quick and accurate classification is crucial for customer satisfaction and operational efficiency.
- We prioritize based on urgency and customer sentiment.
Your tasks:
1. Categorize the ticket into the most appropriate category.
2. Assess the urgency of the issue (low, medium, high, critical).
3. Determine the customer's sentiment.
4. Extract key information that would be helpful for our support team.
5. Suggest an initial action for handling the ticket.
6. Provide a confidence score for your classification.
Remember:
- Be objective and base your analysis solely on the information provided in the ticket.
- If you're unsure about any aspect, reflect that in your confidence score.
- For 'key_information', extract specific details like order numbers, product names, or account issues.
- The 'suggested_action' should be a brief, actionable step for our support team.
Analyze the following customer support ticket and provide the requested information in the specified format.
"""


def classify_ticket(ticket_text: str) -> TicketClassification:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=TicketClassification,
        temperature=0,
        max_retries=3,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": ticket_text},
        ],
    )
    return response


result1 = classify_ticket(ticket1)
result2 = classify_ticket(ticket2)

print(result1.model_dump_json(indent=2))
print(result2.model_dump_json(indent=2))
