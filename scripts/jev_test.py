import os
os.environ["TYPESAFE_API_KEY"] = "apikey_"

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()

ticket = "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP."

response = client.system_one(
    state=ticket,
    questions={
        "department": Choice(
            instructions="Which team should handle this",
            criteria={
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        ),
        "frustration": Score(
            instructions="How frustrated the customer appears",
            criteria=[
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        ),
        "is_urgent": Noul(
            instructions="The message conveys urgency or time-sensitivity",
        ),
    },
)

print(response.answers["department"].choice)  # "technical"
print(response.answers["frustration"].score)  # 1.0
print(response.answers["is_urgent"].noul)     # 1.0

ticket = "Hi, I've been working on the credit card application model bad definition for last 3 days, risk analysis suggested using 60+dpd, however policy team incline 90+dpd"

response = client.system_one(
    state=ticket,
    questions={
        "department": Choice(
            instructions="Which team should makde this decision",
            criteria={
                "risk analysis": "Cut off analysis issue",
                "risk policy": "Policy issue on the risk definition",
                "data science": "Modeling issue",
            },
        ),
        "is_urgent": Noul(
            instructions="The message conveys urgency or time-sensitivity",
        ),
    },
)

dept = response.answers["department"]
print(dept.choice)
print(dept.confidence)          # 0–1
print(dept.probabilities)       # 各选项概率

urg = response.answers["is_urgent"]
print(urg.noul)                 # 是/否的概率
