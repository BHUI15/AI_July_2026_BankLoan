"""
Bank loan application advice Agent Solution
This module implements an AI-driven loan applcation advice agent using Pydantic AI, providing
structured advice basd on application data. 
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

import json
from pathlib import Path

credentials_path = Path(__file__).parent / "gcp_api_credentials.json"
api_key = json.loads(credentials_path.read_text())["api_key"]

# 1. Defin the Schema
class ActionPlan(BaseModel):
    """
    Structured response schema for the loan application advice.
    """

    loan_summary: str = Field(description="A brief summary of the loan application.")

    action: list[str] = Field(description="Three recommended actions for the loan application.", min_length=3, max_length=3)
    reason: str = Field(..., description="The reason behind the recommended action.")
    next_steps: str = Field(..., description="Suggested next steps for the applicant.")


# 2. Create the Agent
provider = GoogleProvider(api_key=api_key)
advice_agent = Agent [None, ActionPlan](
    model = GoogleModel("gemini-3.1-flash-lite", provider=provider),
    #model="google: gemmini-3.1-flash-lite", 
    output_type=ActionPlan,
    system_prompt=("You are a financial advisor specialized in bank loan applications." 
    " Provide structured advice based on the applicant's data."
    " Keep advice practical and empathetic."),
)

# 3. Logic for the Applicaton
def get_loan_advice(application_rejected: bool) -> ActionPlan | None:
    """
    Provides a loan application advice if the loan application is failed.

    Args:
        application_rejected (bool): Indicates if the applicant's loan was rejected.

    Returns:
        A structured ActionPlan if the loan application is failed, otherwise None.
    """
    if not application_rejected:
        return None

    try:
        prompt = (
            "The applicant's loan application has been rejected."
            " Please provide a structured advice on how to improve the chances of approval."
        )
        # Generate advice using the agent
        advice = advice_agent.run_sync(prompt)
        return advice.output

    except Exception as e:
        print(f"Error generating loan advice: {e}")

        #Fallback for demo if no API key
        print("LLM call error - manual fallback")
        fallback_advice = ActionPlan(
            loan_summary="The loan application was rejected.",
            action=["Review credit report", "Increase down payment", "Provide additional documentation"],
            reason="The application did not meet the bank's criteria.",
            next_steps="Consider reapplying with improved financial information."
        )
        return fallback_advice

if __name__ == "__main__":
    # Example usage
    advice = get_loan_advice(application_rejected=True)
    if advice:
        print(advice.model_dump_json(indent=4))
    else:
        print("No advice needed as the application was approved.")