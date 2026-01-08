import os
from google import genai

def generate():
    """
    Single-shot prompting: Direct prompt without examples
    """
    client = genai.Client(
        api_key="AIzaSyAntjHqAvZB3cFjArl4B2lh_7pm39jUk1I"  
    )

    input_question = "can you give me step to step guide on data science"

    prompt = f"""You are an IT support team member. Draft a professional email response to the following user question.

User Question: {input_question}

Instructions:
- From: it@edukron.com
- To: Dear User
- Provide a clear, professional response to the user's question
- Format the email appropriately with subject line, greeting, body, and closing

Draft the email:"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    #  SAVE OUTPUT TO FILE
    file_name = "single_shot_email.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(response.text)

    print("=" * 50)
    print("SINGLE-SHOT RESPONSE SAVED")
    print("=" * 50)
    print(f"📁 File saved as: {file_name}")

if __name__ == "__main__":
    generate()
