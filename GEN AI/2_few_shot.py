import os
from google import genai

def generate():
    """
    Few-shot prompting: A few examples to guide the model's response pattern
    """
    client = genai.Client(
        api_key="AIzaSyAntjHqAvZB3cFjArl4B2lh_7pm39jUk1I"  
    )
    
    input_question = "what is LLM and can you give me 10 point for my project work"
    
    prompt = f"""You are an IT support team member. Here are a few examples of how to structure email responses:

Example:
Question: What is cloud computing?
Response:
From: it@edukron.com
To: Dear User
Subject: Understanding Cloud Computing

Dear User,

Cloud computing refers to the delivery of computing services—including servers, storage, databases, networking, software, analytics, and intelligence—over the Internet ("the cloud") to offer faster innovation, flexible resources, and economies of scale.

Key benefits include:
- Cost efficiency
- Scalability
- Accessibility
- Automatic updates

If you have more questions, feel free to reach out.

Best regards,
IT Support Team
it@edukron.com

---

Now respond to this question following the same format:
Question: {input_question}

Remember:
- From: it@edukron.com
- To: Dear User
- Provide step-by-step information where applicable
"""
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    #  SAVE OUTPUT TO FILE
    file_name = "few_shot_email.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(response.text)

    print("=" * 50)
    print("FEW-SHOT RESPONSE SAVED")
    print("=" * 50)
    print(f"📁 File saved as: {file_name}")

if __name__ == "__main__":
    generate()
