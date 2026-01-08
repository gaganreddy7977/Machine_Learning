import os
from google import genai

def generate():
    client = genai.Client(
        api_key="AIzaSyAntjHqAvZB3cFjArl4B2lh_7pm39jUk1I"  
    )

    input_question = "I am getting an error while I login to my account"

    prompt = '''This response is from IT team, draft like email and make it step by
    step instruction for a specific issue raised by the user.
    From should be always it@edukron.com
    To should be Dear User
    '''

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=input_question + prompt
    )

    # SAVE OUTPUT TO FILE
    with open("app_support_email.txt", "w", encoding="utf-8") as file:
        file.write(response.text)

    print(" app_support_email.txt")

if __name__ == "__main__":
    generate()
