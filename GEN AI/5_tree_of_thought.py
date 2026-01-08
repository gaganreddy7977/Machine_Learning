import os
from google import genai

def generate():
    """
    Tree-of-Thought (ToT) prompting: Exploring multiple reasoning paths
    """
    client = genai.Client(
        api_key="AIzaSyAntjHqAvZB3cFjArl4B2lh_7pm39jUk1I"
    )
    
    input_question = "what is datascience"
    
    # Tree-of-Thought prompt: Multiple reasoning branches
    prompt = f"""You are an IT support team member. Draft an email response by exploring multiple reasoning paths.

User Question: {input_question}

Explore different approaches to answer this question:

Path 1: Technical/Definition Approach
- Define the term precisely
- Explain core concepts
- Use technical terminology
- Include technical examples

Path 2: Practical/Application Approach
- Explain through real-world applications
- Focus on how it's used
- Provide practical examples
- Show relevance to common tasks

Path 3: Beginner-Friendly Approach
- Use simple language
- Avoid jargon
- Use analogies
- Focus on understanding

Path 4: Comprehensive Approach
- Combine definition + applications + examples
- Cover multiple aspects
- Provide depth while maintaining clarity
- Include resources for further learning

Evaluation:
- Path 1 is best for: Technical users
- Path 2 is best for: Users wanting practical knowledge
- Path 3 is best for: Beginners
- Path 4 is best for: General audience (RECOMMENDED)

Selected Path: Path 4 (Comprehensive Approach)

Now, draft the email response using Path 4, incorporating:
- Clear definition
- Real-world applications
- Practical examples
- Step-by-step breakdown where applicable

Email Format:
From: it@edukron.com
To: Dear User

Draft the complete email:"""
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    file_name = "tree_shot_email.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(response.text)

    print("=" * 50)
    print("tree-OF-THOUGHT RESPONSE:")
    print("=" * 50)
    print(f"📁 File saved as: {file_name}")

if __name__ == "__main__":
    generate()
    