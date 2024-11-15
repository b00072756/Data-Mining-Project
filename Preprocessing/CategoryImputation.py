import openai

# Set your API key
openai.api_key = "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"

# Define categories with refined values
CATEGORY_1_COMPETENCIES = [
    "Leadership", "Risk Assessment", "Project Management", "Customer Satisfaction", "Business Development",
    "Strategic Planning", "Communication", "Financial Analysis", "Cybersecurity", "Product Management",
    "Teamwork", "Marketing Strategy", "Compliance", "Sales Strategy", "Technical Documentation",
    "Consulting", "Quality Control", "Customer Support", "Operations Management", "Supply Chain Management"
]

CATEGORY_2_SKILLS = [
    "Data Analysis", "Machine Learning", "Data Mining", "Algorithm Development", "A/B Testing",
    "Technical Writing", "CRM", "Financial Modeling", "Web Development", "Regression Testing",
    "Simulation Modeling", "Predictive Analytics", "Database Querying", "UX Research", "Process Improvement",
    "Statistical Analysis", "Quantitative Research", "Natural Language Processing (NLP)", "Automation", "Risk Modeling"
]

CATEGORY_3_TECHNOLOGY = [
    "Power BI", "SQL", "Excel", "Python", "Tableau", "AWS", "Google Analytics", "Chatbot Development",
    "Django", "TensorFlow", "Blockchain", "Docker", "GitHub", "Salesforce CRM", "Adobe Creative Suite",
    "Java", "Azure", "SAP", "Kubernetes", "MATLAB"
]

def classify_job_description(job_description):
    """
    Classifies a job description into Core Competencies, Core Skills, and Technology/Tools.
    """

    response =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": (
                "You are a classification model that identifies the best fit categories for a given job "
                "description based on three category types: Core Competencies, Core Skills, and Technology/Tools. "
                "Here are the possible values for each category: "
                "1. Core Competencies: " + ", ".join(CATEGORY_1_COMPETENCIES) + ". "
                "2. Core Skills: " + ", ".join(CATEGORY_2_SKILLS) + ". "
                "3. Technology/Tools: " + ", ".join(CATEGORY_3_TECHNOLOGY) + ". "
                "Choose one value from each category that best aligns with the job description provided."
            )},
            {"role": "user", "content": job_description}
        ]
    )

    # Extract the categories from the model's response
    categories = response['choices'][0]['message']['content']
    return categories

# Example usage
job_description = """
Hi,

Would you be able to help me do a case-study this Saturday around 2PM EST? It will take up to 3 hours.

It will require Excel or Sheets, and maybe a little bit of SQL/Python, but I think Excel/Sheets will suffice.

The Case Study will be on a business problem for a tech on-demand delivery startup, and the objective of the Case Study is to to break down a complex problem and present the information in a clear, concise and structured manner. 

How we will do it: We can get on a Google Meet call, and I will share my screen with you. You will guide me live on how to complete it. 

Please let me know if you can do it, and we can talk further.

Bonus if you have previously worked at Uber/DoorDash/Instacart etc. in an Operations Manager role, but it's not necessary.
"""

# Classify the job description
categories = classify_job_description(job_description)
print("Classification Results:", categories)
