import openai
import pandas as pd
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Set your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

openai.log = "debug"


# Define categories
CATEGORY_1_COMPETENCIES = [
    "3d", "3d animation", "3d architectural rendering", "3d art", "3d avatar", "3d clothing design",
    "3d design", "3d drawing", "3d floor plan", "3d game art", "3d illustration", "3d landscape rendering",
    "3d lighting", "3d mockup", "3d model", "3d modeler", "3d modeling", "3d motion tracking",
    "3d printing", "3d product animation", "3d product rendering", "3d rendering", "3d rigging",
    "3d scanning", "3d sculpting", "3d texturing", "3d texturing & lighting", "3d vehicle modeling",
    "3d walkthrough animation", "3ds", "Adobe digital marketing suite", "Affiliate marketing",
    "Agile software development", "Ai mobile app development", "Algorithm development",
    "Android app development", "Android developer", "Api development", "App development",
    "Appsheet developer", "Ar & vr development", "Autocad civil 3d", "Autocad plant 3d",
    "B2b marketing", "B2c marketing", "Back-end development", "Behavior-driven development",
    "Big data", "Bing marketing", "Blackberry app development", "Blockchain development",
    "Blog development", "Book marketing", "Bot development", "Brand development", "Brand marketing",
    "Branding & marketing", "Bulk marketing", "Business applications development", "Business development"
]

CATEGORY_2_SKILLS = [
    "Programming Languages", "Frameworks and Libraries", "Cloud Platforms", "DevOps and CI/CD", "Databases",
    "Data Analysis and Visualization", "Artificial Intelligence", "APIs and Integrations", "E-commerce Platforms",
    "CRM and Marketing Tools", "Mobile Development", "Web Development", "Data Engineering", "Version Control",
    "Gaming", "3D Modeling and Animation", "Cybersecurity", "BI and Analytics", "Automation",
    "Social Media and Marketing", "Blockchain and Crypto", "Testing"
]

CATEGORY_3_TECHNOLOGY = [
    "Java", "Python", "C++", "C#", "JavaScript", "Ruby", "PHP", "Kotlin", "Swift", "R", "Go", "TypeScript", "SQL",
    "React", "Angular", "Vue.js", "Django", "Flask", "Spring Boot", "Node.js", "Bootstrap", "TensorFlow", "PyTorch", "Keras",
    "AWS", "Google Cloud Platform", "Azure", "Firebase", "DigitalOcean", "Heroku", "IBM Cloud", "Alibaba Cloud",
    "Docker", "Kubernetes", "Jenkins", "GitLab", "GitHub Actions", "Ansible", "Terraform", "CircleCI", "Travis CI",
    "MySQL", "PostgreSQL", "MongoDB", "SQL Server", "Oracle Database", "SQLite", "MariaDB", "Firebase Realtime Database", "BigQuery",
    "Tableau", "Power BI", "Excel", "Google Data Studio", "Looker", "MATLAB", "Alteryx", "Qlik Sense", "Splunk",
    "Machine Learning", "Deep Learning", "Natural Language Processing", "Reinforcement Learning", "Computer Vision",
    "API Development", "RESTful API", "GraphQL", "Zapier", "Postman", "Twilio", "Stripe API", "Google APIs",
    "Shopify", "WooCommerce", "Magento", "BigCommerce", "PrestaShop", "Salesforce Commerce Cloud",
    "Salesforce", "HubSpot", "Zoho CRM", "Google Analytics", "Ahrefs", "SEMrush", "Mailchimp",
    "Android", "iOS", "Flutter", "React Native", "Xamarin", "Swift", "Kotlin",
    "HTML", "CSS", "JavaScript", "PHP", "WordPress", "Wix", "Webflow", "Squarespace",
    "ETL", "Data Pipelines", "Airflow", "Glue", "Data Warehousing", "Spark", "Hadoop", "Databricks",
    "Git", "GitHub", "GitLab", "Bitbucket", "Subversion",
    "Unity", "Unreal Engine", "Game Development", "Blender", "Autodesk Maya", "3D Modeling",
    "3D Rendering", "Blender", "Autodesk Maya", "Cinema 4D", "SketchUp", "Lumion", "3ds Max",
    "Vulnerability Assessment", "Penetration Testing", "Firewall Management", "SOC", "SIEM", "Cryptography",
    "Zapier", "Integromat", "RPA", "Automation Anywhere", "UiPath", "Blue Prism",
    "Facebook", "Instagram", "LinkedIn", "TikTok", "YouTube", "Twitter", "Pinterest", "Google Ads",
    "Blockchain", "Ethereum", "Bitcoin", "Smart Contracts", "NFT", "Solidity",
    "Selenium", "JUnit", "TestNG", "Postman", "Automated Testing", "Manual Testing"
]

# Load dataset
freelancer_data = pd.read_csv('Dataset/encoded_data.csv')






# Function to classify job descriptions
def classify_job_description(job_description):
    """
    Classifies a job description into Core Competencies, Core Skills, and Technology/Tools.
    """
    print(f"Sending request to API for job description:\n{job_description}\n")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", #Replace model="gpt-3.5-turbo" with model="gpt-4o-mini".
            messages=[
                {"role": "system", "content": (
                    "You are a classification model that identifies the best fit categories for a given job "
                    "description based on three category types: Core Competencies, Core Skills, and Technology/Tools. "
                    "Here are the possible values for each category: "
                    "1. Core Competencies: " + ", ".join(CATEGORY_1_COMPETENCIES) + ". "
                    "2. Core Skills: " + ", ".join(CATEGORY_2_SKILLS) + ". "
                    "3. Technology/Tools: " + ", ".join(CATEGORY_3_TECHNOLOGY) + ". "
                    "Choose exactly one value for Core Competencies and Core Skills from the lists provided for each category only, and return one or more "
                    "comma-separated values for Technology/Tools if needed from the list provided only. "
                    "FOR ALL THREE CATEGORIES, ONLY RETURN VALUES FROM THE GIVEN LISTS. DO NOT GIVE return OUTSIDE THE LIST. FOR CORE COMPETENCIES AND CORE SKILLS ONLY RETURN A SINGLE VALUE." 
                    "FOR TECHNOLOGY IT CAN BE MORE THAN 1 DEPENDING ON THE JOB DESCRIPTION."
                    "For technologies, only add technologies from the 3. TECHNOLOGY/TOOLS list i provided."
                    "Do not added irrelevant commennts like (if needed) inside the value just return the same word exactly from the lists"
                    "Below is an example you can use for api response"
                    "API Response:"
                    "Core Competencies: BI and Analytics"
                    "Core Skills: Data Engineering"
                    "Technology/Tools: SQL, Snowflake, Looker, Python"
                )},
                {"role": "user", "content": job_description}
            ]
        )
        categories = response['choices'][0]['message']['content']
        #print(f"API Response:\n{categories}\n")
        return categories
    except openai.APIError  as e:
        print(f"Error during OpenAI API call: {e}")
        return None

# Function to filter and handle technologies
def filter_technologies(technologies):
    """
    Retains only valid technologies from the list and adds 'Other' if invalid technologies are found.
    Compares technologies in a case-insensitive manner, trims spaces, and removes duplicates.
    """
    if not technologies or not isinstance(technologies, str):
        return "Other"
    
    # Split technologies, trim spaces, and compare case-insensitively
    technologies_list = [tech.strip() for tech in technologies.split(",") if tech.strip()]
    print("Technologies List: ", technologies_list)

    # Filter valid technologies
    valid_technologies = [
        tech for tech in technologies_list if tech.lower() in [t.lower() for t in CATEGORY_3_TECHNOLOGY]
    ]
    print("Valid Technologies List (Before Removing Duplicates): ", valid_technologies)

    # Remove duplicates
    unique_valid_technologies = list(dict.fromkeys(valid_technologies))  # Maintains order while removing duplicates
    print("Valid Technologies List (After Removing Duplicates): ", unique_valid_technologies)

    if not unique_valid_technologies:
        return "Other"
    elif len(unique_valid_technologies) < len(technologies_list):
        # If some technologies are invalid, add "Other"
        unique_valid_technologies.append("Other")

    return ", ".join(unique_valid_technologies)

freelancer_data_sample = freelancer_data

# Initialize categories as empty
freelancer_data_sample["Category 1"] = None
freelancer_data_sample["Category 2"] = None
freelancer_data_sample["Category 3"] = None

start_index = 0  # Replace with the actual row number to resume from
#for index, row in freelancer_data_sample.iterrows():

for index, row in freelancer_data_sample.iloc[start_index:].iterrows():
    job_description = row["Description"]

    # Skip empty job descriptions
    if pd.isna(job_description) or job_description.strip() == "":
        print(f"Row {index} has an empty job description. Skipping...")
        continue
    print(f"Processing job description for row {index}...")
    
    # Classify the job description
    classification = classify_job_description(job_description)

    try:
        # Parse the response into dictionary-like components
        classification_parts = classification.split("\n")
        core_competencies = (
            classification_parts[0].split(":")[1].strip()
            if len(classification_parts) > 0 and ":" in classification_parts[0]
            else None
        )
        core_skills = (
            classification_parts[1].split(":")[1].strip()
            if len(classification_parts) > 1 and ":" in classification_parts[1]
            else None
        )
        technology_tools = (
            classification_parts[2].split(":")[1].strip()
            if len(classification_parts) > 2 and ":" in classification_parts[2]
            else None
        )
        print(f"Core Competencies: {core_competencies}")
        print(f"Core Skills: {core_skills}")
        print(f"Full Technology/Tools string: {technology_tools}")

        freelancer_data_sample.loc[index, "Category 1"] = core_competencies.strip() if core_competencies else None
        freelancer_data_sample.loc[index, "Category 2"] = core_skills.strip() if core_skills else None
        freelancer_data_sample.loc[index, "Category 3"] = filter_technologies(technology_tools)
        #if index % 50 == 0:  # Save every 50 rows
            #freelancer_data_sample.to_csv('Dataset/Updated_Freelancer_Dataset.csv', index=False)
           # print(f"Progress saved at row {index}.")
           # print(f"Waiting for {9} seconds to respect rate limits...")
           # time.sleep(9) #seconds

    except Exception as e:
        print(f"Error parsing classification results for row {index}: {e}")
        freelancer_data_sample.loc[index, "Category 1"] = None
        freelancer_data_sample.loc[index, "Category 2"] = None
        freelancer_data_sample.loc[index, "Category 3"] = "Other"

# Save the updated dataset
freelancer_data_sample.to_csv('Dataset/Updated_Freelancer_Dataset.csv', index=False)

print(f"Updated dataset saved as 'Dataset/Updated_Freelancer_Dataset.csv'")
