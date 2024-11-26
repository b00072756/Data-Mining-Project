

import openai
import pandas as pd
import os
import asyncio
from aiohttp import ClientSession
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
freelancer_data = pd.read_csv('Dataset/encoded_data.csv')



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

# Function to classify a batch of job descriptions
async def classify_batch(batch, session):
    prompt_template = (
        f"You are a classification model that identifies the best fit categories for a given job description "
        f"based on three category types: Core Competencies, Core Skills, and Technology/Tools. "
        f"Here are the possible values for each category:\n"
        f"1. Core Competencies: {', '.join(CATEGORY_1_COMPETENCIES)}.\n"
        f"2. Core Skills: {', '.join(CATEGORY_2_SKILLS)}.\n"
        f"3. Technology/Tools: {', '.join(CATEGORY_3_TECHNOLOGY)}.\n"
        f"Choose exactly one value for Core Competencies and Core Skills from the lists provided for each category, "
        f"and return one or more comma-separated values for Technology/Tools if needed from the list provided. "
        f"Do not include the job description in the response. Only return the values in this format:\n"
        f"Core Competencies: [value]\nCore Skills: [value]\nTechnology/Tools: [value(s)]"
    )

    batch_content = "\n".join([f"{i + 1}. {desc}" for i, desc in enumerate(batch)])
    prompt = f"{prompt_template}\nJob Descriptions:\n{batch_content}"

    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "system", "content": prompt}]
            },
            headers={"Authorization": f"Bearer {openai.api_key}"}
        ) as response:
            response_data = await response.json()
            if "choices" in response_data:
                return response_data["choices"][0]["message"]["content"]
            else:
                print(f"Error in response: {response_data}")
                return None
    except Exception as e:
        print(f"Error with batch: {batch} - {e}")
        return None

# Function to process data in batches
async def process_data_concurrently(data, batch_size=10, concurrency_limit=5):
    async with ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def process_batch(batch):
            async with semaphore:
                return await classify_batch(batch, session)

        # Split the data into batches
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # Process all batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        # Return the combined results
        return results

# Parse the API response into structured data
def parse_response(response):
    try:
        lines = response.split("\n")
        core_competencies = next((line.split(":")[1].strip() for line in lines if "Core Competencies:" in line), None)
        core_skills = next((line.split(":")[1].strip() for line in lines if "Core Skills:" in line), None)
        technology_tools = next((line.split(":")[1].strip() for line in lines if "Technology/Tools:" in line), None)
        return core_competencies, core_skills, technology_tools
    except Exception as e:
        print(f"Error parsing response: {response}, Error: {e}")
        return None, None, None

# Main function
def main():
    # Limit data to first 50 rows
    job_descriptions = freelancer_data["Description"].dropna().tolist()

    # Reset index to avoid mismatches
    freelancer_data.reset_index(drop=True, inplace=True)

    # Add columns for categories
    freelancer_data["Core Competencies"] = None
    freelancer_data["Core Skills"] = None
    freelancer_data["Technology/Tools"] = None

    # Run the async processing
    results = asyncio.run(process_data_concurrently(job_descriptions, batch_size=10, concurrency_limit=5))

    # Flatten the results and parse them back into the DataFrame
    if results:
        for idx, response in enumerate(results):
            if response:
                core_competencies, core_skills, technology_tools = parse_response(response)
                freelancer_data.loc[idx, "Core Competencies"] = core_competencies
                freelancer_data.loc[idx, "Core Skills"] = core_skills
                freelancer_data.loc[idx, "Technology/Tools"] = technology_tools

    # Save the updated dataset
    freelancer_data.to_csv("Dataset/Updated_Freelancer_Dataset.csv", index=False)
    print("Dataset updated and saved!")

if __name__ == "__main__":
    main()
