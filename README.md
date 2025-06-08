# Jobs alert

Create your own job alert system.

## API endpoint
- Base url: `https://careerscan.io`
- Endpoints:
  - `GET /api/newest`: Returns maximum 1000 newest job postings.
    - Parameters:
      - `since`: Optional. A timestamp in milliseconds. If provided, only returns job postings created after this timestamp. If not provided or omitted, returns 1000 newest job postings.
    - Example:
      ```
      GET /api/newest?since=1700000000000
      ```
    - Response: Returns the following JSON object:
      ```typescript
      {
        "jobs": Array<JobPosting>,
        "total": number,
        "nextSince": number | null
      }
      ```
      Check out https://schema.org/JobPosting for the `JobPosting` type.

    - Rate limit: 10 requests per minute per IP address.

## Script examples

### 1. `keywords-alert.py`: 

A script that sends you a notification when a new job is posted that matches keywords you are interested in.

- Make sure to install the required packages:
```bash
pip install -r requirements.txt
```

- This script use PUSHOVER (https://pushover.net/) to send you notifications to your phone.
  - Go to https://pushover.net/ and create an account.
  - Get your user key
  - Create an application and get your API token
- How to run:
  - You can run the script with the following command, replacing the placeholders with your actual values:
```bash
python keywords-alert.py --pushover_user_key=xxx --pushover_api_token=xxx --notification_interval=3600 --query='your query'
```
  - Parameters:
    - `--pushover_user_key`: Your Pushover user key.
    - `--pushover_api_token`: Your Pushover API token.
    - `--notification_interval`: The interval in seconds to send notifications for matched job postings. Default is 3600 seconds (1 hour).
    - `--query`: The query to search for in job postings. It can be complex query, for example:
      ```
      --query='(description:"Data scientist" OR description:"Data science") AND (description:"Python" OR description:"Pandas" OR description:"SQL")'
      ```
      This will search for job postings that contain "Data scientist" or "Data science" in the description and also contain "Python", "Pandas", or "SQL".
      ```
      --query='(title:"Data scientist" OR title:"Data science") AND (description:"Python" OR description:"Pandas" OR description:"SQL") AND (hiringOrganization.name:"Google" OR hiringOrganization.name:"Meta" OR hiringOrganization.name:"Microsoft") AND employmentType:"FULL_TIME"'
      ```
      This will search for job postings that contain "Data scientist" or "Data science" in the title, "Python", "Pandas", or "SQL" in the description, and are posted by Google, Meta, or Microsoft, and are full-time jobs.
  - The query syntax is similar to the one used in ElasticSearch, you can find more information about it here: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html.
    - Note: The script use https://github.com/edeca/querydict to parse the query. The query does not support wildcards, fuzzy searches, range searches, or proximity searches. It only supports basic queries with AND, OR, and NOT operators.
    - Check out https://schema.org/JobPosting for the fields that can be used in the query.
- Install Pushover app on your phone to receive notifications.

### 2. `resume-alert.py`:
A script that sends you a notification when a new job is posted that matches your resume.
This script combine the `keywords-alert.py` script with matching your resume with the job postings before sending you a notification. Follow the same steps as above to install the required packages and run the script. You also need to install Ollama to run the LLM model for generating the query from your resume. Follow the instructions on https://ollama.com to install Ollama.

- Additional parameters:
  - `--resume_path`: Path to your resume file (PDF). The script will extract the text from the resume and use it to match job postings.
  - `--device`: Default is `cpu`. If you have a GPU, you can set it to `cuda` to use GPU for faster processing. If you use MacOS, you can set it to `mps` to use the Apple Silicon GPU.
  - `--embedding_model`: The model to use for embedding your resume and job postings. Default is `BAAI/bge-m3`. You can use any model from HuggingFace that supports sentence embeddings.
  - `--llm_model`: The model to use for generating the query from your resume. Default is `qwen2.5:7b`.
  - `--min_score`: The minimum score to consider a job posting as a match. Default is `70`. The score is calculated based on the cosine similarity between the embedding of your resume and the embedding of the job posting, rescaled from 0 to 100.
- Example:
```bash
python resume-alert.py --pushover_user_key=xxx --pushover_api_token=xxx --notification_interval=3600 --resume_path='path/to/your/resume.pdf' --device='cuda' --embedding_model='BAAI/bge-m3' --llm_model='qwen2.5:7b' --min_score=70
```