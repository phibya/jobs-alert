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
      This will search for job postings that contain "Data scientist" or "Data science" in the title, "Python", "Pandas", or "SQL" in the description, and do not contain "intern" or "internship" in the description, and are posted by Google, Meta, or Microsoft, and are full-time jobs.
  - The query syntax is similar to the one used in ElasticSearch, you can find more information about it here: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html.
    - Note: The script use https://github.com/edeca/querydict to parse the query. The query does not support wildcards, fuzzy searches, range searches, or proximity searches. It only supports basic queries with AND, OR, and NOT operators.
    - Check out https://schema.org/JobPosting for the fields that can be used in the query.

### Coming soon: Use LLM to match your resume with job postings and send you an email or push notification when a new job is posted that matches your resume.