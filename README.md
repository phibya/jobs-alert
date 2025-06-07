# Jobs alert

Create your own job alert system.

## API endpoint
- Base url: `https://careerscan.io`
- Endpoints:
  - `GET /api/newest`: Returns maximum 1000 newest job postings.
    - Parameters:
      - `since`: Optional. A timestamp in milliseconds. If provided, only returns job postings created after this timestamp. If not provided or omitted, returns 1000 newest job postings.
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

