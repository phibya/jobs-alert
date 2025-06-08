# %%
# Import necessary libraries
import time
import os
import argparse
from api.JobPosting import NewestJobPostingAPI
from api.PushOver import PushOverAPI
from querydict.parser import QueryEngine

# %%
# Set up variables

# get args from argparse
parser = argparse.ArgumentParser("python keywords-alert.py")
parser.add_argument("--pushover_user_key", type=str, help="Pushover user key")
parser.add_argument("--pushover_api_token", type=str, help="Pushover API token")
parser.add_argument("--notification_interval", type=int, default=3600, help="Interval in seconds to send email notifications")
parser.add_argument("--query", type=str, help="Keywords to match in job postings")

args = parser.parse_args()

# Update PUSHOVER_CONFIG with command line arguments
PUSHOVER_CONFIG = {
    "user_key": args.pushover_user_key,
    "api_token": args.pushover_api_token
}

EMAIL_INTERVAL = args.email_interval  # Interval in seconds to send email notifications
# Keywords to match in job postings

QUERY = args.query

# Ensure the query is not empty
if not QUERY:
    raise ValueError("Query cannot be empty. Please provide keywords to match in job postings.")

qe = QueryEngine(QUERY)

# %%
print("Your query: {}\n".format(QUERY))

# Modify this function if you want to change the matching logic
def match_func(job_posting: dict) -> bool:
    return qe.match(job_posting)

# %%
def main():

    # Initialize the API client
    api = NewestJobPostingAPI()
    pushover_api = PushOverAPI(
        api_token=PUSHOVER_CONFIG["api_token"],
        user_key=PUSHOVER_CONFIG["user_key"]
    )

    # Initialize variables for email notification
    last_email_sent_time = time.time()
    jobs_to_notify = []

    # Loop to continuously check for new job postings
    while True:

        # Fetch the latest job postings
        job_postings = api.get()
        # try:
        #     job_postings = api.get()
        # except Exception as e:
        #     print(f"Error fetching job postings: {e}")
        #     job_postings = []
        #     time.sleep(10)

        # Filter job postings based on the match function
        matched_jobs = [job for job in job_postings if match_func(job)]
        print(f"Found {len(matched_jobs)} matched job postings.")

        for job in matched_jobs:
            print(f"Matched Job: {job.get("title")} at {job.get("hiringOrganization").get("name")}")
            jobs_to_notify.append(job)

        # Check if enough time has passed to send an email
        current_time = time.time()
        if len(jobs_to_notify) > 0 and (current_time - last_email_sent_time) >= EMAIL_INTERVAL:
            # Send email notification
            print(f"Sending email notification for {len(jobs_to_notify)} jobs.")

            message = f"""
            {len(jobs_to_notify)} new job postings found. Here are the details:
            
            {
                "\n\n".join(
                    f"Title: {job.get("title")}\n"
                    f"Company: {job.get("hiringOrganization").get("name")}\n"
                    f"Link: {job.get("linkedInUrl")}\n" # linkedInUrl is a specific field return by the API
                    for job in jobs_to_notify
                )
            }
            """

            # Send the notification using PushOver
            response = pushover_api.send_message(
                message=message,
                title="New Job Postings Alert",
                priority=1
            )
            print(f"PushOver response: {response}")

            # Log the response from PushOver
            if response.get("status") != 1:
                print(f"Failed to send notification: {response.get('errors', 'Unknown error')}")
            else:
                # Update the last email sent time
                print(f"Email sent at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")

                # Reset the last email sent time
                last_email_sent_time = current_time

                # Clear the jobs to notify list after sending the email
                jobs_to_notify.clear()

        # Wait for a while before checking again
        time.sleep(10)

if __name__ == "__main__":
    main()
