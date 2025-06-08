from typing import Optional, List
import requests

class NewestJobPostingAPI:
    base_url: str = "https://careerscan.io/api"

    def __init__(self):
        self.since: int = 0

    def get(self) -> 'List[JobPosting]':
        req = requests.get(
            f"{NewestJobPostingAPI.base_url}/newest?since={self.since}"
        )

        json = req.json()
        if not json or "jobs" not in json:
            return []
        self.since = json.get("nextSince", 0)
        return json.get("jobs")
