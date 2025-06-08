import requests

class PushOverAPI:
    def __init__(self, api_token: str, user_key: str):
        self.api_token = api_token
        self.user_key = user_key

    def send_message(self, message: str, title: str = None, priority: int = 0):

        url = "https://api.pushover.net/1/messages.json"
        payload = {
            "token": self.api_token,
            "user": self.user_key,
            "message": message,
            "title": title,
            "priority": priority
        }

        response = requests.post(url, data=payload)
        return response.json()