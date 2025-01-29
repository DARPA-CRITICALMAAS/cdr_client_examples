import httpx
import requests
import json
import urllib
from datetime import datetime


class API:    
    def __init__(self,minmod_username,minmod_password, limit):
        self.username=minmod_username
        self.password=minmod_password
        self.endpoint='https://dev.minmod.isi.edu/api/v1'
        self.cookies=None
        self.limit = limit
        self.current_date = datetime.now().date()
        self.count = 0
        
        
    def increment(self):
        today = datetime.now().date()
        print(f"Checking the increment: current date: {today} count: {self.count} limit: {self.limit}")
        if today != self.current_date:
            self.current_date = today
            self.count = 0

        if self.count < self.limit:
            self.count += 1
            return self.count, True
        else:
            return self.count, False
    
    
    def login(self):
        endpoint = f"{self.endpoint}/login"
        params={'username':self.username,'password':self.password}
        response = httpx.post(endpoint,json=params)
        response.raise_for_status()
        self.cookies=response.cookies
        return response.json()
    
    def whoami(self):
        endpoint = f"{self.endpoint}/whoami"
        response = httpx.get(endpoint,cookies=self.cookies)
        response.raise_for_status()
        return response.json()
   
  