import httpx
import requests
import json

class API:
    def __init__(self,minmod_username,minmod_password):
        self.username=minmod_username
        self.password=minmod_password
        self.endpoint='https://dev.minmod.isi.edu/api/v1'
        self.cookies=None
    
    def create_site(self,site_record):
        endpoint = f"{self.endpoint}/mineral-sites"
        params=site_record
        response = httpx.post(endpoint,json=params,cookies=self.cookies)
        # print(response.json())
        response.raise_for_status()
        return response.json()
    
    def query_site(self,cdr_id):
        endpoint = f"{self.endpoint}/mineral-sites/make-id"
        params={'source_id':"mining-report::https://api.cdr.land/v1/docs/documents",'record_id':cdr_id}
        response = requests.get(endpoint,params=params,cookies=self.cookies)
        if response.status_code != 200:
            raise Exception(f"ERROR: get status code {response.status_code}. Reason: {response.text}")
        else:
            print("Status Code 200: Posted to API")
        # print(response.json())
        response.raise_for_status()
        return response.json()
    
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