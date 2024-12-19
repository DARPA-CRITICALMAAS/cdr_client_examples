import httpx
import requests
import json
import urllib


class API:
    def __init__(self,minmod_username,minmod_password):
        self.username=minmod_username
        self.password=minmod_password
        self.endpoint='https://dev.minmod.isi.edu/api/v1'
        self.cookies=None
    
    def create_site(self,site_record):
        endpoint = f"{self.endpoint}/mineral-sites"
        params=site_record
        response = httpx.post(endpoint,json=params,cookies=self.cookies, timeout=None)

        if response.status_code != 200:
            raise Exception(f"ERROR: get status code {response.status_code}. Reason: {response.text}")
        else:
            print("Status Code 200: Posted to API")
        # print(response.json())
        response.raise_for_status()
        return response.json()
    
    def query_site(self,cdr_id):
        endpoint = f"{self.endpoint}/mineral-sites/make-id"
        params={'source_id':"mining-report::https://api.cdr.land/v1/docs/documents",'record_id':cdr_id}
        response = requests.get(endpoint,params=params,cookies=self.cookies)
        # print(response.json())
        # response.raise_for_status()
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
    
    def _get_mineral_site(self, record_id):
        # Base URL of the API
        api_url = "https://dev.minmod.isi.edu/api/v1/mineral-sites/make-id"
        
        # Query parameters
        params = {
            "source_id": "mining-report::https://api.cdr.land/v1/docs/documents",
            "record_id": record_id.strip()
        }
        
        # Headers
        headers = {
            "accept": "application/json"
        }
        
        try:
            # Send a GET request
            response = requests.get(api_url, params=params, headers=headers)
            
            # Raise an error if the request was unsuccessful
            response.raise_for_status()
            
            # Return the response JSON
            return response.json()
        except requests.exceptions.RequestException as e:
            # Handle errors
            return {"error": str(e)}


    def _get_mineral_site_details(self, site_id):
        # Base URL for the second API
        encoded_site_id = urllib.parse.quote(site_id, safe="")
        api_url = f"https://dev.minmod.isi.edu/api/v1/mineral-sites/{encoded_site_id}?format=json"
        
        # Headers
        headers = {
            "accept": "application/json"
        }
        
        try:
            # Send a GET request
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()  # Raise an error if the request was unsuccessful
            return True, response.json()
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

    def check_existence(self, record_id):
        response = self._get_mineral_site(record_id)
        boolean, message = self._get_mineral_site_details(response)

        return boolean, message