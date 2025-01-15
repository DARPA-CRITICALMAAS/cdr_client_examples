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
    
    def create_site(self,site_record):
        endpoint = f"{self.endpoint}/mineral-sites"
        params=site_record
        response = httpx.post(endpoint,json=params,cookies=self.cookies,timeout=None)
        if json.dumps(response.json()).find('exists')>=0:
            print(response.json())
            return {}
        
        response.raise_for_status()
        print(response.json())
        return response.json()
    
    def update_site(self,cdr_id,site_record):
        site_id=self.get_id(cdr_id)
        
        endpoint = f"{self.endpoint}/mineral-sites/{site_id}"
        response = httpx.put(endpoint,json=site_record,cookies=self.cookies,timeout=None)
        
        print(response.json())
        response.raise_for_status()
        return response.json()
    
    def get_id(self,cdr_id):
        endpoint = f"{self.endpoint}/mineral-sites/make-id"
        params={'source_id':"mining-report::https://api.cdr.land/v1/docs/documents",'record_id':cdr_id}
        response = requests.get(endpoint,params=params,cookies=self.cookies,timeout=None)
        #print(response.json())
        response.raise_for_status()
        url=response.json()
        id=url.split('/')[-1]
        return id
    
    def get_site(self,cdr_id):
        site_id=self.get_id(cdr_id)
        
        endpoint = f"{self.endpoint}/mineral-sites/{site_id}"
        response = httpx.get(endpoint,cookies=self.cookies,timeout=None) #params=params,
        if json.dumps(response.json()).find('does not exist')>=0:
            print(response.json())
            return {}
        
        response.raise_for_status() 
        return response.json()
    
    def merge(self,old_record,new_record):
        #override everything except for deposit_type_candidate and created_by
        merged_record=copy.deepcopy(old_record)
        for k in new_record:
            #ignore empty items
            if new_record[k] is None:
                continue
            
            if k=='deposit_type_candidate':
                if not k in merged_record:
                    merged_record[k]=new_record[k]
                else:
                    merged_record[k]+=new_record[k]
                
                #Remove identical records to prevent pollution
                deposit_type_predictions=merged_record[k]
                deposit_type_predictions={json.dumps(x):x for x in deposit_type_predictions}
                deposit_type_predictions=[deposit_type_predictions[x] for x in deposit_type_predictions]
                merged_record[k]=deposit_type_predictions
            elif k=='created_by':
                pass
            else:
                merged_record[k]=new_record[k]
        
        return merged_record
    
    def update_site_safe(self,cdr_id,site_record):
        #Get site
        result=self.create_site(site_record)
        if len(result)==0:
            old_record=self.get_site(cdr_id)
            new_record=self.merge(old_record,site_record)
            result=self.update_site(cdr_id,new_record)
        
        #response.raise_for_status()
        return result