import argparse
import atexit
import hashlib
import hmac
import os
from fastapi.security import APIKeyHeader
import httpx
import ngrok
import uvicorn
import uvicorn.logging
from cdr_schemas.events import Event
from fastapi import (BackgroundTasks, Depends, FastAPI, HTTPException, Request,
                     status)
from pydantic_settings import BaseSettings
from connectAPI import API
from cdr_schemas.document import Document
import sys
sys.path.append(os.path.abspath('/home/ubuntu/ta2_extraction'))

import first_pass.HelperFunctions as helper
import extraction_package.pipeline as extract 
from dotenv import load_dotenv

# Specify the path to the .env file
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './.env'))

# Load the .env file
load_dotenv(dotenv_path)

minmod_api=API('inferlink','inferlink37@')
minmod_api.login()
print("Logged into minmod API: ",minmod_api.whoami())

parser = argparse.ArgumentParser()
args = parser.parse_args()


class Settings(BaseSettings):
    # TO BE CHANGED BY TA3-4 system.
    system_name: str = "xcorp_my_doc_system"
    system_version: str = "0.0.1"
    ml_model_name: str = "xcorp_docs_model"
    ml_model_version: str = "0.0.1"

    # Local port to run on
    local_port: int = 9999
    # To be filled in programmatically via ngrok below.
    callback_url: str = ""
    # Secret string used for signature verification on callback.  Changed by TA3-4 system.
    registration_secret: str = "mysecret"

    # To be provided to TA3-4 system by CDR admin
    user_api_token: str = os.environ["CDR_API_TOKEN"]
    cdr_host: str = "https://api.cdr.land"
    admin_cdr_host: str = "https://admin.cdr.land"
    # For local development
    # cdr_host: str = "http://0.0.0.0:8333"
    # admin_cdr_host: str = "http://0.0.0.0:3333"

    # To be filled in programmatically after registration process below.  Needed to remove registration.
    registration_id: str = ""

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


app_settings = Settings()

# Get ngrok to give us an endpoint
listener = ngrok.forward(app_settings.local_port, authtoken_from_env=True)
app_settings.callback_url = listener.url() + "/hook"


def clean_up():
    # delete our registered system at CDR on program end
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    client.delete(f"{app_settings.cdr_host}/user/me/register/{app_settings.registration_id}", headers=headers)


# register clean_up
atexit.register(clean_up)

app = FastAPI()


async def event_handler(evt: Event):
    try:
        match evt:
            case Event(event="ping"):
                print("Received PING!")
            case Event(event="document.process"):
                print("Received document process event!")
                print(evt.payload)
                document = Document(**evt.payload)
                download_link = f"https://docs.polymer.rocks/cdr/download/{document.id}"
                print(download_link)
                record_id = document.id
                download_dir = "/home/ubuntu/cdr_client_examples/sample_cdr_doc_subscriber/sample_cdr_doc_subscriber/downloaded_reports/"
                output_folder_path = "/home/ubuntu/cdr_client_examples/sample_cdr_doc_subscriber/sample_cdr_doc_subscriber/finished_extractions/"
                file_name = helper.download_document(record_id, download_dir)
                print(f"Finished Downloading: {file_name}")
                
                if file_name is not None:
                    print(f"Going to start extracting: {file_name}")
                    json_output = extract.run(download_dir, file_name, output_folder_path)
                    try: 
                        print(minmod_api.create_site(json_output[0]))
                        print("Finished posting to the API!")
                    except Exception as e:
                        print(f"Had some sort of error: {e}")
                    
                    
                    # download_file_path = os.path.join(download_dir, file_name)
                    
                    # if os.path.exists(download_file_path):
                    #     try:
                    #         os.remove(download_file_path)
                    #     except:
                    #         print("Couldn't delete file")
                            
                    # try:
                    #     for filename in os.listdir(output_folder_path):
                    #         file_path = os.path.join(output_folder_path, filename)
                    #         if os.path.isfile(file_path):
                    #             os.remove(file_path)
                    #             print(f"Successfully deleted {filename} from {output_folder_path}")
                    #         else:
                    #             print(f"{filename} is not a file, skipping.")
                    # except Exception as e:
                    #     print(f"Error deleting files in {output_folder_path}: {e}")
                    
                else:
                    print("Did not extract file")                               
                
            case _:
                print("Nothing to do for event: %s", evt)

    except Exception:
        print("background processing event: %s", evt)
        raise

cdr_signiture = APIKeyHeader(name="x-cdr-signature-256")


async def verify_signature(request: Request, signature_header: str = Depends(cdr_signiture)):

    payload_body = await request.body()
    if not signature_header:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="x-hub-signature-256 header is missing!")

    hash_object = hmac.new(app_settings.registration_secret.encode(
        "utf-8"), msg=payload_body, digestmod=hashlib.sha256)
    expected_signature = hash_object.hexdigest()
    if not hmac.compare_digest(expected_signature, signature_header):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Request signatures didn't match!")

    return True


@app.post("/hook")
async def hook(
    evt: Event,
    background_tasks: BackgroundTasks,
    request: Request,
    verified_signature: bool = Depends(verify_signature),
):
    """Our main entry point for CDR calls"""

    background_tasks.add_task(event_handler, evt)
    return {"ok": "success"}


def run():
    """Run our web hook"""
    uvicorn.run("__main__:app", host="0.0.0.0",
                port=app_settings.local_port, reload=False)


def register_system():
    """Register our system to the CDR using the app_settings"""
    global app_settings
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}

    registration = {
        "name": app_settings.system_name,
        "version": app_settings.system_version,
        "callback_url": app_settings.callback_url,
        "webhook_secret": app_settings.registration_secret,
        # Leave blank if callback url has no auth requirement
        "auth_header": "",
        "auth_token": "",
        # Registers for ALL events
        "events": []

    }

    client = httpx.Client(follow_redirects=True)

    r = client.post(f"{app_settings.cdr_host}/user/me/register",
                    json=registration, headers=headers)

    # Log our registration_id such we can delete it when we close the program.
    app_settings.registration_id = r.json()["id"]


if __name__ == "__main__":
    register_system()
    run()
