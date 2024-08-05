import argparse
import os
from pathlib import Path

from cdr_schemas.cdr_responses.prospectivity import ProspectModelMetaData

from pydantic_settings import BaseSettings
from .common import run_ta3_pipeline, get_event_payload_result


parser = argparse.ArgumentParser()
parser.add_argument("--event_id")
args = parser.parse_args()


class Settings(BaseSettings):
    # TO BE CHANGED BY TA3-4 system.
    system_name: str = "xcorp_prospectivity"
    system_version: str = "0.0.1"
    ml_model_name: str = "xcorp_prospectivity_model"
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
    # cdr_host: str = "http://0.0.0.0:8333"
    # admin_cdr_host: str = "http://0.0.0.0:3333"

    # To be filled in programmatically after registration process below.  Needed to remove registration.
    registration_id: str = ""

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"



app_settings = Settings()


if __name__ == "__main__":
    # make sure folders are created
    datasources_path = Path('./datasources')
    if not datasources_path.exists():
        datasources_path.mkdir(parents=True, exist_ok=True)
    outputs_path = Path('./outputs')
    if not outputs_path.exists():
        outputs_path.mkdir(parents=True, exist_ok=True)
    
    # event id is created from a new model run event. Should be provided by ta4
    if args.event_id:
        event_id = args.event_id
        model_resp = get_event_payload_result(id=event_id, app_settings=app_settings)

        if model_resp.get("event") != "prospectivity_model_run.process":
            raise Exception("Event is not found or is not a model run event")
        
        model_payload= model_resp.get("payload")

        run_ta3_pipeline(
            ProspectModelMetaData(
                model_run_id = model_payload.get("model_run_id"),
                cma = model_payload.get("cma"),
                model_type = model_payload.get("model_type"),
                train_config = model_payload.get("train_config"),
                evidence_layers = model_payload.get("evidence_layers"),
                ), 
                app_settings=app_settings)
        
