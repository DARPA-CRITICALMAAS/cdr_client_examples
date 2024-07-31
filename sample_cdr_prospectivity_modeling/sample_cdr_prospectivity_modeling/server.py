import argparse
import asyncio
import atexit
import hashlib
import hmac
import os
import uuid
from typing import Any
import shutil

import httpx
import ngrok
import rasterio as rio
import rasterio.transform as riot
import uvicorn
import uvicorn.logging
from cdr_schemas.events import Event, MapEventPayload
from cdr_schemas.prospectivity_input import (ProspectivityOutputLayer, SaveProcessedDataLayer)
from cdr_schemas.cdr_responses.prospectivity import ProspectModelMetaData
from fastapi import (BackgroundTasks, Depends, FastAPI, HTTPException, Request,
                     status)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from pydantic_settings import BaseSettings



parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--cog_id")
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


async def get_feature_result(id: str):
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True, timeout=None)
    resp = client.get(f"{app_settings.cdr_host}/v1/maps/extractions/{id}",
                      headers=headers)
    return resp.json()


async def post_feature_results(cog_id: str, cog_url: str):

    # Download .cog
    print("Downloading...")
    if not os.path.exists(f"{cog_id}.cog.tif"):
        r = httpx.get(cog_url, timeout=1000)
        with open(f"{cog_id}.cog.tif", "wb") as f:
            f.write(r.content)

    payload = dummy_feature_results(cog_id)

    print("Sending Feature to CDR...")
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    resp = client.post(f"{app_settings.cdr_host}/v1/maps/publish/features",
                       data=payload.model_dump_json(), headers=headers)
    print("Posted Features to CDR!")

def read_tiff(file_path):
    with rio.open(file_path) as src:
        return src.read(1), src

def clip_tiff(tiff_path, mask_tiff_path, output_path):
    # Read the mask TIFF file
    mask_data, mask_src = read_tiff(mask_tiff_path)

    # Create a mask from the mask TIFF (assuming the mask is defined by non-zero values)
    mask = mask_data != 0

    # Read the TIFF file to be clipped
    with rio.open(tiff_path) as src:
        # Clip the image with the mask
        out_image, out_transform = mask(src, [mask], crop=True)

        # Update the metadata
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        # Write the clipped image to a new TIFF file
        with rio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


def prepare_data_sources(payload):
    print("downloading template cma file")
    if not os.path.exists(f"datasources/{payload.cma.download_url.split("/")[-1]}"):
        r = httpx.get(payload.cma.download_url, timeout=5000)
        with open(f"datasources/{payload.cma.download_urls.split("/")[-1]}", "wb") as f:
            f.write(r.content)
    print('preparing data sources')
    print("loop over evidence layers specified by the UI")
    
    for layer in payload.evidence_layers:
        print(f"downloading datasource layer from cdr: {payload.data_source_id} ")
        if not os.path.exists(f"datasources/{layer.download_url.split("/")[-1]}"):
            r = httpx.get(payload.download_url, timeout=5000)
            with open(f"datasources/{layer.download_urls.split("/")[-1]}", "wb") as f:
                f.write(r.content)
    
    print("CMA template and layers are downloaded. Now clip them to template extent")
    for layer in payload.evidence_layers:
        clip_tiff(
            tiff_path = f"datasources/{layer.download_url.split("/")[-1]}", 
            mask_tiff_path = f"datasources/{payload.cma.download_urls.split("/")[-1]}", 
            output_path = f"datasources/clipped_{layer.download_url.split("/")[-1]}"
        )

    return


def train_model(payload):
    print("Train model on new process stack ...")
    print("model is trained")  
    return


def run_model(payload):
    print('run model to generate output')
    shutil.copy(f"datasources/{payload.cma.download_urls.split("/")[-1]}", "outputs/model_ouput_uncertainty.tif")
    shutil.copy(f"datasources/{payload.cma.download_urls.split("/")[-1]}", "outputs/model_ouput_likelihood.tif")
    print("model runs have finished")
    return


def send_outputs(payload):
    print("send outputs to cdr.")
    print("Sending Result to CDR...")

    results = ProspectivityOutputLayer(**{
        "system": app_settings.system_name,
        "system": app_settings.system_name,
        "model": app_settings.ml_model_name,
        "model_version": app_settings.ml_model_version,
        "model_run_id": payload.model_run_id,
        "cma": payload.cma.cma_id,
        "output_type": "uncertainty",
        "title":"model_ouput_uncertainty.tif"
    })
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    resp = client.post(f"{app_settings.cdr_host}/v1/prospectivity/prospectivity_output_layers",
                       data={
                           "metadata": results.model_dump_json(exclude_none=True)
                           },
                            files=["outputs/model_ouput_uncertainty.tif"],
                            headers=headers)
    print("Finished sending uncertainty!")
    result_2 = ProspectivityOutputLayer(**{
        "system": app_settings.system_name,
        "system_version": app_settings.system_version,
        "model": app_settings.ml_model_name,
        "model_version": app_settings.ml_model_version,
        "model_run_id": payload.model_run_id,
        "cma_id": payload.cma.cma_id,
        "output_type": "likelihood",
        "title":"model_ouput_likelihood.tif"
    })
    resp = client.post(f"{app_settings.cdr_host}/v1/prospectivity/prospectivity_output_layers",
            data={
                "metadata": result_2.model_dump_json(exclude_none=True)
                },
                files=["outputs/model_ouput_likelihood.tif"],
                headers=headers)
    print("Finished sending likelihood!")
    return

def send_stack(payload):
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    print("Now sending processed data layers")
    for layer in payload.evidence_layers:

        data_layer = SaveProcessedDataLayer(**{
            "system": app_settings.system_name,
            "system_version": app_settings.system_version,
            "data_source_id": layer.data_source_id,
            "model_run_id": payload.model_run_id,
            "cma_id": payload.cma.cma_id,
            "transform_methods":layer.transform_methods,
            "title":f"processed_{layer.data_source_id}"
            })
        print(data_layer)
        resp = client.post(f"{app_settings.cdr_host}/v1/prospectivity/propectivity_input_layer",
        data={
            "metadata": data_layer.model_dump_json(exclude_none=True)
            },
            files=[f"datasources/{layer.download_urls.split("/")[-1]}"],
            headers=headers)


def run_ta3_pipeline(payload):
    prepare_data_sources(payload=payload)
    train_model(payload=payload)
    run_model(payload=payload)
    send_outputs(payload=payload)
    send_stack(payload=payload)
    print("finished!")


async def event_handler(evt: Event):
    try:
        match evt:
            case Event(event="ping"):
                print("Received PING!")
            case Event(event="prospectivity_model_run.process"):
                print("Received model run event payload!")
                print(evt.payload)
                prepare_data_sources(evt.payload)
            case _:
                print("Nothing to do for event: %s", evt)

    except Exception:
        print("background processing event: %s", evt)
        raise



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
    if args.mode == 'host':
        register_system()
        run()
    if args.mode == 'process':
        asyncio.run(run_ta3_pipeline(
            ProspectModelMetaData(
                **{
                "train_config": {
                "grid_type": "rectangular",
                "initial_neighborhood_size": 0,
                "som_type": "toroid",
                "dimensions_y": 20,
                "dimensions_x": 20,
                "initial_learning_rate": 0,
                "final_neighborhood_size": 1,
                "gaussian_neighborhood_coefficient": 0.5,
                "learning_rate_decay": "linear",
                "final_learning_rate": 0,
                "size": 20,
                "num_initializations": 5,
                "neighborhood_decay": "linear",
                "num_epochs": 10,
                "som_initialization": "random",
                "neighborhood_function": "gaussian"
                },
                "evidence_layers": [
                {
                    "transform_methods": [
                    "log",
                    "minmax"
                    ],
                    "title": "wowow",
                    "data_source": {
                    "evidence_layer_raster_prefix": "evidence_layer_raster_prefix",
                    "format": "tif",
                    "description": "description",
                    "reference_url": "http",
                    "type": "continuous",
                    "resolution": [
                        3,
                        3
                    ],
                    "derivative_ops": "derivative_ops",
                    "download_url": "http://minio.cdr.geo:9000/public.cdr.land/prospectivity/inputs/14bc3b6d0d124aae8174816328841242.tif",
                    "publication_date": "2024-07-30T14:27:38.140927",
                    "subcategory": "subcategory",
                    "category": "geophysics",
                    "data_source_id": "evidence_layer_raster_prefix_res0_3_res1_3_cat_LayerCategoryGEOPHYSICS",
                    "authors": [
                        "kyle"
                    ],
                    "DOI": "DOI"
                    }
                }
                ],
                "cma": {
                "extent": {
                    "coordinates": [
                    [
                        [
                        [
                            -124.409591,
                            32.534156
                        ],
                        [
                            -114.131211,
                            32.534156
                        ],
                        [
                            -114.131211,
                            42.009518
                        ],
                        [
                            -124.409591,
                            42.009518
                        ],
                        [
                            -124.409591,
                            32.534156
                        ]
                        ]
                    ]
                    ],
                    "type": "MultiPolygon"
                },
                "crs": "EPSG:4267",
                "cogs": [],
                "download_url": "http://minio.cdr.geo:9000/public.cdr.land/prospectivity/cmas/EPSG:4267_a19ad220b36b06dfbf8438e418c7718e776f5e6be4f2f1fa6f56605006b400a7__res0_3_res1_3_li/template_raster.tif",
                "description": "This is it",
                "mineral": "li",
                "resolution": [
                    3,
                    3
                ],
                "cma_id": "EPSG:4267_a19ad220b36b06dfbf8438e418c7718e776f5e6be4f2f1fa6f56605006b400a7__res0_3_res1_3_li"
                },
                "model_type": "som",
                "model_run_id": "d6d9fe92954145749a1a49fb748388dc"
            }
            )
        )
        )
        #     {"cog_id": args.cog_id, "cog_url": f"https://s3.amazonaws.com/public.cdr.land/cogs/{args.cog_id}.cog.tif"}))
        # asyncio.run(post_feature_results(cog_id=args.cog_id,
        #             cog_url=f"https://s3.amazonaws.com/public.cdr.land/cogs/{args.cog_id}.cog.tif"))
        # asyncio.run(get_all_extraction_results(cog_id=args.cog_id))
