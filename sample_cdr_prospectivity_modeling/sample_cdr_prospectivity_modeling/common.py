import argparse
import asyncio
import os
import shutil
from pathlib import Path

import httpx
import rasterio as rio
from rasterio.mask import mask

from cdr_schemas.prospectivity_input import (ProspectivityOutputLayer, SaveProcessedDataLayer)

def get_event_payload_result(id: str, app_settings):
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True, timeout=None)
    resp = client.get(f"{app_settings.admin_cdr_host}/admin/events/event/{id}",
                      headers=headers)
    return resp.json()



def read_tiff(file_path):
    with rio.open(file_path) as src:
        return src.read(1), src


def clip_tiff(tiff_path, mask_tiff_path, output_path):
    # placeholder to clip tiff
    shutil.copy(mask_tiff_path, output_path)


def prepare_data_sources(payload):
    print("Downloading template cma file")
    updated_url=  payload.cma.download_url
    
    # to remove. local testing
    if "minio.cdr.geo" in payload.cma.download_url:
        updated_url= "http://0.0.0.0:9000/" + payload.cma.download_url.split(":9000/")[-1]
    
    if not os.path.exists(f"datasources/{payload.cma.download_url.split('/')[-1]}"):
        r = httpx.get(updated_url, timeout=5000)
        with open(f"datasources/{payload.cma.download_url.split('/')[-1]}", "wb") as f:
            f.write(r.content)

    print('preparing data sources')
    print("loop over evidence layers specified by the UI")
    for layer in payload.evidence_layers:
        print(f"downloading datasource layer from cdr: {layer} ")
        if not os.path.exists(f"datasources/{layer.data_source.download_url.split('/')[-1]}"):
            r = httpx.get(updated_url, timeout=5000)
            with open(f"datasources/{layer.data_source.download_url.split('/')[-1]}", "wb") as f:
                f.write(r.content)
    
    print("CMA template and layers are downloaded. Now clip them to template extent")
    for layer in payload.evidence_layers:
        clip_tiff(
            tiff_path = f"datasources/{layer.data_source.download_url.split('/')[-1]}", 
            mask_tiff_path = f"datasources/{payload.cma.download_url.split('/')[-1]}", 
            output_path = f"datasources/clipped_{layer.data_source.download_url.split('/')[-1]}"
        )

    return


def train_model(payload):
    print("Train model on new process stack ...")
    print("model is trained")  
    return


def run_model(payload):
    print('run model to generate output')
    shutil.copy(f"datasources/{payload.cma.download_url.split('/')[-1]}", "outputs/model_output_uncertainty.tif")
    shutil.copy(f"datasources/{payload.cma.download_url.split('/')[-1]}", "outputs/model_output_likelihood.tif")
    print("model runs have finished")
    return


def send_outputs(payload, app_settings):
    print("Sending Output layers to CDR...")

    #  send output layers from model run
    #  create output layer metadata
    results = ProspectivityOutputLayer(**{
        "system": app_settings.system_name,
        "system_version": app_settings.system_version,
        "model": app_settings.ml_model_name,
        "model_version": app_settings.ml_model_version,
        "model_run_id": payload.model_run_id,
        "cma_id": payload.cma.cma_id,
        "output_type": "uncertainty",
        "title":"model_ouput_uncertainty.tif"
    })
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    files_ = {"input_file": (
        "model_output_uncertainty.tif", 
        open("./outputs/model_output_uncertainty.tif", "rb"), "application/octet-stream")
        }

    resp = client.post(f"{app_settings.cdr_host}/v1/prospectivity/propectivity_output_layer",
                       data={
                        "metadata": results.model_dump_json(exclude_none=True)
                        },
                        files=files_,
                        headers=headers)
    if resp.status_code != 200 or resp.status_code != 204:
        print("An Error Occurred sending uncertainty layer")
        print(resp.text)
    else:
        print("Finished sending uncertainty!")
    
    #  additional output layer's metadata
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
    files_ = {"input_file": (
        "model_output_likelihood.tif",
        open("./outputs/model_output_likelihood.tif", "rb"),
        "application/octet-stream")
        }

    resp = client.post(f"{app_settings.cdr_host}/v1/prospectivity/propectivity_output_layer",
            data={
                "metadata": result_2.model_dump_json(exclude_none=True)
                },
            files=files_,
            headers=headers)
    if resp.status_code != 200 or resp.status_code != 204:
        print("An Error Occurred sending likelihood layer")
        print(resp.text)
    else:
        
        print("Finished sending likelihood!")
    return

def send_stack(payload, app_settings):
    headers = {'Authorization': f'Bearer {app_settings.user_api_token}'}
    client = httpx.Client(follow_redirects=True)
    print("Now sending processed data layers")
    for layer in payload.evidence_layers:
        data_layer = SaveProcessedDataLayer(**{
            "system": app_settings.system_name,
            "system_version": app_settings.system_version,
            "data_source_id": layer.data_source.data_source_id,
            "model_run_id": payload.model_run_id,
            "cma_id": payload.cma.cma_id,
            "transform_methods":layer.transform_methods,
            "title":f"processed_{layer.data_source.data_source_id}"
            })
        
        files_ = {"input_file": (
                    f"{layer.data_source.download_url.split('/')[-1]}",
                    open(f"./datasources/{layer.data_source.download_url.split('/')[-1]}", "rb"),
                    "application/octet-stream"
                )}

        resp = client.post(f"{app_settings.cdr_host}/v1/prospectivity/propectivity_input_layer",
            data={
                "metadata": data_layer.model_dump_json(exclude_none=True)
                },
            files=files_,
            headers=headers
            )
        if resp.status_code != 200 or resp.status_code != 204:
            print("An Error Occurred sending input layer")
            print(resp.text)
        else:
            
            print("Finished sending input layer!")

# stub steps to mimic ta3 model code
def run_ta3_pipeline(payload, app_settings):
    prepare_data_sources(payload=payload)
    train_model(payload=payload)
    run_model(payload=payload)
    send_outputs(payload=payload, app_settings=app_settings)
    send_stack(payload=payload, app_settings=app_settings)
    print("finished!")
