## A sample project which registers to the CDR as a prospectivity model.

This project leverages the schemas https://github.com/DARPA-CRITICALMAAS/cdr_schemas for inputs and outputs and connects to the CDR to receive `prospectivity_model_run.process` events and to produce output/input raster layers.

In order to use you must have a CDR token **_with georef_** access. Please ensure your token has this before running by asking CDR admin.

You should also change the system information which starts at the line:
`class Settings(BaseSettings):`
to be what you want.

### Requirements

```
python >= 3.10
poetry
```

This sample uses ngrok (https://dashboard.ngrok.com/signup) to obtain and use a throw-away public secure webhook. If you want to try this code as is, please sign up to get an auth token. It's free!

### Install

`poetry install`

### Run (2 modes) - Choose 1.

#### Host as webhook, received prospectivity_model_run.process events from CDR and process as received (integration)

`NGROK_AUTHTOKEN=<YOUR_NGROK_TOKEN> CDR_API_TOKEN=<YOUR_CDR_TOKEN> poetry run python sample_cdr_prospectivity_modeling/server.py`

#### Process single model run event based on event_id (to be found via CDR/TA4).

`NGROK_AUTHTOKEN=<YOUR_NGROK_TOKEN> CDR_API_TOKEN=<YOUR_CDR_TOKEN> poetry run python sample_cdr_prospectivity_modeling/script.py process --event_id <EVENT_ID>`

This will pull the event information from the cdr and produce fake model output files and fake processed data layers used to train the model and push those back to the cdr with correct metadata.
