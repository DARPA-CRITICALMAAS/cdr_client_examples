## A sample project which registers to the CDR as a georeferencer.

This project leverages the schemas https://github.com/DARPA-CRITICALMAAS/cdr_schemas for inputs and outputs and connects to the CDR to receive `process.map` events and to produce `georef` results.

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

#### Host as webhook, received process.map events from CDR and process as received (integration)

`NGROK_AUTHTOKEN=<YOUR_NGROK_TOKEN> CDR_API_TOKEN=<YOUR_CDR_TOKEN> poetry run python sample_cdr_georef/server.py host`

#### Process single cog based on cog_id (to be found via CDR getter map functions).

`NGROK_AUTHTOKEN=<YOUR_NGROK_TOKEN> CDR_API_TOKEN=<YOUR_CDR_TOKEN> poetry run python sample_cdr_georef/server.py process --cog_ig <COG_ID>`

This will process only the single map and exit - you can use this for testing and to send results to the CDR for any maps you'd like.
