## A sample project which registers to the CDR as a document subscriber.

This project leverages the schemas https://github.com/DARPA-CRITICALMAAS/cdr_schemas for inputs and outputs and connects to the CDR to receive `document` events.

In order to use you must have a CDR token **\_with doc** access. Please ensure your token has this before running by asking CDR admin.

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

#### Host as webhook, received document.process events from CDR and spits out event information. Add business logic as needed.

`NGROK_AUTHTOKEN=<YOUR_NGROK_TOKEN> CDR_API_TOKEN=<YOUR_CDR_TOKEN> poetry run python sample_cdr_document_subscriber/server.py`
