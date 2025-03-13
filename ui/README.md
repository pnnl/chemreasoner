# ChemReasoner Web App

This is a web app for visualizing ChemReasoner output data. It is a Svelte-based web interface with a Python Flask backend.

## Setup

1. Install prerequisites: [Node/NPM](https://nodejs.org/en), [Python](https://www.python.org/) (For MacOS, [Homebrew](https://brew.sh/) makes this easy)
1. `cd client`
1. `npm ci`
1. `npm run build`
1. `cd ../server`
1. `python3 -m venv .venv`
1. `source .venv/bin/activate`
1. `pip install -r requirements.txt`


cd client
vi .env.production

# Rebuilding server:
rm -fr .venv

## Data

The `server/app.py` file is currently hard-coded to read `server/data/search_tree_6.json` for the graph data and `server/data/processed_structures/*/*.xyz`, which are not committed to the repository.

## Run

1. `cd server`
1. `./serve.sh --expose` (omit `--expose` to run only on localhost and not exposed on the network). This will run the service in the background. To run it in the foreground: `python3 app.py --expose`.
1. Open `http://localhost:8000` in a web browser (if "exposed" you can change `localhost` to the name of your computer and the URL can be shared within PNNL as long as your computer is serving the web app)
