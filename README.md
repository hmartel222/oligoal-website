# OligoAL Website

The ASO workbench serves a browser interface and a Python design API. It
scores supported chemistries with four released OligoGym datasets and
exports ranked candidates and duplex files. The original project checkout
is optional. See the [workbench guide](app/README.md) for scoring coverage
and the PyMOL workflow.

## Run locally

From the repository root with [uv](https://docs.astral.sh/uv/) installed:

```sh
./scripts/setup-oligogym.sh
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-website.txt
./"Run Website.command"
```

Open [localhost:8128](http://127.0.0.1:8128/). On macOS, you can also
double-click `Run Website.command` after setup. Gene lookup uses Ensembl.
Paste a transcript to run a design without an Ensembl lookup.

A fresh clone starts with legal-move traversal for 2′-MOE design.
To enable exploratory PPO, run the following after setup and restart the
server. The checkpoint and transcript cache are local runtime artifacts.

```sh
export PYTHONPATH="$PWD/src:$PWD/vendor/OligoGym"
.venv/bin/python app/server.py --train-policy artifacts/aso-app-policy
```

## Host the website

The Dockerfile targets Linux x86-64. It starts the API and serves the
frontend on the same port. Set `PORT` if your host supplies a port.
Allow startup time for the OligoGym model fits. A fresh macOS checkout
took about 94 seconds to fit and score both chemistry panels, with a peak
resident memory of 3.3 GiB. Measure the Linux container before choosing an
instance size. Render's free web service has a 512 MB limit.
[Render compute plans](https://render.com/docs/compute-plans)

```sh
docker build --platform linux/amd64 -t oligoal-website .
docker run --rm --platform linux/amd64 -p 8080:8080 oligoal-website
```

For a free temporary demo, run the app locally on port 8128 and run
`cloudflared tunnel --url http://localhost:8128`. Use
`--allowed-mail you@example.com` to require an email PIN. The URL lasts
while the local app and tunnel are running. [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)

An independent cloud deployment needs a host that runs the Python backend.
GitHub Pages serves static files only. Cloud Run can run the container
and has a free usage allowance on a billing account. [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages),
[Cloud Run pricing](https://cloud.google.com/run/pricing)
