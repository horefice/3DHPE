#!/bin/bash

# ssh -R narvis:80:localhost:8099 serveo.net &
../../ngrok http 8099 &
python -m visdom.server -p 8099