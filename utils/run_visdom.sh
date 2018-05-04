#!/bin/bash

python -m visdom.server -p 8099 &
../../ngrok http 8099
