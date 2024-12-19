#!/bin/bash
python3 -m uvicorn main:app --workers 4 --reload --host 0.0.0.0 --port 9003