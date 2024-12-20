import os
import time
import numpy as np
import warnings
import logging
from fastapi import FastAPI, Request, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tritonclient.grpc as tritongrpclient
from grpc_client import DetectionTriton, RecognitionTriton
from utils import readb64, image_to_base64
from config import settings

if not os.path.exists("logs"):
    os.mkdir("logs")

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')

triton_client = tritongrpclient.InferenceServerClient(url=settings.TRITON_SERVER_URL, verbose=False)
app = FastAPI(
    title=settings.APP_NAME,
    description="API for license plate detection and recognition",
    version="1.0.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.post('/api/anpr')
async def anpr_api(file: UploadFile = File(...), request: Request = None, region: str = Form(...)):
    """
    Analyze an image for license plate detection and recognition.
    """

    try:
        region_data = settings.TRITON_REGIONS.get(region)
        if region_data:
            detector = DetectionTriton(triton_client, model_name=region_data["detector"],
                                       img_width=settings.IMAGE_WIDTH, img_height=settings.IMAGE_HEIGHT)
            recognizer = RecognitionTriton(triton_client, model_name=region_data["recognizer"])
        else:
            raise HTTPException(
                status_code=401,
                detail=f"ANPR not supported for {region}"
            )

        # Read file content
        contents = await file.read()
        file_size = len(contents)

        # Validate file size
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )

        # Validate file type
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=402,
                detail=f"File type not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )

        # Process image
        image = readb64(contents)

        t1 = time.time()

        # Detection
        result_image, plate_images, flag = detector.detect(image)

        if not flag:
            return JSONResponse(
                status_code=404,
                content={"status": False, "data": "No license plate detected"}
            )

        # Recognition
        labels, probs = recognizer.recognize(plate_images)

        t2 = time.time()
        exec_time = t2 - t1

        return JSONResponse(
            status_code=200,
            content={'status': True,
            'data': {
                'plates': {str(label): float(prob) for label, prob in zip(labels, probs)},
                # 'result_image': image_to_base64(result_image),
                'exec_time': exec_time
            }
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


