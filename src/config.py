from pydantic_settings import BaseSettings

class Settings():
    APP_NAME: str = "License Plate Recognition API"
    DEBUG: bool = False
    IMAGE_WIDTH: int = 512
    IMAGE_HEIGHT: int = 512
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png"}
    TRITON_REGIONS: dict = {
        "USA": {
            "detector" : "usa_detection",
            "recognizer": "usa_ensemble"},
        "CIS": {
            "detector" : "cis_detection",
            "recognizer": "cis_ensemble"
        }
    }
    TRITON_SERVER_URL: str = "127.0.0.1:8001"
    class Config:
        env_file = ".env"

settings = Settings()
