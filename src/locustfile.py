from locust import HttpUser, task, between
import random
import time
import os

class ANPRApiUser(HttpUser):
    wait_time = between(1, 2)  # Virtual user waits 1 to 2 seconds between tasks

    @task
    def anpr_api(self):
        start_time = time.time()
        region = random.choice(["USA"])  # Random region
        # Choose an image for testing
        file_path = "../debug/usa/usa.jpeg"
        with open(file_path, "rb") as f:
            file_data = f.read()

        # Prepare request data
        files = {
            "file": (os.path.basename(file_path), file_data, "image/jpeg"),
            "region": (None, region)
        }

        response = self.client.post("/api/anpr", files=files)
        end_time = time.time()
        print(f"Request Time: {end_time - start_time} seconds")