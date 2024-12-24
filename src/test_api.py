import requests
import base64
from pathlib import Path
import time

def save_base64_image(base64_image: str, output_path: str):
    """
    Decodes a base64 string and saves it as an image file.

    Args:
        base64_image (str): Base64-encoded string of the image.
        output_path (str): Path to save the decoded image.
    """
    image_data = base64.b64decode(base64_image)
    with open(output_path, "wb") as file:
        file.write(image_data)


def check_anpr_api(endpoint_url: str, image_path: str, region: str, output_image_path: str):
    """
    Tests the ANPR API by sending an image and saving the resulting image.

    Args:
        endpoint_url (str): URL of the ANPR API endpoint.
        image_path (str): Path to the input image file.
        region (str): Region to specify in the API call.
        output_image_path (str): Path to save the result image.
    """
    try:
        # Open the image file
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            data = {"region": region}

            # Send POST request to the API
            response = requests.post(endpoint_url, files=files, data=data)

        # Check the response status
        if response.status_code == 200:
            result = response.json()
            if result["status"]:
                # Save the result image
                # save_base64_image(result["data"]["result_image"], output_image_path)
                print(f"Result image saved at {output_image_path}")
                print(f"Detected plates: {result['data']['plates']}")
                print(f"execution time: {result['data']['exec_time']* 1000:.2f} ms")
            else:
                print(f"API returned no plates detected: {result['data']}")
        else:
            print(f"Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"Error: {e}")

# Usage
if __name__ == "__main__":
    API_URL = "http://10.0.11.91:9003/api/anpr"  # 185.4.182.34
    INPUT_IMAGE = "debug/cis/squared.jpg"
    REGION = "CIS"  # Replace with the desired region
    OUTPUT_IMAGE = "result_image.jpg"  # Path to save the result image
    for i in range(1):
        check_anpr_api(API_URL, INPUT_IMAGE, REGION, OUTPUT_IMAGE)