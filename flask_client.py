import requests

SERVER_URL = "http://lamb.mech.northwestern.edu:5000/get_3d_mesh"

def request_3d_mesh(image_path, mask_path, seed=42, output_ply="splat_client.ply"):
    files = {
        "image": ("image.png", open(image_path, "rb"), "image/png"),
        "mask": ("mask.png", open(mask_path, "rb"), "image/png"),
    }
    data = {
        "seed": str(seed)
    }

    resp = requests.post(SERVER_URL, files=files, data=data)

    if resp.status_code == 200:
        with open(output_ply, "wb") as f:
            f.write(resp.content)
        print(f"Saved reconstruction to {output_ply}")
    else:
        print("Request failed:", resp.status_code, resp.text)

if __name__ == "__main__":
    request_3d_mesh("test_image.png", "test_mask.png", seed=42)
