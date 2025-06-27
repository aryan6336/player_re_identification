import gdown

# Replace with your actual file ID
file_id = "1A2B3CxyzABC12345678"
url = f"https://drive.google.com/uc?id={file_id}"

output = "models/best.pt"
gdown.download(url, output, quiet=False)

print("✅ Model downloaded successfully.")
