import wget
import os

def download_file(url, destination):
    print(f"Downloading {destination} ...")
    wget.download(url, destination)
    print(f"\nDownload complete.")

if __name__ == "__main__":
    # 下載 3D bbox weights
    confirm_url = 'https://docs.google.com/uc?export=download&id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA'
    confirm_cmd = f"wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate '{confirm_url}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p'"
    confirm_code = os.popen(confirm_cmd).read().strip()

    download_url = f"https://docs.google.com/uc?export=download&confirm={confirm_code}&id=1yEiquJg9inIFgR3F-N5Z3DbFnXJ0aXmA"
    download_file(download_url, "epoch_10.pkl")

    # 移除 cookies.txt
    os.remove("/tmp/cookies.txt")

    # 下載 YOLO weights
    yolo_url = 'https://pjreddie.com/media/files/yolov3.weights'
    download_file(yolo_url, "yolov3.weights")
