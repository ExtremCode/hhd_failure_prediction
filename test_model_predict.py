import requests

if __name__ == "__main__":
    url = "http://localhost:5000/smart"
    with open("data.txt", "r") as f:
        data = [[float(el) for el in f.read().strip().split()]]
    response = requests.post(url, json={"raw_smart": data})
    print(100 * "=")
    print(f"Predicted result: '{response.json()["result"]}'")
    print(100 * "=")
    