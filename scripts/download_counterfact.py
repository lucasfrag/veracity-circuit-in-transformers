import json
import urllib.request
from pathlib import Path

URL = "https://rome.baulab.info/data/dsets/counterfact.json"
OUT_DIR = Path("data/counterfact")
OUT_FILE = OUT_DIR / "counterfact.json"

def download():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists():
        print(f"Já existe: {OUT_FILE}. Pulando download.")
        return

    print(f"Baixando CounterFact...")
    urllib.request.urlretrieve(URL, OUT_FILE)
    print(f"Salvo em {OUT_FILE}")

    with open(OUT_FILE) as f:
        data = json.load(f)

    print(f"Total de entradas: {len(data)}")
    print("\nExemplo:")
    rw = data[0]["requested_rewrite"]
    print(f"  Prompt      : {rw['prompt'].format(rw['subject'])}")
    print(f"  Target true : {rw['target_true']['str']}")
    print(f"  Target false: {rw['target_false']['str']}")

if __name__ == "__main__":
    download()