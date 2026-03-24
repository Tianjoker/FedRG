import os
import json
import numbers


def _save_client_selection(stage: str, round_idx: int, client_indexes):
    def to_int_list(xs):
        out = []
        for x in xs:
            if isinstance(x, numbers.Number):
                out.append(int(x))
            else:
                out.append(int(str(x)))
        return out

    client_list = to_int_list(client_indexes)
    save_dir = "output_dir"
    file_name = "client_selections-noprojector.json"
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, file_name)
    payload = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}

    payload.setdefault(stage, {})
    payload[stage][f"round_{round_idx}"] = client_list

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_client_selection_cached():
    json_path = os.path.join("/home/cmcc/went/IFNLL-NEW/IFNLL-11-08/output_dir", "client_selections.json")
    print(json_path)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}

