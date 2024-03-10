
import pandas as pd
import json

def read_in(json_file: str) -> pd.DataFrame:
    data = []
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Load each line as a JSON object and append to the list
            data.append(json.loads(line))

    # Create a DataFrame from the list of dictionaries
    return pd.DataFrame(data)