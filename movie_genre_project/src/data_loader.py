import pandas as pd

def load_dataset(file_path):
    ids, titles, genres, descriptions = [], [], [], []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ::: ")

            if len(parts) == 4:
                ids.append(parts[0])
                titles.append(parts[1])
                genres.append(parts[2])
                descriptions.append(parts[3])

    df = pd.DataFrame({
        "id": ids,
        "title": titles,
        "genre": genres,
        "description": descriptions
    })

    return df