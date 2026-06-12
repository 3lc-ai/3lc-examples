"""Create a small demo table with images fetched from picsum.photos.

Each row stores an ``img-https://`` URL that the HttpImageUrlAdapter resolves
at read time. The images are persistent — the same ``/id/<N>/`` path always
returns the same photo.

Usage::

    pip install -e .   # install the adapter first
    python -m http_image_url_adapter.create_demo_table
"""

import tlc

# A curated selection of picsum.photos images (deterministic via /id/<N>/).
# Each tuple is (picsum_id, author/subject, width, height).
IMAGES = [
    (10, "forest trail", 400, 300),
    (11, "dark leaves", 400, 300),
    (12, "sand dunes", 400, 300),
    (14, "mountain road", 400, 300),
    (15, "hilltop ruins", 400, 300),
    (16, "rocky beach", 400, 300),
    (17, "ocean horizon", 400, 300),
    (18, "misty forest", 400, 300),
    (19, "autumn trees", 400, 300),
    (20, "stacked books", 400, 300),
]

PROJECT_NAME = "3LC Tutorials - HTTP Image URL Adapter"
DATASET_NAME = "picsum-photos"
TABLE_NAME = "initial"


def main() -> None:
    table = tlc.Table.from_dict(
        data={
            "image": [f"img-https://picsum.photos/id/{pid}/{w}/{h}" for pid, _, w, h in IMAGES],
            "description": [desc for _, desc, _, _ in IMAGES],
            "picsum_id": [pid for pid, _, _, _ in IMAGES],
        },
        schema={
            "image": tlc.schemas.ImageSchema(sample_type="url"),
            "description": tlc.schemas.StringSchema(),
            "picsum_id": tlc.schemas.Int32Schema(writable=False),
        },
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=TABLE_NAME,
        if_exists="overwrite",
    )

    print(f"Created table with {len(table)} rows")
    print(f"Table URL: {table.url}")
    print()
    print("Sample row:")
    print(f"  image URL:   {table.table_rows[0]['image']}")
    print(f"  description: {table.table_rows[0]['description']}")
    print()
    print("Open the 3LC Dashboard to browse the images.")


if __name__ == "__main__":
    main()
