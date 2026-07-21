import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Document
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

# Now retrieve them
api_key = os.getenv("QDRANT_API_KEY")
url = os.getenv("QDRANT_URL")

client = QdrantClient(
    url=url,
    api_key=api_key,
    cloud_inference=True,
)

points = [
    PointStruct(
        id=1,
        payload={"topic": "cooking", "type": "dessert"},
        vector=Document(
            text="Responsible Staff Member for Learning in Real and Virtual Humans",
            model="sentence-transformers/all-minilm-l6-v2",
        ),
    )
]

client.upsert(collection_name="university_knowledge_base", points=points)

points = client.query_points(
    collection_name="university_knowledge_base",
    query=Document(
        text="Responsible Staff Member for Learning in Real and Virtual Humans",
        model="sentence-transformers/all-minilm-l6-v2",
    ),
)

print(points)
