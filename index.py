import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os

TRAINING_FILE = 'BC7-LitCovid-Train.csv'
MODEL = 'ls-da3m0ns/bge_large_medical'

EMD_FN = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL)
MODEL = SentenceTransformer(MODEL)

CHUNK_SIZE = 100

def metadata(label):
    md = {'Epidemic Forecasting': 'false',
          'Diagnosis': 'false',
          'Mechanism': 'false',
          'Case Report': 'false',
          'Prevention': 'false',
          'Treatment': 'false',
          'Transmission': 'false'}
    for l in label.split(';'):
        md[l] = 'true'
    return md

def process_data(data):
    data['pmid'] = data['pmid'].astype(str).apply(lambda x: 'i' + x)
    data['metadata'] = data['label'].apply(metadata)
    return data[['pmid', 'abstract', 'metadata']]


def load_data(filename):
    data = pd.read_csv(filename)
    return data

df = load_data(TRAINING_FILE)
## data[['abstract','label']].to_csv(output_file, index=False)
chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection(name='litcovid', embedding_function=EMD_FN)

# convert the addition to collection to occur in chunks of 100
print(f'Processing {len(df)} records:')
data = process_data(df)
for i in range(0, len(df), CHUNK_SIZE):
    print(f'Processing block {i} to {i+CHUNK_SIZE}')
    block = data[i:i+CHUNK_SIZE]
    print(f'  Encoding {len(block)} records...', end='')
    embeddings = MODEL.encode(block['abstract'].tolist())
    print(f' Indexing records')
    collection.add(
            ids = block['pmid'].tolist(),
            embeddings = embeddings.tolist(),
            documents = block['abstract'].tolist(),
            metadatas = block['metadata'].tolist())

