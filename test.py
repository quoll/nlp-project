import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os

TEST_FILE = 'BC7-LitCovid-Test-GS.csv'
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

def count_matches(expected, actual, accumulator):
    match accumulator:
        case None:
            accumulator = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'full_tp': 0, 'count': 0, 'full_count': 0}
        case {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'full_tp': full_tp, 'count': count, 'full_count': full_count}:
            full_count += 1
            if expected == actual:
                full_tp += 1
            for k in expected.keys():
                count += 1
                if expected[k] == 'true':
                    if actual[k] == 'true':
                        tp += 1
                    else:
                        fn += 1
                elif actual[k] == 'true':
                    fp += 1
                else:
                    tn += 1
            accumulator = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'full_tp': full_tp, 'count': count, 'full_count': full_count}
        case _:
            raise ValueError('Invalid accumulator')
    return accumulator

def get_stats(stats):
    match stats:
        case {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'full_tp': full_tp, 'count': count, 'full_count': full_count}:
            accuracy = (tp + tn) / count
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            microacc = full_tp / full_count
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            return f'acc: {accuracy:.4f}, prec: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, μacc: {microacc:.4f}'
        case _:
            raise ValueError('Invalid stats')


def load_data(filename):
    data = pd.read_csv(filename)
    return data

df = load_data(TEST_FILE)
## data[['abstract','label']].to_csv(output_file, index=False)
chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection(name='litcovid', embedding_function=EMD_FN)

# convert the addition to collection to occur in chunks of 100
print(f'Processing {len(df)} records:')
data = process_data(df)
stats = None
for i in range(0, len(df), CHUNK_SIZE):
    print(f'Testing block {i} to {i+CHUNK_SIZE}')
    block = data[i:i+CHUNK_SIZE]
    print(f'  Encoding {len(block)} records...', end='')
    embeddings = MODEL.encode(block['abstract'].tolist())
    print(f'  Querying records')
    results = collection.query(query_embeddings = embeddings.tolist(), n_results = 1)
    assert len(results['ids']) == len(block), f'Expected {len(block)} results, got {len(results["ids"])}'
    for j in range(len(block)):
        expected = block['metadata'][j + i]  # because block is a slice of data, we need to adjust the index
        stats = count_matches(expected, results['metadatas'][j][0], stats)
    print(get_stats(stats))
print('Final stats: ' + get_stats(stats).upper())

