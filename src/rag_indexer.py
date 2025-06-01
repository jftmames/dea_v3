# src/rag_indexer.py

import os
import glob

def reindex_rag(corpus_folder: str):
    """
    Recorre todos los documentos en corpus_folder y reindexa embeddings.
    """
    docs = glob.glob(os.path.join(corpus_folder, "*.pdf")) + glob.glob(os.path.join(corpus_folder, "*.txt"))
    # Lógica para vectorizar cada doc y actualizar la base de vectores
    for doc in docs:
        # extraer texto, generar embedding, guardar en DB
        pass
