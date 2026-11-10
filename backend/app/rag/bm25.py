"""Retriever BM25 sederhana (tanpa dependensi tambahan).

Dipakai dulu supaya simulator langsung jalan. Antarmuka `search()` sengaja dibuat sama
dengan retriever lain, jadi semantic search dan hybrid nanti bisa ditukar tanpa mengubah pipeline.
"""
import math
import re
from collections import Counter
from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from app.rag.chunking import Chunk, Hit

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "atau", "pada", "adalah",
    "ada", "apa", "apakah", "bisa", "boleh", "saya", "aku", "kami", "kita", "anda", "mau", "ingin",
    "gak", "ga", "nggak", "tidak", "ya", "dong", "sih", "kah", "nya", "akan", "juga", "sudah",
    "bagaimana", "berapa", "kalau", "jika", "agar", "sama", "lagi", "saja", "aja", "kak", "bu", "pak",
}
_stemmer = StemmerFactory().create_stemmer()


@lru_cache(maxsize=20000)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def tokenize(text: str) -> list[str]:
    """lowercase -> pecah huruf/angka -> buang stopword -> stemming Sastrawi."""
    text = text.lower()
    text = re.sub(r"(?<=\d)[.,](?=\d)", "", text)  # 200.000 -> 200000
    tokens = []
    for tok in re.findall(r"[a-z]+|\d+", text):  # "rp200000" -> "rp", "200000"
        tok = tok if tok.isdigit() else _stem(tok)  # "pembayaran" -> "bayar", "bayarnya" -> "bayar"
        if tok not in STOPWORDS and len(tok) > 1:
            tokens.append(tok)
    return tokens


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.index_text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(chunks)) if chunks else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = tokenize(query)
        scored = []
        for i, chunk in enumerate(self.chunks):
            tf, dl = self._tf[i], self._len[i]
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avg))
            if s > 0:
                scored.append(Hit(chunk, s))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
