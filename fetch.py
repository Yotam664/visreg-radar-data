#!/usr/bin/env python3
"""VISReg radar data fetcher (stdlib only).

Runs inside GitHub Actions. Collects (a) every paper that cites VISReg from
Semantic Scholar and OpenAlex, and (b) candidate related papers from arXiv,
Hugging Face papers and OpenAlex text search. Writes data/latest.json.
It never decides what is "new" - the Claude agent does that against its ledger.
Every source reports ok / failed honestly so a failure is never read as "nothing found".
"""
import json, os, sys, time, datetime, urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET

ARXIV_ID = "2606.02572"
DOI = "10.48550/arXiv.2606.02572"
TITLE = "VISReg: Variance-Invariance-Sketching Regularization for JEPA training"
SINCE = "2026-06-01"
MAILTO = os.environ.get("CONTACT_EMAIL", "visreg-radar@users.noreply.github.com")
S2_KEY = os.environ.get("S2_API_KEY", "").strip()
UA = f"visreg-radar/1.0 (mailto:{MAILTO})"

# Discovery queries (Task A). Each is run on arXiv, Hugging Face and OpenAlex.
QUERIES = [
    "VISReg", "variance invariance sketching regularization",
    "JEPA collapse regularization", "JEPA anti-collapse",
    "sliced Wasserstein regularization self-supervised",
    "sketched isotropic Gaussian regularizer", "SIGReg LeJEPA",
    "VICReg alternative covariance regularization", "distribution matching embedding regularizer self-supervised",
    "joint embedding predictive architecture regularization", "representation collapse self-supervised learning regularizer",
    "Cramer-Wold sketch embedding Gaussian", "sliced distribution matching representation learning",
    "latent world model JEPA stable training", "isotropic Gaussian embeddings self-supervised heuristics-free",
    "dimensional collapse variance covariance regularization",
    "Haiyu Wu Balestriero", "Balestriero JEPA", "LeJEPA",
]

status = {}          # source -> {"ok": bool, "detail": str}
def mark(name, ok, detail=""):
    status[name] = {"ok": bool(ok), "detail": detail}

def get(url, headers=None, tries=6, base_sleep=5):
    h = {"User-Agent": UA, "Accept": "application/json"}
    if headers: h.update(headers)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503, 504):
                ra = e.headers.get("Retry-After")
                time.sleep(min(float(ra), 90) if ra and ra.replace('.', '', 1).isdigit() else base_sleep * (2 ** i))
                continue
            raise RuntimeError(last + " " + url)
        except Exception as e:
            last = repr(e)
            time.sleep(base_sleep * (i + 1))
    raise RuntimeError(f"gave up after {tries} tries: {last} {url}")

def jget(url, headers=None):
    return json.loads(get(url, headers))

# ---------------------------------------------------------------- Semantic Scholar
def s2_citations():
    hdr = {"x-api-key": S2_KEY} if S2_KEY else {}
    fields = ("title,year,publicationDate,venue,externalIds,authors,abstract,url,"
              "contexts,intents,isInfluential,publicationVenue")
    out, offset, total_seen = [], 0, 0
    base = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{ARXIV_ID}/citations"
    while True:
        u = f"{base}?fields={urllib.parse.quote(fields)}&limit=100&offset={offset}"
        d = jget(u, hdr)
        for row in d.get("data", []):
            p = row.get("citingPaper") or {}
            out.append({
                "source": "semantic_scholar", "s2_id": p.get("paperId"),
                "title": p.get("title"), "year": p.get("year"), "date": p.get("publicationDate"),
                "venue": p.get("venue"), "external_ids": p.get("externalIds"),
                "authors": [a.get("name") for a in (p.get("authors") or [])][:12],
                "abstract": (p.get("abstract") or "")[:1200], "url": p.get("url"),
                "contexts": row.get("contexts") or [], "intents": row.get("intents") or [],
                "is_influential": row.get("isInfluential"),
            })
        time.sleep(1.2 if S2_KEY else 4)
        if d.get("next") is None:
            break
        offset = d["next"]
    return out

# ---------------------------------------------------------------- OpenAlex
def oa(path, **params):
    params["mailto"] = MAILTO
    return jget("https://api.openalex.org/" + path + "?" + urllib.parse.urlencode(params))

def oa_work(w):
    ids = w.get("ids") or {}
    loc = w.get("primary_location") or {}
    src = (loc.get("source") or {}).get("display_name")
    return {
        "source": "openalex", "openalex_id": w.get("id"), "doi": w.get("doi"),
        "title": w.get("title"), "date": w.get("publication_date"), "year": w.get("publication_year"),
        "venue": src, "type": w.get("type"),
        "external_ids": {"DOI": w.get("doi"), "arXiv": ids.get("arxiv") if isinstance(ids, dict) else None},
        "authors": [(a.get("author") or {}).get("display_name") for a in (w.get("authorships") or [])][:12],
        "url": (loc.get("landing_page_url") or w.get("doi") or w.get("id")),
    }

def oa_visreg_id():
    for key in ("doi:10.48550/arxiv.2606.02572", "doi:" + DOI.lower()):
        try:
            d = jget(f"https://api.openalex.org/works/{key}?mailto={urllib.parse.quote(MAILTO)}")
            if d.get("id"): return d["id"].rsplit("/", 1)[-1], d
        except Exception:
            pass
    d = oa("works", search=TITLE, **{"per-page": 5})
    for w in d.get("results", []):
        if (w.get("title") or "").lower().startswith("visreg"):
            return w["id"].rsplit("/", 1)[-1], w
    raise RuntimeError("VISReg not found in OpenAlex")

def oa_citations(wid):
    out, cursor, count = [], "*", None
    while cursor:
        d = oa("works", filter=f"cites:{wid}", **{"per-page": 100, "cursor": cursor})
        count = (d.get("meta") or {}).get("count", count)
        out += [oa_work(w) for w in d.get("results", [])]
        cursor = (d.get("meta") or {}).get("next_cursor")
        time.sleep(1)
        if not d.get("results"): break
    return out, count

def oa_search(q):
    d = oa("works", search=q, filter=f"from_publication_date:{SINCE}", **{"per-page": 25, "sort": "publication_date:desc"})
    return [oa_work(w) for w in d.get("results", [])]

# ---------------------------------------------------------------- arXiv
NS = {"a": "http://www.w3.org/2005/Atom"}
def arxiv_search(q):
    sq = f'all:"{q}"' if " " in q else f"all:{q}"
    u = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": sq, "start": 0, "max_results": 50, "sortBy": "submittedDate", "sortOrder": "descending"}))
    root = ET.fromstring(get(u, {"Accept": "application/atom+xml"}))
    res = []
    for e in root.findall("a:entry", NS):
        idurl = e.findtext("a:id", "", NS)
        pub = e.findtext("a:published", "", NS)[:10]
        if pub < SINCE:
            continue
        res.append({
            "source": "arxiv", "arxiv_id": idurl.rsplit("/abs/", 1)[-1],
            "title": " ".join((e.findtext("a:title", "", NS) or "").split()),
            "abstract": " ".join((e.findtext("a:summary", "", NS) or "").split())[:1200],
            "authors": [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)][:12],
            "date": pub, "updated": e.findtext("a:updated", "", NS)[:10],
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
            "url": idurl,
        })
    time.sleep(3.5)   # arXiv asks for >= 3 s between calls
    return res

# ---------------------------------------------------------------- Hugging Face
def hf_search(q):
    d = jget("https://huggingface.co/api/papers/search?q=" + urllib.parse.quote(q))
    res = []
    for p in d:
        pub = (p.get("publishedAt") or p.get("paper", {}).get("publishedAt") or "")[:10]
        pp = p.get("paper", p)
        res.append({"source": "huggingface", "arxiv_id": pp.get("id"), "title": pp.get("title"),
                    "date": pub, "url": f"https://huggingface.co/papers/{pp.get('id')}"})
    time.sleep(1)
    return res

# ---------------------------------------------------------------- main
def run_family(name, fn, queries):
    hits, fails = [], 0
    for q in queries:
        try:
            for h in fn(q):
                h["matched_query"] = q
                hits.append(h)
        except Exception as e:
            fails += 1
            print(f"[{name}] query failed: {q}: {e}", file=sys.stderr)
    ok = fails <= len(queries) * 0.2
    mark(name, ok, f"{len(queries) - fails}/{len(queries)} queries ok, {len(hits)} raw hits")
    return hits

def main():
    started = datetime.datetime.now(datetime.timezone.utc)
    out = {"generated_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "since": SINCE,
           "visreg": {"arxiv_id": ARXIV_ID, "doi": DOI, "title": TITLE}}

    # Task B: citation lists
    try:
        c = s2_citations(); out["s2_citations"] = c
        mark("semantic_scholar_citations", True, f"{len(c)} citing papers retrieved (full paging)")
    except Exception as e:
        out["s2_citations"] = []; mark("semantic_scholar_citations", False, str(e)[:300])
    try:
        wid, w = oa_visreg_id(); out["visreg"]["openalex_id"] = wid
        out["visreg"]["openalex_cited_by_count"] = w.get("cited_by_count")
        c, cnt = oa_citations(wid); out["openalex_citations"] = c
        ok = cnt is None or len(c) >= cnt
        mark("openalex_citations", ok, f"{len(c)} retrieved of reported {cnt}")
    except Exception as e:
        out["openalex_citations"] = []; mark("openalex_citations", False, str(e)[:300])

    # Task A: discovery
    out["arxiv_hits"] = run_family("arxiv_search", arxiv_search, QUERIES)
    out["huggingface_hits"] = run_family("huggingface_search", hf_search, QUERIES)
    out["openalex_hits"] = run_family("openalex_search", oa_search, QUERIES)

    out["source_status"] = status
    out["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cit_ok = status["semantic_scholar_citations"]["ok"] or status["openalex_citations"]["ok"]
    disc_ok = sum(status[k]["ok"] for k in ("arxiv_search", "huggingface_search", "openalex_search"))
    out["reliable"] = bool(cit_ok and disc_ok >= 2)
    os.makedirs("data", exist_ok=True)

    def dump(name, obj):
        with open(f"data/{name}", "w") as f:
            json.dump(obj, f, indent=1, ensure_ascii=False)

    # 1) tiny status file - the agent reads this first
    dump("status.json", {
        "generated_utc": out["generated_utc"], "finished_utc": out["finished_utc"],
        "reliable": out["reliable"], "since": SINCE, "visreg": out["visreg"],
        "source_status": status,
        "counts": {"s2_citations": len(out["s2_citations"]), "openalex_citations": len(out["openalex_citations"]),
                   "discovery_unique_papers": 0},
    })
    # 2) citations (Task B)
    dump("citations.json", {"generated_utc": out["generated_utc"],
                            "s2_citations": out["s2_citations"], "openalex_citations": out["openalex_citations"]})
    # 3) discovery (Task A): one row per paper, merged across sources and queries
    merged = {}
    for h in out["arxiv_hits"] + out["huggingface_hits"] + out["openalex_hits"]:
        aid = (h.get("arxiv_id") or "")
        aid = aid.split("v")[0] if aid else ""
        key = aid or h.get("openalex_id") or (h.get("title") or "").lower()
        m = merged.setdefault(key, {"key": key, "title": h.get("title"), "date": h.get("date"),
                                    "authors": h.get("authors"), "abstract": (h.get("abstract") or "")[:600],
                                    "url": h.get("url"), "sources": [], "matched_queries": []})
        if h["source"] not in m["sources"]: m["sources"].append(h["source"])
        if h.get("matched_query") not in m["matched_queries"]: m["matched_queries"].append(h.get("matched_query"))
        if not m["abstract"] and h.get("abstract"): m["abstract"] = h["abstract"][:600]
        if not m["authors"] and h.get("authors"): m["authors"] = h["authors"]
    disc = sorted(merged.values(), key=lambda x: x.get("date") or "", reverse=True)
    dump("discovery.json", {"generated_utc": out["generated_utc"], "papers": disc})
    st = json.load(open("data/status.json")); st["counts"]["discovery_unique_papers"] = len(disc)
    dump("status.json", st)
    print(json.dumps(status, indent=1))
    print("reliable:", out["reliable"], "| discovery papers:", len(disc))

if __name__ == "__main__":
    main()
