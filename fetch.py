#!/usr/bin/env python3
"""
fetch_github.py - GitHub watcher for the VISReg Adversarial Radar.

Runs inside the GitHub Action (next to fetch.py) and writes data/github.json.
It looks for:
  1. new forks of HaiyuWu/visreg (and their latest commit messages),
  2. new repositories matching adversarial / VISReg-family searches,
  3. recent issues and pull requests on HaiyuWu/visreg,
  4. (best effort) code-search hits for "VISReg" + adversarial terms.
Every item is flagged with adv_keyword = True when it mentions adversarial-type words.

Design rules:
  - NEVER fail the workflow: all errors are caught and written to source_status.
  - Standard library only (urllib), so no pip install is needed.
  - Uses GITHUB_TOKEN from the environment (provided by Actions) for rate limits.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

OWNER_REPO = "HaiyuWu/visreg"
SINCE = "2026-06-01"
OUT_PATH = os.path.join("data", "github.json")
API = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN", "")

ADV_RE = re.compile(
    r"adversar|robust|attack|perturb|defen[cs]e|certif|pgd|fgsm|autoattack|min-?max",
    re.IGNORECASE,
)

REPO_QUERIES = [
    "VISReg adversarial",
    "VISReg robustness",
    "VISReg attack",
    "adversarial JEPA",
    "adversarial training JEPA",
    "adversarial robustness JEPA self-supervised",
    "SIGReg adversarial",
    "LeJEPA adversarial",
    "sliced Wasserstein regularization adversarial self-supervised",
    "isotropic Gaussian embedding adversarial robustness",
    "VISReg",
]

CODE_QUERIES = [
    "VISReg adversarial",
    "visreg pgd",
    "visreg attack",
]


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def api_get(path, params=None):
    """GET from the GitHub API; returns parsed JSON. Raises on HTTP errors."""
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "visreg-radar-bot",
    }
    if TOKEN:
        headers["Authorization"] = "Bearer " + TOKEN
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read().decode("utf-8"))


def adv(*texts):
    return any(t and ADV_RE.search(t) for t in texts)


def safe_run(name, fn, status, bucket):
    """Run one collector; record ok/detail; never raise."""
    try:
        items, detail = fn()
        bucket.extend(items)
        status[name] = {"ok": True, "detail": detail}
    except urllib.error.HTTPError as e:
        status[name] = {"ok": False, "detail": "HTTP %s %s" % (e.code, e.reason)}
    except Exception as e:  # noqa: BLE001 - must never fail the workflow
        status[name] = {"ok": False, "detail": "%s: %s" % (type(e).__name__, e)}


def collect_forks():
    items = []
    for page in (1, 2, 3):
        batch = api_get(
            "/repos/%s/forks" % OWNER_REPO,
            {"sort": "newest", "per_page": 100, "page": page},
        )
        if not batch:
            break
        for f in batch:
            items.append(
                {
                    "full_name": f.get("full_name"),
                    "url": f.get("html_url"),
                    "description": f.get("description"),
                    "created_at": f.get("created_at"),
                    "pushed_at": f.get("pushed_at"),
                    "adv_keyword": adv(f.get("full_name"), f.get("description")),
                    "recent_commits": [],
                }
            )
        if len(batch) < 100:
            break
        time.sleep(1)
    # Latest commit messages of the most recently pushed forks (that is where
    # adversarial-training code usually first appears).
    pushed = [i for i in items if i["pushed_at"] and i["pushed_at"] > i["created_at"]]
    pushed.sort(key=lambda i: i["pushed_at"], reverse=True)
    for it in pushed[:30]:
        try:
            commits = api_get(
                "/repos/%s/commits" % it["full_name"], {"per_page": 5}
            )
            msgs = [
                (c.get("commit", {}).get("message") or "").split("\n")[0][:200]
                for c in commits
            ]
            it["recent_commits"] = msgs
            if any(adv(m) for m in msgs):
                it["adv_keyword"] = True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return items, "%d forks listed, %d with commits checked" % (
        len(items),
        min(len(pushed), 30),
    )


def collect_repos():
    seen = {}
    failures = 0
    for q in REPO_QUERIES:
        try:
            res = api_get(
                "/search/repositories",
                {
                    "q": "%s created:>=%s" % (q, SINCE),
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 30,
                },
            )
            for r in res.get("items", []):
                key = r.get("full_name")
                if key in seen:
                    seen[key]["matched_queries"].append(q)
                    continue
                seen[key] = {
                    "full_name": key,
                    "url": r.get("html_url"),
                    "description": r.get("description"),
                    "created_at": r.get("created_at"),
                    "pushed_at": r.get("pushed_at"),
                    "topics": r.get("topics", []),
                    "adv_keyword": adv(key, r.get("description"), " ".join(r.get("topics", []))),
                    "matched_queries": [q],
                }
        except Exception:  # noqa: BLE001
            failures += 1
        time.sleep(2.5)  # search API: 30 requests/minute
    if failures > len(REPO_QUERIES) // 2:
        raise RuntimeError("%d of %d repo searches failed" % (failures, len(REPO_QUERIES)))
    return list(seen.values()), "%d queries, %d failed, %d unique repos" % (
        len(REPO_QUERIES),
        failures,
        len(seen),
    )


def collect_issues():
    items = []
    for kind in ("issues",):  # the issues endpoint also returns pull requests
        batch = api_get(
            "/repos/%s/%s" % (OWNER_REPO, kind),
            {"state": "all", "sort": "created", "direction": "desc", "per_page": 50},
        )
        for i in batch:
            items.append(
                {
                    "number": i.get("number"),
                    "title": i.get("title"),
                    "url": i.get("html_url"),
                    "is_pull_request": "pull_request" in i,
                    "state": i.get("state"),
                    "created_at": i.get("created_at"),
                    "adv_keyword": adv(i.get("title"), (i.get("body") or "")[:1000]),
                }
            )
    return items, "%d recent issues/PRs" % len(items)


def collect_code():
    seen = {}
    for q in CODE_QUERIES:
        res = api_get("/search/code", {"q": q, "per_page": 20})
        for c in res.get("items", []):
            repo = c.get("repository", {}).get("full_name")
            key = "%s:%s" % (repo, c.get("path"))
            if key in seen:
                continue
            seen[key] = {
                "repo": repo,
                "path": c.get("path"),
                "url": c.get("html_url"),
                "matched_query": q,
                "adv_keyword": adv(c.get("path"), repo),
            }
        time.sleep(2.5)
    return list(seen.values()), "%d queries, %d unique code hits" % (
        len(CODE_QUERIES),
        len(seen),
    )


def main():
    started = now_utc()
    status = {}
    out = {"forks": [], "repos": [], "issues": [], "code": []}

    safe_run("forks", collect_forks, status, out["forks"])
    safe_run("repo_search", collect_repos, status, out["repos"])
    safe_run("issues", collect_issues, status, out["issues"])
    safe_run("code_search", collect_code, status, out["code"])

    ok_count = sum(1 for s in status.values() if s["ok"])
    result = {
        "generated_utc": started,
        "finished_utc": now_utc(),
        "reliable": ok_count >= 2,
        "since": SINCE,
        "watched_repo": OWNER_REPO,
        "source_status": status,
        "counts": {k: len(v) for k, v in out.items()},
        "adv_keyword_counts": {
            k: sum(1 for x in v if x.get("adv_keyword")) for k, v in out.items()
        },
        **out,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print("github.json written:", result["counts"], result["adv_keyword_counts"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - never fail the workflow
        print("fetch_github.py failed softly:", exc)
        sys.exit(0)
