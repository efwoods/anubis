"""Turn a pile of page visits into something a model can actually read.

A month of browsing is tens of thousands of rows, and a model reading the raw
rows spends its attention on repetition: forty visits to the same inbox say
one thing, not forty. So the visits are folded into a digest — what the person
visits most, what the person typed into search boxes, which pages the person
opened, when in the day the person browses, and which websites are new in this
period — and the digest is what the analysis reads.

Everything here is a pure function over the rows the daemon returned, so the
shape of the digest is unit-tested without a model, a machine, or a browser.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

# How much of each kind of evidence reaches the model. Searches are the
# person's own words and carry the most meaning per character, so more of
# those survive than of anything else.
TOP_SITE_COUNT = 40
SEARCH_COUNT = 120
TITLE_COUNT = 120
ADDRESS_COUNT = 80
NEW_SITE_COUNT = 25

# Hosts whose visits say nothing about the person: pages the person's own
# software opens on the person's behalf, and the endless background chatter of
# a signed-in session.
MACHINERY_HOST_MARKERS = (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "accounts.google.com",
    "login.microsoftonline.com",
    "auth0.com",
    "okta.com",
    "duckduckgo.com/ac",
    "safebrowsing",
    "doubleclick.net",
    "googletagmanager.com",
    "google-analytics.com",
    "gstatic.com",
    "googleapis.com",
    "cloudfront.net",
    "cdn.",
)


def _visit_moment(visit: Mapping[str, Any]) -> datetime | None:
    """Return the moment of one visit, or ``None`` when the row carries no time."""
    text = str(visit.get("visited_at") or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_machinery(visit: Mapping[str, Any]) -> bool:
    """Whether a visit was made by the person's software rather than by the person."""
    host = str(visit.get("host") or "").lower()
    if not host:
        return True
    return any(marker in host for marker in MACHINERY_HOST_MARKERS)


def meaningful_visits(visits: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the visits a person actually chose to make, oldest first."""
    kept = [dict(visit) for visit in visits if not is_machinery(visit)]
    kept.sort(key=lambda visit: str(visit.get("visited_at") or ""))
    return kept


def _address_worth_showing(visit: Mapping[str, Any]) -> str:
    """Return the web address of a visit, when the address names something specific.

    A bare home page adds nothing the host name has not already said; an
    address with a path names the exact repository, product, article, or
    document the person opened, which is the part worth reading.
    """
    address = str(visit.get("url") or "")
    try:
        parsed = urlparse(address)
    except ValueError:
        return ""
    path = (parsed.path or "").strip("/")
    if not path or len(path) < 2:
        return ""
    return address


def summarize_visits(visits: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold visits into the counts and samples the digest is written from."""
    kept = meaningful_visits(visits)
    hosts = Counter(str(visit.get("host") or "") for visit in kept)
    searches: list[str] = []
    titles: list[str] = []
    addresses: list[str] = []
    hours: Counter[int] = Counter()
    weekdays: Counter[str] = Counter()
    days: set[str] = set()
    browsers: Counter[str] = Counter()
    first_seen: dict[str, str] = {}
    for visit in kept:
        search = str(visit.get("search_terms") or "").strip()
        if search:
            searches.append(search)
        title = str(visit.get("title") or "").strip()
        if title:
            titles.append(title)
        address = _address_worth_showing(visit)
        if address:
            addresses.append(address)
        browsers[str(visit.get("browser_name") or "")] += 1
        moment = _visit_moment(visit)
        if moment is not None:
            hours[moment.hour] += 1
            weekdays[moment.strftime("%A")] += 1
            days.add(moment.date().isoformat())
        host = str(visit.get("host") or "")
        if host and host not in first_seen:
            first_seen[host] = str(visit.get("visited_at") or "")
    return {
        "visit_count": len(kept),
        "distinct_hosts": len(hosts),
        "days_covered": len(days),
        "top_hosts": hosts.most_common(TOP_SITE_COUNT),
        "searches": _most_recent_unique(searches, SEARCH_COUNT),
        "titles": _most_recent_unique(titles, TITLE_COUNT),
        "addresses": _most_recent_unique(addresses, ADDRESS_COUNT),
        "hours": dict(sorted(hours.items())),
        "weekdays": dict(weekdays),
        "browsers": dict(browsers),
        "first_seen": first_seen,
        "period_start": kept[0].get("visited_at") if kept else "",
        "period_end": kept[-1].get("visited_at") if kept else "",
    }


def _most_recent_unique(values: list[str], keep: int) -> list[str]:
    """Return the last ``keep`` distinct values, in the order they last occurred.

    Recent evidence beats old evidence when a limit bites: what the person
    searched for this morning bears on who the person is now more than what
    the person searched for three weeks ago.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for value in reversed(values):
        folded = value.strip()
        if not folded or folded.lower() in seen:
            continue
        seen.add(folded.lower())
        kept.append(folded)
        if len(kept) >= keep:
            break
    kept.reverse()
    return kept


def _hour_band(hours: Mapping[int, int]) -> str:
    """Describe when in the day the person browses, in one sentence."""
    if not hours:
        return "unknown"
    bands = defaultdict(int)
    for hour, count in hours.items():
        if 5 <= hour < 12:
            bands["morning"] += count
        elif 12 <= hour < 17:
            bands["afternoon"] += count
        elif 17 <= hour < 22:
            bands["evening"] += count
        else:
            bands["late night"] += count
    total = sum(bands.values()) or 1
    ordered = sorted(bands.items(), key=lambda entry: entry[1], reverse=True)
    return ", ".join(
        f"{name} {round(100 * count / total)}%" for name, count in ordered if count
    )


def render_digest(
    visits: Iterable[Mapping[str, Any]],
    *,
    known_hosts: Iterable[str] = (),
    max_characters: int = 24000,
) -> str:
    """Write the digest the analysis reads.

    ``known_hosts`` are the websites earlier passes already saw, so this pass
    can say which websites are NEW — a new website in a person's life is the
    single most informative row in a period.
    """
    summary = summarize_visits(visits)
    if not summary["visit_count"]:
        return ""
    already_known = {str(host).lower() for host in known_hosts}
    new_hosts = [
        host
        for host, _count in summary["top_hosts"]
        if host and host.lower() not in already_known
    ][:NEW_SITE_COUNT]
    sections: list[str] = [
        "=== BROWSING RECORD ===",
        (
            f"{summary['visit_count']} page visits across {summary['distinct_hosts']} "
            f"websites over {summary['days_covered']} days, from "
            f"{summary['period_start']} to {summary['period_end']}."
        ),
        f"Browsers used: {', '.join(name for name in summary['browsers'] if name) or 'unknown'}.",
        f"When the browsing happened: {_hour_band(summary['hours'])}.",
    ]
    if summary["weekdays"]:
        busiest = sorted(
            summary["weekdays"].items(), key=lambda entry: entry[1], reverse=True
        )[:3]
        sections.append(
            "Busiest days: " + ", ".join(f"{day} ({count})" for day, count in busiest) + "."
        )
    sections.append("")
    sections.append("--- Websites visited most ---")
    sections.extend(
        f"{host} — {count} visits" for host, count in summary["top_hosts"] if host
    )
    if new_hosts:
        sections.append("")
        sections.append("--- Websites new in this period ---")
        sections.append(", ".join(new_hosts))
    if summary["searches"]:
        sections.append("")
        sections.append("--- Words typed into search boxes (the person's own words) ---")
        sections.extend(f'"{search}"' for search in summary["searches"])
    if summary["titles"]:
        sections.append("")
        sections.append("--- Pages opened, by title ---")
        sections.extend(summary["titles"])
    if summary["addresses"]:
        sections.append("")
        sections.append("--- Specific addresses opened ---")
        sections.extend(summary["addresses"])
    digest = "\n".join(sections)
    if max_characters and len(digest) > max_characters:
        # Trim from the end: the head carries the counts and the searches,
        # which are the parts the analysis leans on hardest.
        digest = digest[:max_characters].rsplit("\n", 1)[0] + "\n…"
    return digest


def hosts_of(visits: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every distinct website in a batch of visits, for the next pass to compare against."""
    return sorted(
        {
            str(visit.get("host") or "").lower()
            for visit in visits
            if str(visit.get("host") or "").strip()
        }
    )
