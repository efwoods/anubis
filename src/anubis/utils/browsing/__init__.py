"""Learning who the owner is from what the owner reads.

The owner's web browsing is the largest honest record of a person most people
ever produce, and every machine running the Neural Nexus connector can hand it
over the same way, whatever platform the machine is. This package turns that
record into two things the avatar carries permanently — facts about the owner,
and scored traits of how the owner thinks and works — plus a report the owner
can read.

The modules, in the order a pass uses them:

* ``history_client`` — asks each machine what is new, then reads only that,
  keeping a watermark per machine so nothing is ever read or paid for twice.
* ``digest`` — folds tens of thousands of visits into the evidence worth
  reading: the most-visited websites, the words typed into search boxes, the
  page titles, the specific addresses, and when in the day the browsing
  happened.
* ``insights`` — one structured-output call that reads the digest into facts
  and traits, and the writes that put each of those where the avatar reads
  them every turn.
* ``sweeper`` — the continuous loop: cheap count, threshold, analyse, advance.
* ``tools`` — the same pass, on demand, when the owner asks.
"""

from src.anubis.utils.browsing.digest import render_digest, summarize_visits
from src.anubis.utils.browsing.history_client import (
    new_visit_count,
    read_new_visits,
    read_watermark,
    watermark_namespace,
    write_watermark,
)
from src.anubis.utils.browsing.insights import (
    BROWSING_DIMENSION,
    BROWSING_REPORT_KIND,
    BrowsingInsights,
    analyze_visits,
    apply_insights,
)
from src.anubis.utils.browsing.sweeper import (
    analyse_account,
    analyse_machine,
    run_browsing_sweep_once,
    run_browsing_sweeper,
)
from src.anubis.utils.browsing.tools import build_browsing_insight_tools

__all__ = [
    "BROWSING_DIMENSION",
    "BROWSING_REPORT_KIND",
    "BrowsingInsights",
    "analyse_account",
    "analyse_machine",
    "analyze_visits",
    "apply_insights",
    "build_browsing_insight_tools",
    "new_visit_count",
    "read_new_visits",
    "read_watermark",
    "render_digest",
    "run_browsing_sweep_once",
    "run_browsing_sweeper",
    "summarize_visits",
    "watermark_namespace",
    "write_watermark",
]
