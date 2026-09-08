"""Website audit analysis, vendor recipe parsing, and connected-site text extraction."""

from __future__ import annotations

from src.anubis.utils.connected_accounts import recipes, website_tools
from src.anubis.utils.connected_accounts.browser_session_tools import html_to_text


def test_analyze_page_reports_search_and_accessibility_issues():
    html = """<html><head><title>Neural Nexus — talk to your own avatar and to the people you admire, any time</title>
    <meta name="description" content="Short"></head><body><h1>One</h1><h1>Two</h1>
    <img src="a.png"><img src="b.png" alt="ok"><form><input type="text"></form>
    <p>""" + ("word " * 100) + "</p></body></html>"
    analysis = website_tools.analyze_page("https://x.test/", html)
    issues = set(analysis["issues"])
    assert "2 <h1> headings" in issues
    assert "1 images without alt text" in issues
    assert "no canonical link" in issues
    assert "no lang attribute on <html>" in issues
    assert "1 form inputs without a label" in issues
    assert any(issue.startswith("title longer") for issue in issues)
    assert "thin content" not in " ".join(issues)


def test_links_are_absolute_and_hashes_ignore_markup():
    links = website_tools.extract_links('<a href="/a">a</a><a href="mailto:x@y">m</a><a href="https://o.test/b#frag">b</a>', "https://x.test/p/")
    assert links == ["https://x.test/a", "https://o.test/b"]
    assert website_tools.content_hash("<p>Hello <b>world</b></p>") == website_tools.content_hash("<div>hello world</div>")


def test_recipe_parsers_and_urls():
    openai_rows = recipes.parse_openai_costs(
        {"data": [{"start_time": 1725580800, "results": [{"amount": {"value": 1.5}}, {"amount": {"value": 2}}]}]}, {}
    )
    assert openai_rows == [{"day": "2024-09-06", "metric": "cost", "value": 3.5, "unit": "usd"}]
    anthropic_rows = recipes.parse_anthropic_usage({"data": [{"starting_at": "2026-09-01T00:00:00Z", "cost_usd": 4.25, "input_tokens": 100}]}, {})
    assert {"day": "2026-09-01", "metric": "cost", "value": 4.25, "unit": "usd"} in anthropic_rows
    langsmith_rows = recipes.parse_langsmith_usage({"data": [{"date": "2026-09-02", "trace_count": 12}]}, {})
    assert langsmith_rows == [{"day": "2026-09-02", "metric": "traces", "value": 12.0, "unit": "count"}]
    text_rows = recipes.parse_text_numbers("Total $1,234.56 this month and $9", {})
    assert [row["value"] for row in text_rows] == [1234.56, 9.0]
    url = recipes.render_url(recipes.RECIPES["openai"]["costs"], "7d")
    assert "bucket_width=1d" in url and "start_time=" in url
    assert set(recipes.recipes_for("langsmith")) == {"usage", "usage_page"}
    assert recipes.recipes_for("unknown") == {}


def test_html_to_text_drops_scripts_and_keeps_lines():
    text = html_to_text("<html><script>var x=1</script><style>a{}</style><h1>Title</h1><p>Body text</p></html>")
    assert "var x" not in text and "Title" in text and "Body text" in text
