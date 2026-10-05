import re, markdown, asyncio, base64, os
from playwright.sync_api import sync_playwright
t = open("manuscript/manuscript_full.md").read()
t = re.sub(r"^TITLE: (.*)$", r"<h1 class='title'>\1</h1>", t, flags=re.M); t = re.sub(r"^CENTER: (.*)$", r"<p class='c'>\1</p>", t, flags=re.M); t = t.replace("PAGEBREAK", "<div style='page-break-after:always'></div>")
def img(m):
    cap, path = m.group(1), m.group(2)
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f"<figure><img src='data:image/png;base64,{b}'><figcaption>{cap}</figcaption></figure>"
t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", img, t)
html = markdown.markdown(t, extensions=["tables"])
css = "body{font-family:'Times New Roman',serif;font-size:10.5pt;line-height:1.35;max-width:17cm;margin:auto}h1{font-size:15pt;margin-top:1.4em}h1.title{text-align:center;font-size:17pt}h2{font-size:12.5pt}p.c{text-align:center}table{border-collapse:collapse;font-size:8pt;margin:8px auto;width:100%}td,th{border:1px solid #999;padding:2px 4px}th{background:#e6e5e1}figure{margin:10px 0;text-align:center}figure img{max-width:100%}figcaption{font-size:9pt;text-align:left}"
open("manuscript/manuscript.html", "w").write(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>")
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None)
    pg = b.new_page(); pg.goto("file://" + os.path.abspath("manuscript/manuscript.html")); pg.pdf(path="manuscript/manuscript.pdf", format="A4", margin=dict(top="2cm", bottom="2cm", left="2cm", right="2cm")); b.close()
print("pdf ok")
