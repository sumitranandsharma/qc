"""
Generates an elegant, publication-grade HTML report from report/capstone_report.md
for offline viewing, presentation, and PDF export.
"""

import re
import html
from pathlib import Path

def markdown_to_html(md_text: str) -> str:
    # Convert code blocks
    def replace_code_block(match):
        lang = match.group(1) or ""
        code = html.escape(match.group(2).strip())
        return f'<pre><code class="language-{lang}">{code}</code></pre>'

    md_text = re.sub(r'```(\w+)?\n([\s\S]*?)```', replace_code_block, md_text)

    # Convert tables
    def replace_table(match):
        table_str = match.group(0).strip()
        lines = [l.strip() for l in table_str.split("\n") if l.strip()]
        if len(lines) < 2:
            return table_str

        # Header
        headers = [c.strip() for c in lines[0].strip("|").split("|")]
        # Separator line is lines[1]
        rows = []
        for line in lines[2:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append(cells)

        header_html = "".join(f"<th>{c}</th>" for c in headers)
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in rows
        )
        return f'<div class="table-container"><table><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>'

    table_pattern = re.compile(r'(\|[^\n]+\|\n\|(?:\s*:?-+:?\s*\|)+\n(?:\|[^\n]+\|\n?)+)')
    md_text = table_pattern.sub(replace_table, md_text)

    # Convert Headings
    md_text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', md_text, flags=re.MULTILINE)

    # Convert horizontal rules
    md_text = re.sub(r'^---$', r'<hr class="divider">', md_text, flags=re.MULTILINE)

    # Convert inline code
    md_text = re.sub(r'`([^`]+)`', r'<code>\1</code>', md_text)

    # Convert bold and italic
    md_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', md_text)
    md_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', md_text)

    # Convert lists
    def replace_unordered_list(match):
        items = match.group(0).strip().split('\n')
        bullet_re = re.compile(r"^[-*]\s+")
        list_html = ''.join('<li>' + bullet_re.sub('', item) + '</li>' for item in items if item.strip())
        return f'<ul>{list_html}</ul>'

    md_text = re.sub(r'(?:^[-*]\s+.*$\n?)+', replace_unordered_list, md_text, flags=re.MULTILINE)

    # Numbered lists
    def replace_ordered_list(match):
        items = match.group(0).strip().split('\n')
        num_re = re.compile(r"^\d+\.\s+")
        list_html = ''.join('<li>' + num_re.sub('', item) + '</li>' for item in items if item.strip())
        return f'<ol>{list_html}</ol>'

    md_text = re.sub(r'(?:^\d+\.\s+.*$\n?)+', replace_ordered_list, md_text, flags=re.MULTILINE)

    # Paragraphs (blocks not enclosed in tags)
    paragraphs = md_text.split('\n\n')
    formatted_paras = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith(('<h1', '<h2', '<h3', '<pre', '<div', '<hr', '<ul', '<ol', '<table')):
            formatted_paras.append(p)
        else:
            # Replace single newlines with spaces or <br>
            p_clean = p.replace('\n', '<br>')
            formatted_paras.append(f'<p>{p_clean}</p>')

    return '\n\n'.join(formatted_paras)

def build_full_html():
    md_path = Path("C:/SAH/report/capstone_report.md")
    html_out = Path("C:/SAH/report/capstone_report.html")

    if not md_path.exists():
        print(f"Error: {md_path} not found.")
        return

    content_md = md_path.read_text(encoding="utf-8")
    body_html = markdown_to_html(content_md)

    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Quantum AI Medical Imaging System - Capstone Project Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary: #4F46E5;
      --primary-light: #EEF2FF;
      --primary-dark: #3730A3;
      --text-main: #0F172A;
      --text-muted: #64748B;
      --border-color: #E2E8F0;
      --bg-page: #F8FAFC;
      --bg-card: #FFFFFF;
      --success: #059669;
      --warning: #D97706;
      --danger: #DC2626;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.65;
      color: var(--text-main);
      background-color: var(--bg-page);
      padding: 40px 20px;
    }}

    .container {{
      max-width: 980px;
      margin: 0 auto;
      background: var(--bg-card);
      padding: 56px 64px;
      border-radius: 16px;
      box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
      border: 1px solid var(--border-color);
    }}

    /* Header styling */
    h1 {{
      font-size: 26px;
      font-weight: 700;
      color: #1E1B4B;
      letter-spacing: -0.02em;
      margin-bottom: 12px;
      line-height: 1.3;
    }}

    h2 {{
      font-size: 19px;
      font-weight: 700;
      color: #312E81;
      margin-top: 36px;
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    h3 {{
      font-size: 15px;
      font-weight: 600;
      color: #4338CA;
      margin-top: 24px;
      margin-bottom: 10px;
    }}

    p {{
      margin-bottom: 16px;
      font-size: 14.5px;
      color: #334155;
    }}

    strong {{
      font-weight: 600;
      color: #0F172A;
    }}

    /* Divider */
    .divider {{
      border: 0;
      height: 1px;
      background: var(--border-color);
      margin: 28px 0;
    }}

    /* Lists */
    ul, ol {{
      margin-left: 24px;
      margin-bottom: 18px;
      color: #334155;
      font-size: 14.5px;
    }}

    li {{
      margin-bottom: 6px;
      line-height: 1.6;
    }}

    /* Tables */
    .table-container {{
      overflow-x: auto;
      margin: 20px 0 24px 0;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
      text-align: left;
    }}

    thead th {{
      background: #F1F5F9;
      color: #1E293B;
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-color);
    }}

    tbody td {{
      padding: 10px 14px;
      border-bottom: 1px solid #F1F5F9;
      color: #334155;
    }}

    tbody tr:last-child td {{
      border-bottom: none;
    }}

    tbody tr:nth-child(even) {{
      background-color: #F8FAFC;
    }}

    /* Code & Pre */
    code {{
      font-family: 'JetBrains Mono', Consolas, monospace;
      font-size: 12.5px;
      background: #F1F5F9;
      color: #4338CA;
      padding: 2px 6px;
      border-radius: 4px;
      border: 1px solid #E2E8F0;
    }}

    pre {{
      background: #0F172A;
      color: #F8FAFC;
      padding: 18px 20px;
      border-radius: 10px;
      overflow-x: auto;
      margin: 18px 0 24px 0;
      font-size: 12.5px;
      line-height: 1.5;
      border: 1px solid #1E293B;
    }}

    pre code {{
      background: transparent;
      color: inherit;
      padding: 0;
      border: none;
      font-size: inherit;
    }}

    /* Metadata Box */
    .meta-box {{
      background: #F8FAFC;
      border-left: 4px solid var(--primary);
      padding: 16px 20px;
      margin: 20px 0 28px 0;
      border-radius: 0 8px 8px 0;
      font-size: 14px;
      color: #475569;
    }}

    .meta-box p {{
      margin-bottom: 4px;
    }}

    .meta-box p:last-child {{
      margin-bottom: 0;
    }}

    /* Printable styles */
    @media print {{
      body {{ background: #fff; padding: 0; }}
      .container {{ border: none; box-shadow: none; padding: 0; max-width: 100%; }}
      h1 {{ font-size: 22px; }}
      h2 {{ font-size: 16px; margin-top: 24px; }}
      pre {{ white-space: pre-wrap; word-break: break-all; }}
      .table-container {{ box-shadow: none; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    {body_html}
  </div>
</body>
</html>
"""

    html_out.write_text(template, encoding="utf-8")
    print(f"[Report] Successfully converted {md_path} -> {html_out} ({len(template)} bytes)")

if __name__ == "__main__":
    build_full_html()
