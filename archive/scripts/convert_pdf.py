import re
import os
import subprocess
import markdown

md_path = r"C:\Users\felix\.gemini\antigravity\brain\aee07002-d331-41c2-9452-94458b8d4ed8\institutional_report.md"
html_path = r"C:\Users\felix\.gemini\antigravity\scratch\mnq_liquidity_feature_engine\report.html"
pdf_path = r"C:\Users\felix\.gemini\antigravity\brain\aee07002-d331-41c2-9452-94458b8d4ed8\institutional_report.pdf"

def main():
    if not os.path.exists(md_path):
        print(f"Error: {md_path} does not exist.")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Convert markdown to HTML first
    html_body = markdown.markdown(content, extensions=['tables', 'fenced_code'])
    
    # 1. Replace the mermaid pre/code block with a styled HTML flowchart
    flowchart_html = """
    <div class="flowchart">
        <div class="flow-step">
            <span class="step-num">1</span>
            <div class="step-content"><strong>Level Selection</strong><p>Identify HTF Liquidity Level</p></div>
        </div>
        <div class="flow-arrow">&darr;</div>
        <div class="flow-step">
            <span class="step-num">2</span>
            <div class="step-content"><strong>The Sweep</strong><p>Price wicks past level</p></div>
        </div>
        <div class="flow-arrow">&darr;</div>
        <div class="flow-step">
            <span class="step-num">3</span>
            <div class="step-content"><strong>Confirmation</strong><p>Displacement & FVG Rejection</p></div>
        </div>
        <div class="flow-arrow">&darr;</div>
        <div class="flow-step">
            <span class="step-num">4</span>
            <div class="step-content"><strong>Risk Assessment</strong><p>Check Risk Cap (R &le; 60 pts)</p></div>
        </div>
        <div class="flow-arrow">&darr;</div>
        <div class="flow-step execution">
            <span class="step-num">5</span>
            <div class="step-content"><strong>Execution</strong><p>Enter Trade (Split TP at 2.0R)</p></div>
        </div>
    </div>
    """
    mermaid_pattern = r'<pre><code class="language-mermaid">[\s\S]*?</code></pre>'
    html_body = re.sub(mermaid_pattern, flowchart_html, html_body)
    
    # 2. Replace LaTeX math formula with styled HTML equation
    math_html = """
    <div class="equation">
        <span class="eq-label">Portfolio Sharpe</span> = 
        <div class="fraction">
            <span class="numerator">0.5(1.0) + 0.5(0.66)</span>
            <span class="denominator">&radic;(0.5<sup>2</sup> + 0.5<sup>2</sup>)</span>
        </div>
        &nbsp;=&nbsp;
        <div class="fraction">
            <span class="numerator">0.83</span>
            <span class="denominator">0.707</span>
        </div>
        &nbsp;=&nbsp;
        <strong>1.17</strong>
    </div>
    """
    latex_pattern = r'<p>\$\$\\text\{Sharpe\}_\{\\text\{portfolio\}\}[\s\S]*?\$\$</p>'
    html_body = re.sub(latex_pattern, math_html, html_body)
    
    # 3. Replace inline LaTeX variables (e.g. $R$ and $\rho$)
    html_body = html_body.replace("$R$", "<em>R</em>")
    html_body = html_body.replace("$\\rho \\approx 0.0$", "<em>&rho;</em> &approx; 0.0")
    html_body = html_body.replace("$\\beta$", "<em>&beta;</em>")
    html_body = html_body.replace("$\\alpha$", "<em>&alpha;</em>")
    html_body = html_body.replace("Return = (\\beta \\times \\text{Market Return}) + \\alpha", "Return = (<em>&beta;</em> &times; Market Return) + <em>&alpha;</em>")
    
    # Premium corporate layout CSS
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Institutional Research Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        @page {{
            size: A4;
            margin: 2.5cm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Inter', sans-serif;
                font-size: 9pt;
                color: #64748b;
            }}
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            font-size: 10.5pt;
            line-height: 1.6;
            color: #1e293b;
            margin: 0;
            padding: 0;
        }}
        
        h1 {{
            font-size: 20pt;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            line-height: 1.2;
        }}
        
        hr {{
            border: 0;
            height: 2px;
            background: #cbd5e1;
            margin: 20px 0 30px 0;
        }}
        
        h2 {{
            font-size: 13pt;
            font-weight: 600;
            color: #1e3a8a;
            margin-top: 35px;
            margin-bottom: 15px;
            border-left: 4px solid #2563eb;
            padding-left: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        h3 {{
            font-size: 11pt;
            font-weight: 600;
            color: #0f172a;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        
        p {{
            margin-top: 0;
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        ul {{
            margin-top: 0;
            margin-bottom: 20px;
            padding-left: 20px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        /* Premium Table Styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 9.5pt;
        }}
        
        th {{
            background-color: #f8fafc;
            color: #334155;
            font-weight: 600;
            text-align: center;
            padding: 10px 12px;
            border-top: 1px solid #e2e8f0;
            border-bottom: 2px solid #cbd5e1;
        }}
        
        td {{
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
            text-align: center;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        
        strong {{
            font-weight: 600;
            color: #0f172a;
        }}
        
        /* Flowchart Styling */
        .flowchart {{
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 25px 0;
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
        
        .flow-step {{
            display: flex;
            align-items: center;
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 10px 15px;
            width: 80%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        
        .flow-step.execution {{
            border-color: #2563eb;
            background: #eff6ff;
        }}
        
        .step-num {{
            background: #2563eb;
            color: white;
            font-weight: 700;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-size: 9pt;
        }}
        
        .step-content {{
            flex-grow: 1;
        }}
        
        .step-content strong {{
            display: block;
            font-size: 10pt;
            color: #1e293b;
        }}
        
        .step-content p {{
            margin: 2px 0 0 0;
            font-size: 8.5pt;
            color: #64748b;
            text-align: left;
        }}
        
        .flow-arrow {{
            font-size: 14pt;
            color: #94a3b8;
            margin: 6px 0;
            font-weight: bold;
        }}
        
        /* Equation Layout */
        .equation {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11pt;
            margin: 25px 0;
            color: #0f172a;
        }}
        
        .eq-label {{
            font-weight: 600;
            margin-right: 8px;
        }}
        
        .fraction {{
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            vertical-align: middle;
            font-size: 9.5pt;
            margin: 0 4px;
        }}
        
        .numerator {{
            border-bottom: 1px solid #1e293b;
            padding: 0 4px 2px 4px;
            text-align: center;
        }}
        
        .denominator {{
            padding: 2px 4px 0 4px;
            text-align: center;
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>
"""
    
    # Save the HTML report
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"Generated HTML report at {html_path}")
    
    # Compile to PDF using Edge in headless mode
    edge_executable = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    cmd = [
        edge_executable,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    
    print("Compiling HTML to PDF using headless Edge...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Successfully generated PDF report at {pdf_path}")
    else:
        print(f"Error compiling to PDF: {res.stderr}")

if __name__ == "__main__":
    main()
