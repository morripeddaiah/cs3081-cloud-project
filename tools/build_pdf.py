import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

md_path = r'C:/Users/pedda/OneDrive/Desktop/cloud_assignment/project/report/report.md'
pdf_path = r'C:/Users/pedda/OneDrive/Desktop/cloud_assignment/project/report/report.pdf'

with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=15, leading=19, textColor=colors.HexColor('#1a365d'), spaceAfter=4)
subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10.5, leading=14, textColor=colors.HexColor('#2b6cb0'), spaceAfter=6)
h1_style = ParagraphStyle('Heading1_C', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=colors.HexColor('#2c5282'), spaceBefore=8, spaceAfter=3, keepWithNext=True)
h2_style = ParagraphStyle('Heading2_C', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=colors.HexColor('#2d3748'), spaceBefore=5, spaceAfter=2, keepWithNext=True)
body_style = ParagraphStyle('Body_C', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10.5, textColor=colors.HexColor('#2d3748'), spaceAfter=3)
bullet_style = ParagraphStyle('Bullet_C', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10.5, textColor=colors.HexColor('#2d3748'), leftIndent=10, spaceAfter=2)
code_style = ParagraphStyle('Code_C', parent=styles['Normal'], fontName='Courier', fontSize=6.5, leading=8.5, textColor=colors.HexColor('#1a202c'))

def clean_html(s):
    s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    parts = s.split('`')
    res = []
    for idx, p in enumerate(parts):
        if idx % 2 == 1:
            res.append('<font name="Courier">' + p + '</font>')
        else:
            res.append(p)
    return ''.join(res)

story = []
lines = text.splitlines()
i = 0
table_lines = []

while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    if stripped.startswith('|') and stripped.endswith('|'):
        table_lines.append(stripped)
        i += 1
        continue
    elif table_lines:
        rows_data = []
        for tl in table_lines:
            if '---' in tl: continue
            cells = [c.strip() for c in tl.split('|')[1:-1]]
            p_cells = [Paragraph(clean_html(c), body_style) for c in cells]
            rows_data.append(p_cells)
        if rows_data:
            col_widths = [540 / len(rows_data[0])] * len(rows_data[0])
            t = Table(rows_data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            story.append(t)
            story.append(Spacer(1, 4))
        table_lines = []

    if not stripped:
        i += 1
        continue

    if stripped.startswith('```'):
        i += 1
        code_block = []
        while i < len(lines) and not lines[i].strip().startswith('```'):
            code_block.append(lines[i].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace(' ', '&nbsp;'))
            i += 1
        i += 1
        code_p = Paragraph('<br/>'.join(code_block), code_style)
        t = Table([[code_p]], colWidths=[540])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7fafc')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t)
        story.append(Spacer(1, 4))
        continue

    safe_text = clean_html(stripped)
    if stripped.startswith('# CS3081'):
        story.append(Paragraph(safe_text.replace('# ', ''), title_style))
    elif stripped.startswith('## Knowledge'):
        story.append(Paragraph(safe_text.replace('## ', ''), subtitle_style))
    elif stripped.startswith('## Task') or stripped.startswith('## Executive'):
        story.append(Paragraph(safe_text.replace('## ', ''), h1_style))
    elif stripped.startswith('### '):
        story.append(Paragraph(safe_text.replace('### ', ''), h2_style))
    elif stripped.startswith('- ') or stripped.startswith('* '):
        story.append(Paragraph('&bull; ' + safe_text[2:], bullet_style))
    elif stripped.startswith(('1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ')):
        story.append(Paragraph(safe_text, bullet_style))
    elif stripped.startswith('---'):
        story.append(Spacer(1, 3))
    else:
        story.append(Paragraph(safe_text, body_style))
    i += 1

doc.build(story)

with open(pdf_path, 'rb') as fp:
    c = fp.read()
pages = len(re.findall(rb'/Type\s*/Page[^s]', c))
print('SUCCESS! report.pdf created at:', pdf_path)
print('Pages count:', pages, '(Must be <= 8)')

