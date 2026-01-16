"""
Gerador de PDF para Relatórios de Inteligência - Formato Executivo
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
from pathlib import Path

def gerar_pdf_relatorio(conteudo_texto: str, titulo: str = "Relatório de Inteligência", nome_arquivo: str = None, dados_risco: dict = None) -> Path:
    """
    Gera um PDF com o relatório de inteligência em formato executivo.
    
    Args:
        conteudo_texto: Texto do relatório
        titulo: Título do documento
        nome_arquivo: Nome do arquivo (sem .pdf)
        dados_risco: Dict com dados de risco {bairro: valor}
    
    Returns:
        Path ao arquivo criado
    """
    
    if not nome_arquivo:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"relatorio_{timestamp}"
    
    output_dir = Path(__file__).parent.parent / "outputs" / "reports" / "relatorio_usuario"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    arquivo_pdf = output_dir / f"{nome_arquivo}.pdf"
    
    # Cria documento PDF
    doc = SimpleDocTemplate(
        str(arquivo_pdf),
        pagesize=A4,
        rightMargin=0.6*inch,
        leftMargin=0.6*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch,
    )
    
    # Estilos executivos
    styles = getSampleStyleSheet()
    
    style_titulo = ParagraphStyle(
        'ExecTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor("#1a1f2e"),
        spaceAfter=6,
        alignment=1,
        fontName='Helvetica-Bold'
    )
    
    style_subtitulo = ParagraphStyle(
        'ExecSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#666666"),
        spaceAfter=12,
        alignment=1,
        fontName='Helvetica-Oblique'
    )
    
    style_secao = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor("#ffffff"),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        textTransform='uppercase',
        backColor=colors.HexColor("#1a1f2e"),
        leftIndent=6,
        rightIndent=6,
        topPadding=6,
        bottomPadding=6
    )
    
    style_corpo = ParagraphStyle(
        'ExecBody',
        parent=styles['BodyText'],
        fontSize=10,
        textColor=colors.HexColor("#333333"),
        spaceAfter=8,
        leading=13,
        alignment=4,
    )
    
    style_destaques = ParagraphStyle(
        'Highlight',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=colors.HexColor("#e94560"),
        spaceAfter=6,
        fontName='Helvetica-Bold',
    )
    
    # Conteúdo
    content = []
    
    # Cabeçalho Executivo
    content.append(Paragraph("SIGERAIO", style_titulo))
    content.append(Paragraph("Análise de Inteligência Operacional", style_subtitulo))
    content.append(Spacer(1, 0.2*inch))
    
    # Info documento
    data_atual = datetime.now().strftime("%d de %B de %Y às %H:%M")
    info_table = Table([
        ["Documento:", "Análise de Inteligência"],
        ["Data/Hora:", data_atual],
        ["Classificação:", "Uso Operacional Interno"]
    ], colWidths=[2*inch, 4*inch])
    
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#333333")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    
    content.append(info_table)
    content.append(Spacer(1, 0.3*inch))
    
    # Áreas Críticas
    content.append(Paragraph("Áreas de Risco Crítico", style_secao))
    
    if dados_risco:
        dados_tabela = [["Área", "Índice de Risco", "Status"]]
        for area, valor in sorted(dados_risco.items(), key=lambda x: x[1], reverse=True)[:5]:
            if valor > 1.5:
                status = "🔴 CRÍTICO"
            elif valor > 1.0:
                status = "🟠 ALTO"
            else:
                status = "🟡 MODERADO"
            dados_tabela.append([area, f"{valor:.2f}", status])
        
        risk_table = Table(dados_tabela, colWidths=[3.5*inch, 1.5*inch, 1.5*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a1f2e")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        content.append(risk_table)
        content.append(Spacer(1, 0.2*inch))
    
    # Análise
    content.append(Paragraph("Análise Operacional", style_secao))
    content.append(Spacer(1, 0.1*inch))
    
    linhas = conteudo_texto.split('\n')
    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue
        if linha.startswith('**') and linha.endswith('**'):
            content.append(Paragraph(linha.replace('**', ''), style_destaques))
        elif linha.startswith('#'):
            content.append(Paragraph(linha.replace('#', '').strip(), style_secao))
        else:
            content.append(Paragraph(linha, style_corpo))
    
    content.append(Spacer(1, 0.3*inch))
    content.append(Paragraph("_" * 80, styles['Normal']))
    content.append(Spacer(1, 0.1*inch))
    
    footer_text = "SIGERAIO - Conselheiro de Inteligência | CPRAIO<br/>" \
                  "<font size=8>Análise baseada em ST-GCN. Uso operacional interno.</font>"
    content.append(Paragraph(footer_text, styles['Normal']))
    
    doc.build(content)
    return arquivo_pdf
