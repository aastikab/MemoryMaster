"""Export functionality for summaries to PDF and Markdown."""

from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import markdown
from io import BytesIO

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class SummaryExporter:
    """Export summaries to PDF or Markdown formats."""
    
    def __init__(self):
        self.reportlab_available = REPORTLAB_AVAILABLE
    
    def export_to_markdown(self, summaries: List[Dict], output_path: Optional[Path] = None) -> str:
        """Export summaries to Markdown format.
        
        Args:
            summaries: List of summary dictionaries with keys like 'summary', 'note', 'similarity', etc.
            output_path: Optional path to save the file. If None, returns content as string.
        
        Returns:
            Markdown content as string
        """
        md_content = f"# Summary Export\n\n"
        md_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_content += f"---\n\n"
        
        for i, summary_data in enumerate(summaries, 1):
            md_content += f"## Summary {i}\n\n"
            
            if 'similarity' in summary_data:
                md_content += f"**Similarity Score:** {summary_data['similarity']:.2%}\n\n"
            
            if 'summary' in summary_data:
                if isinstance(summary_data['summary'], str):
                    md_content += f"{summary_data['summary']}\n\n"
                else:
                    # Handle structured summary
                    md_content += f"{summary_data['summary'].summary}\n\n"
                    if hasattr(summary_data['summary'], 'citation1'):
                        md_content += f"**Citation 1:** {summary_data['summary'].citation1.text}\n\n"
                        md_content += f"**Citation 2:** {summary_data['summary'].citation2.text}\n\n"
            
            if 'note' in summary_data:
                md_content += f"**Related Note:**\n\n```\n{summary_data['note'][:500]}...\n```\n\n"
            
            md_content += "---\n\n"
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(md_content, encoding='utf-8')
        
        return md_content
    
    def export_to_pdf(self, summaries: List[Dict], output_path: Optional[Path] = None) -> BytesIO:
        """Export summaries to PDF format.
        
        Args:
            summaries: List of summary dictionaries
            output_path: Optional path to save the file. If None, returns BytesIO object.
        
        Returns:
            BytesIO object with PDF content
        """
        if not self.reportlab_available:
            raise ImportError("reportlab is required for PDF export. Install it with: pip install reportlab")
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='#1f77b4',
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#1f77b4',
            spaceAfter=12,
            spaceBefore=12
        )
        
        story = []
        story.append(Paragraph("Summary Export", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Spacer(1, 0.2*inch))
        
        for i, summary_data in enumerate(summaries, 1):
            story.append(Paragraph(f"Summary {i}", heading_style))
            
            if 'similarity' in summary_data:
                story.append(Paragraph(f"<b>Similarity Score:</b> {summary_data['similarity']:.2%}", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            if 'summary' in summary_data:
                summary_text = summary_data['summary']
                if not isinstance(summary_text, str):
                    summary_text = summary_data['summary'].summary
                # Escape HTML and convert newlines
                summary_text = summary_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                summary_text = summary_text.replace('\n', '<br/>')
                story.append(Paragraph(summary_text, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            if 'note' in summary_data:
                note_preview = summary_data['note'][:500] + "..." if len(summary_data['note']) > 500 else summary_data['note']
                note_preview = note_preview.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph("<b>Related Note:</b>", styles['Normal']))
                story.append(Paragraph(note_preview, styles['Normal']))
            
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph("─" * 50, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        buffer.seek(0)
        
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'wb') as f:
                f.write(buffer.getvalue())
            buffer.seek(0)
        
        return buffer

