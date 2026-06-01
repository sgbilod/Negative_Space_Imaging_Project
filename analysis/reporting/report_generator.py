#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Report Generator Module for Negative Space Imaging Project
Author: Stephen Bilodeau
Date: August 13, 2025

This module generates reports from analysis results.
"""

import os
import json
import logging
import datetime
from pathlib import Path
import glob

# For PDF report generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)

# Configure logging
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Class for generating reports from analysis results."""

    def __init__(self):
        """Initialize the ReportGenerator."""
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.templates = {
            'standard': self._generate_standard_report,
            'detailed': self._generate_detailed_report,
            'executive': self._generate_executive_report
        }

    def generate_report(self, input_dir, output_file, template='standard'):
        """
        Generate a report from analysis results.

        Args:
            input_dir (str): Directory containing analysis results
            output_file (str): Path to save the report
            template (str): Report template to use
                Options: 'standard', 'detailed', 'executive'

        Returns:
            str: Path to the generated report
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load results
        try:
            results = self._load_results(input_dir)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise

        # Generate report
        if template in self.templates:
            logger.info(f"Generating {template} report")
            report_path = self.templates[template](results, output_file)
            return report_path
        else:
            available_templates = ", ".join(self.templates.keys())
            logger.error(f"Unsupported template: {template}. "
                       f"Available templates: {available_templates}")
            raise ValueError(f"Unsupported template: {template}")

    def _load_results(self, input_dir):
        """
        Load analysis results from the input directory.

        Args:
            input_dir (str): Directory containing analysis results

        Returns:
            dict: Analysis results
        """
        input_path = Path(input_dir)
        results = {}

        # Look for results.json files
        for results_file in input_path.glob('**/results.json'):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    # Get the category from the parent directory name
                    category = results_file.parent.name
                    results[category] = data
            except Exception as e:
                logger.warning(f"Could not load {results_file}: {e}")

        # Look for visualization images
        visualizations = {}
        for img_file in input_path.glob('**/*.png'):
            # Get the category from the filename
            name = img_file.stem
            visualizations[name] = str(img_file)

        results['visualizations'] = visualizations

        # Add metadata
        results['metadata'] = {
            'timestamp': self.timestamp,
            'report_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_results': len(results) - 1  # Subtract 1 for the visualizations key
        }

        return results

    def _generate_standard_report(self, results, output_file):
        """Generate a standard report."""
        doc = SimpleDocTemplate(
            output_file,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name='Center',
                parent=styles['Heading1'],
                alignment=1
            )
        )

        # Build the document
        elements = []

        # Title
        elements.append(Paragraph(
            "Negative Space Imaging Project - Analysis Report",
            styles['Center']
        ))
        elements.append(Spacer(1, 20))

        # Metadata
        elements.append(Paragraph("Report Information", styles['Heading2']))
        elements.append(Paragraph(
            f"Generated on: {results['metadata']['report_date']}",
            styles['Normal']
        ))
        elements.append(Paragraph(
            f"Number of analysis results: {results['metadata']['num_results']}",
            styles['Normal']
        ))
        elements.append(Spacer(1, 20))

        # Results sections
        for category, data in results.items():
            if category in ['metadata', 'visualizations']:
                continue

            elements.append(Paragraph(
                f"{category.title()} Analysis",
                styles['Heading2']
            ))
            elements.append(Spacer(1, 10))

            # Handle different result types
            if isinstance(data, dict):
                # Create a table for the results
                table_data = [['Metric', 'Value']]
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        continue  # Skip complex nested structures
                    table_data.append([key, str(value)])

                if len(table_data) > 1:  # Only add table if we have data
                    table = Table(table_data, colWidths=[200, 250])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(table)
                    elements.append(Spacer(1, 10))

            # Add visualizations if available
            for viz_name, viz_path in results['visualizations'].items():
                if category.lower() in viz_name.lower():
                    img = Image(viz_path, width=450, height=300)
                    elements.append(img)
                    elements.append(Paragraph(
                        f"Figure: {viz_name.replace('_', ' ').title()}",
                        styles['Italic']
                    ))
                    elements.append(Spacer(1, 10))

            elements.append(PageBreak())

        # Build the PDF
        doc.build(elements)

        return output_file

    def _generate_detailed_report(self, results, output_file):
        """Generate a detailed report with more comprehensive information."""
        # For now, use the standard report as a base
        # In a real implementation, this would include more details
        return self._generate_standard_report(results, output_file)

    def _generate_executive_report(self, results, output_file):
        """Generate an executive summary report."""
        # For now, use the standard report as a base
        # In a real implementation, this would be more concise
        return self._generate_standard_report(results, output_file)


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Generate analysis reports')
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing analysis results'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='analysis_results/report.pdf',
        help='Output report file path'
    )
    parser.add_argument(
        '--template',
        type=str,
        default='standard',
        choices=['standard', 'detailed', 'executive'],
        help='Report template to use'
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    generator = ReportGenerator()
    report_path = generator.generate_report(
        args.input_dir,
        args.output_file,
        args.template
    )

    print(f"Report saved to: {report_path}")
