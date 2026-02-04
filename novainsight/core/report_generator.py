from __future__ import annotations

import logging
import base64
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from jinja2 import Template, FileSystemLoader, Environment

from novainsight.core.base_module import BaseModule
from novainsight.config.config import NovaInsightConfig
from novainsight.schemas.analysis_report import (
    AnalysisReport,
    Finding
)

logger = logging.getLogger(__name__)

class ReportGenerator(BaseModule):
    """
    The 'Publisher' of NovaInsight.
    Compiles the AnalysisReport, findings, and plots into a polished HTML document.
    """

    def __init__(self, config: NovaInsightConfig):
        super().__init__(config)
        self.findings: List[Finding] = []

    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        """
        Orchestrates the report generation.
        1. Encodes all images to Base64.
        2. Renders the Jinja2 template.
        3. Saves the HTML file.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(current_dir, '../templates')

        file_loader = FileSystemLoader(template_dir)
        env = Environment(loader=file_loader)

        try:
            logger.info("Starting report generation...")
            
            images = self._prepare_images(report)
            
            sample_html = df.head(5).to_html(
                classes="table table-custom table-hover table-striped",
                border=0,
                index=False
            )

            # Render Template
            print(Path('../templates').exists())
            template = env.get_template('report_template.html')
            html_content = template.render(
                report=report,
                images=images,
                sample_data_html=sample_html,
                includes_column_name=self.includes_column_name
            )

            # Save to Disk
            output_dir = Path(report.metadata.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            safe_title = report.metadata.report_title.replace(" ", "_").lower()
            filename = f"report_{safe_title}.html"
            output_path = output_dir / filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            logger.info(f"HTML Report successfully generated at: {output_path}")
            
            # Add an INFO finding to the report so the user knows where it is
            logger.info(f"HTML Report generated: {output_path}")
            report.findings.append(Finding(
                level="INFO",
                message=f"HTML Report generated: {output_path}"
            ))

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            self.findings.append(Finding(
                level="ERROR",
                message=f"Failed to generate HTML report: {e}"
            ))
            report.findings.extend(self.findings)

        return report

    def _prepare_images(self, report: AnalysisReport) -> Dict[str, Any]:
        """
        Reads image paths from the report schema and converts them to 
        Base64 strings for embedding in HTML.
        
        Returns a nested dictionary:
        {
            'univariate': {'col_name': 'b64...'},
            'bivariate': {'plot_name': 'b64...'},
            'correlation': {'name': 'b64...'},
            'dimensionality': {'name': 'b64...'}
        }
        """
        images = {
            'univariate': {},
            'bivariate': {},
            'correlation': {},
            'dimensionality': {}
        }
        
        if not report.visualizations:
            return images

        viz = report.visualizations

        # Univariate
        for col, path in viz.univariate_plots.items():
            b64 = self._image_to_base64(path)
            if b64: images['univariate'][col] = b64

        # Bivariate
        for name, path in viz.bivariate_plots.items():
            b64 = self._image_to_base64(path)
            if b64: images['bivariate'][name] = b64

        if viz.correlation_heatmap_plots:
            for name, path in viz.correlation_heatmap_plots.items():
                b64 = self._image_to_base64(path)
                if b64: images['correlation'][name] = b64

        if viz.dim_reduction_scatter_plots:
            for name, path in viz.dim_reduction_scatter_plots.items():
                b64 = self._image_to_base64(path)
                if b64: images['dimensionality'][name] = b64

        return images

    def _image_to_base64(self, path: str | Path) -> Optional[str]:
        """Reads an image file and returns base64 string."""
        if not path:
            return None
        
        try:
            p = Path(path)
            if not p.exists():
                logger.warning(f"Image not found during report generation: {p}")
                return None
                
            with open(p, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to encode image {path}: {e}")
            return None
    
    @staticmethod
    def includes_column_name(name: str, string: str) -> bool:
        pattern = rf"\b{re.escape(name)}\b"
        return bool(re.search(pattern, string))