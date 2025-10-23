import click
import logging
import sys
from pathlib import Path
from novainsight.config.config import load_config
from novainsight.core.orchestrator import AnalysisPipeline
from novainsight.schemas.analysis_report import Operator

config = load_config()
logger = logging.getLogger(__name__)

@click.group()
def main():
    """NovaInsight: Autonomous Data Analysis Agent"""
    pass

@main.command()
@click.option(
  '--output-dir',
  type=click.Path(file_okay=False, writable=True),
  default=None,
  help="Directory to save all output files."
)

@click.option(
  '--task',
  type=click.Choice(['supervised', 'unsupervised'], case_sensitive=False),
  default='supervised',
  help="Specify the analysis task type."
)

@click.option(
  # For now, it does nothing since no advanced analysis techniques have been implemented.
  '--advanced',
  is_flag=True,
  help="Enable advanced, computationally intensive analytical techniques."
)

@click.option(
  '--modules',
  type=str,
  default=None,
  help="Comma-separated list of modules to run (e.g., 'profiler,stats')."
)

@click.option('--target', type=str, default=None, help="Manually specify the target variable.")
@click.option('--title', type=str, default=None, help="Provide a custom title for the reports.")
@click.option(
  '--mode',
  type=click.Choice(['fast', 'full'], case_sensitive=False),
  default=None,
  help="Specify the analysis mode."
)

@click.option('--force-rerun', is_flag=True, help="Ignore cache and run a fresh analysis.")
@click.argument(
  'file_path',
  type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True)
)

def analyze(file_path, **kwargs):
    """
    Performs a comprehensive analysis on a given tabular dataset.
    """
    logger.info("NovaInsight CLI started.")
    file_path_obj = Path(file_path)
    if kwargs.get('modules'):
      module_strings = [m.strip().upper() for m in kwargs['modules'].split(',')]
      kwargs['requested_modules'] = [Operator[m].value for m in module_strings if m in Operator.__members__]
    else:
      kwargs['requested_modules'] = None
    try:
      pipeline = AnalysisPipeline(
        file_path=file_path_obj,
        config=config,
        output_dir=kwargs.get('output_dir'),
        task=kwargs.get('task'),
        requested_modules=kwargs.get('requested_modules'),
        user_target=kwargs.get('target'),
        report_title=kwargs.get('title'),
        analysis_mode=kwargs.get('mode'),
        force_rerun=kwargs.get('force_rerun'),
        advanced=kwargs.get('advanced')
      )
      pipeline.run()
    except Exception as e:
      logger.error(f"A fatal error occurred during the analysis: {e}")
    sys.exit(1)

if __name__ == '__main__':
    main()