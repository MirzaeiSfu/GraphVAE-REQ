"""Generation-only entry: downstream structural evaluation is run separately."""
import runpy
import sys
from pathlib import Path

source = Path('/local-scratch2/mirzaei/QM9_FULL_TEST_20260925/defog_source')
sys.path[:0] = [str(source / 'src'), str(source)]
from graph_discrete_flow_model import GraphDiscreteFlowModel

def deferred_metrics(self, *args, **kwargs):
    print('Full-test sampling complete; built-in sampling metrics deferred to the common structural evaluator.', flush=True)
    return {}

GraphDiscreteFlowModel.evaluate_samples = deferred_metrics
runpy.run_path(str(source / 'src/main.py'), run_name='__main__')
