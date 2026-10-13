"""Template exporter for DeFoG or another PyG-based generator.

Replace ``generate_graphs`` with the model-specific generation call. The
returned objects must already contain final node/edge attributes with shared
channel semantics.
"""

from pathlib import Path

from ggm_eval import save_pyg_collection


def export_generated_graphs(
    generated_graphs,
    output_path,
    *,
    model_name,
    dataset,
    feature_schema,
):
    """Validate and save generated PyG Data objects for shared evaluation."""

    return save_pyg_collection(
        Path(output_path),
        generated_graphs,
        metadata={
            "generator": model_name,
            "dataset": dataset,
            "feature_schema": feature_schema,
            "split": "generated",
        },
    )


# Example inside a generator repository:
#
# generated = generate_graphs(checkpoint, count=len(real_test_graphs))
# export_generated_graphs(
#     generated,
#     "artifacts/defog_generated.pt",
#     model_name="DeFoG",
#     dataset="PROTEINS",
#     feature_schema="proteins-node-edge-onehot-v1",
# )
