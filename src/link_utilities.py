from dash_html_components.B import B
from nglui import statebuilder


def generate_statebuilder(
    client,
    base_root_id=None,
    base_color="#ffffff",
    preselect_all=True,
    anno_layer="syns",
):
    img = statebuilder.ImageLayerConfig(
        client.info.image_source(), contrast_controls=True, black=0.35, white=0.65
    )
    if preselect_all:
        selected_ids_column = ["post_pt_root_id"]
    else:
        selected_ids_column = None
    if base_root_id is None:
        base_root_id = []
        base_color = [None]
    else:
        base_root_id = [base_root_id]
        base_color = [base_color]

    print(selected_ids_column, base_root_id, base_color)
    seg = statebuilder.SegmentationLayerConfig(
        client.info.segmentation_source(),
        selected_ids_column=selected_ids_column,
        fixed_ids=base_root_id,
        fixed_id_colors=base_color,
        alpha_3d=0.8,
    )

    points = statebuilder.PointMapper(
        "ctr_pt_position",
        linked_segmentation_column="post_pt_root_id",
        group_column="post_pt_root_id",
        set_position=True,
    )
    anno = statebuilder.AnnotationLayerConfig(
        anno_layer,
        mapping_rules=points,
        linked_segmentation_layer=seg.name,
        filter_by_segmentation=True,
    )
    sb = statebuilder.StateBuilder([img, seg, anno], client=client)
    return sb


def generate_statebuilder_pre(client):
    img = statebuilder.ImageLayerConfig(
        client.info.image_source(), contrast_controls=True, black=0.35, white=0.65
    )
    seg = statebuilder.SegmentationLayerConfig(
        client.info.segmentation_source(),
        selected_ids_column=["pre_pt_root_id"],
        alpha_3d=0.8,
    )
    points = statebuilder.PointMapper(
        "ctr_pt_position",
        linked_segmentation_column="post_pt_root_id",
        set_position=True,
    )
    anno = statebuilder.AnnotationLayerConfig(
        "output_syns", mapping_rules=points, linked_segmentation_layer=seg.name
    )
    sb = statebuilder.StateBuilder([img, seg, anno], client=client)
    return sb


def generate_statebuilder_post(client):
    img = statebuilder.ImageLayerConfig(
        client.info.image_source(), contrast_controls=True, black=0.35, white=0.65
    )
    seg = statebuilder.SegmentationLayerConfig(
        client.info.segmentation_source(),
        selected_ids_column=["post_pt_root_id"],
        alpha_3d=0.8,
    )
    points = statebuilder.PointMapper(
        "ctr_pt_position",
        linked_segmentation_column="pre_pt_root_id",
        set_position=True,
    )
    anno = statebuilder.AnnotationLayerConfig(
        "input_syns", mapping_rules=points, linked_segmentation_layer=seg.name
    )
    sb = statebuilder.StateBuilder([img, seg, anno], client=client)
    return sb


def generate_url_synapses(selected_rows, edge_df, syn_df, direction, client):
    if direction == "pre":
        other_col = "post_pt_root_id"
        self_col = "pre_pt_root_id"
        anno_layer = ("output_syns",)
    else:
        other_col = "pre_pt_root_id"
        self_col = "post_pt_root_id"
        anno_layer = ("input_syn",)
    edge_df[other_col] = edge_df[other_col].astype(int)
    syn_df[other_col] = syn_df[other_col].astype(int)
    syn_df[self_col] = syn_df[self_col].astype(int)

    other_oids = edge_df.loc[selected_rows][other_col].values
    preselect = len(other_oids) == 1  # Only show all targets if just one is selected
    sb = generate_statebuilder(
        client,
        syn_df[self_col].iloc[0],
        preselect_all=preselect,
        anno_layer=anno_layer,
    )
    return sb.render_state(syn_df.query(f"{other_col} in @other_oids"), return_as="url")
