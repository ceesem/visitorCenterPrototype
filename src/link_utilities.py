from dash_html_components.B import B
from nglui import statebuilder


def generate_statebuilder(
    client, base_root_id=None, base_color="#ffffff", preselect_all=True
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
        "syns",
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
        "syns", mapping_rules=points, linked_segmentation_layer=seg.name
    )
    sb = statebuilder.StateBuilder([img, seg, anno], client=client)
    return sb
