from nglui import statebuilder


def generate_statebuilder(client):
    img = statebuilder.ImageLayerConfig(
        client.info.image_source(), contrast_controls=True, black=0.35, white=0.65
    )
    seg = statebuilder.SegmentationLayerConfig(
        client.info.segmentation_source(),
        selected_ids_column=["pre_pt_root_id", "post_pt_root_id"],
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
