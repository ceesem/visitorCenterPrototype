import pandas as pd
import numpy as np
import datetime
from functools import lru_cache

from typing import *
from annotationframeworkclient import FrameworkClient
import datetime

import plotly.graph_objects as go
from .dataframe_utilities import *
from .config import *

table_columns = [
    "pt_root_id",
    num_syn_col,
    net_size_col,
    mean_size_col,
    soma_dist_col,
    ct_col,
    soma_depth_col,
    valence_col,
    num_soma_col,
]


class NeuronData(object):
    def __init__(
        self,
        oid: int,
        client: FrameworkClient,
        timestamp: Union[datetime.datetime, None] = None,
        axon_only: bool = False,
        split_threshold: float = 0.7,
        synapse_table: str = synapse_table,
        cell_type_table: str = cell_type_table,
        soma_table: str = soma_table,
    ) -> None:
        self._oid = oid
        self._client = client
        if timestamp is None:
            timestamp = datetime.datetime.now()
        self._timestamp = timestamp
        self.axon_only = axon_only

        self.synapse_table = synapse_table
        self.soma_table = soma_table
        self.cell_type_table = cell_type_table

        self.split_threshold = split_threshold

    @property
    def oid(self) -> int:
        return self._oid

    @property
    def client(self) -> FrameworkClient:
        return self._client

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @lru_cache
    def pre_syn_df(self) -> pd.DataFrame:
        return pre_synapse_df(
            self.synapse_table,
            self.oid,
            self.client,
            self.timestamp,
        )

    @lru_cache
    def post_syn_df(self) -> pd.DataFrame:
        return post_synapse_df(
            self.synapse_table,
            self.oid,
            self.client,
            self.timestamp,
        )

    @lru_cache
    def syn_df(self) -> pd.DataFrame:
        pre_df = self.pre_syn_df()
        pre_df["direction"] = "pre"
        post_df = self.post_syn_df()
        post_df["direction"] = "post"
        syn_df = pd.concat([pre_df, post_df])
        syn_df["x"] = 0
        return syn_df

    def _violin_plot(self, direction, name, side, color, xaxis, yaxis):
        return go.Violin(
            x=self.syn_df().query("direction == @direction")["x"],
            y=self.syn_df().query("direction == @direction")[syn_depth_col],
            side=side,
            scalegroup="syn",
            name=name,
            points=False,
            line_color=f"rgb{color}",
            fillcolor=f"rgb{color}",
            xaxis=xaxis,
            yaxis=yaxis,
        )

    def post_violin_plot(
        self,
        color: Iterable[float],
        xaxis: Union[str, None] = None,
        yaxis: Union[str, None] = None,
    ) -> go.Violin:
        return self._violin_plot(
            "post", "Post", "negative", color, xaxis=xaxis, yaxis=yaxis
        )

    def pre_violin_plot(
        self,
        color: Iterable[float],
        xaxis: Union[str, None] = None,
        yaxis: Union[str, None] = None,
    ) -> go.Violin:
        return self._violin_plot(
            "pre", "Post", "positive", color, xaxis=xaxis, yaxis=yaxis
        )

    def _get_own_soma_loc(self):
        own_soma_df = get_specific_soma(
            self.soma_table, self.oid, self.client, self.timestamp
        )
        if len(own_soma_df) != 1:
            own_soma_loc = np.nan
        else:
            own_soma_loc = own_soma_df["pt_position"].values[0]
        return own_soma_loc

    @lru_cache
    def soma_location(self) -> np.ndarray:
        return np.array(self._get_own_soma_loc())

    def soma_location_list(self, length: int) -> list:
        return np.repeat(np.atleast_2d(self.soma_location()), length, axis=0).tolist()

    def _get_ct_soma_df(self, target_ids):
        targ_ct_soma_df = cell_typed_soma_df(
            self.soma_table,
            self.cell_type_table,
            target_ids,
            self.client,
            self.timestamp,
        )
        return targ_ct_soma_df

    def _target_ids(self):
        pre_oids = self.pre_syn_df()["post_pt_root_id"]
        post_oids = self.post_syn_df()["pre_pt_root_id"]
        return np.unique(np.concatenate([pre_oids, post_oids]))

    @lru_cache
    def targ_soma_df(self):
        return self._get_ct_soma_df(self._target_ids())

    def _compute_pre_targ_df(self):
        pre_df = self.pre_syn_df()
        targ_soma_df = self.targ_soma_df()

        pre_targ_df = pre_df.merge(
            targ_soma_df,
            left_on="post_pt_root_id",
            right_on="pt_root_id",
            how="left",
        ).drop(columns=["pt_root_id"])
        pre_targ_df[num_soma_col].fillna(0, inplace=True)
        pre_targ_df[num_soma_col] = pre_targ_df[num_soma_col].astype(int)

        pre_targ_df[own_soma_col] = self.soma_location_list(len(pre_targ_df))
        pre_targ_df[soma_dist_col] = pre_targ_df.apply(
            lambda x: radial_distance(
                x, soma_position_col, own_soma_col, voxel_resolution
            ),
            axis=1,
        )
        return pre_targ_df

    @lru_cache
    def pre_targ_df(self) -> pd.DataFrame:
        return self._compute_pre_targ_df().fillna(np.nan)

    def _compute_post_targ_df(self):
        post_df = self.post_syn_df()
        targ_soma_df = self.targ_soma_df()

        post_targ_df = post_df.merge(
            targ_soma_df,
            left_on="pre_pt_root_id",
            right_on="pt_root_id",
            how="left",
        ).drop(columns=["pt_root_id"])
        post_targ_df[num_soma_col].fillna(0, inplace=True)
        post_targ_df[num_soma_col] = post_targ_df[num_soma_col].astype(int)
        post_targ_df[own_soma_col] = self.soma_location_list(len(post_targ_df))
        post_targ_df[soma_dist_col] = post_targ_df.apply(
            lambda x: radial_distance(
                x, soma_position_col, own_soma_col, voxel_resolution
            ),
            axis=1,
        )
        return post_targ_df

    @lru_cache
    def post_targ_df(self) -> pd.DataFrame:
        return self._compute_post_targ_df().fillna(np.nan)

    def synapse_soma_scatterplot(
        self,
        valence_colors: dict,
        xaxis: Union[str, None] = None,
        yaxis: Union[str, None] = None,
    ) -> go.Scattergl:
        pre_targ_df = self.pre_targ_df().dropna()
        return go.Scattergl(
            x=pre_targ_df[soma_depth_col],
            y=pre_targ_df[syn_depth_col],
            mode="markers",
            marker=dict(
                color=valence_colors[pre_targ_df[valence_col].astype(int)],
                line_width=0,
                size=5,
                opacity=0.5,
            ),
            xaxis=xaxis,
            yaxis=yaxis,
        )

    def bar_data(self) -> pd.Series:
        pre_targ_df = self.pre_targ_df().dropna()
        return pre_targ_df.groupby(ct_col).count()["size"]

    def _bar_plot(self, name, indices, color):
        return go.Bar(
            name=name,
            y=self.bar_data().loc[indices].index,
            x=self.bar_data().loc[indices].values,
            marker_color=f"rgb{color}",
            orientation="h",
        )

    def excitatory_bar_plot(self, color: Iterable[float]) -> go.Bar:
        return self._bar_plot(
            "Exc. Targets", exc_types, tuple(np.floor(255 * color).astype(int))
        )

    def inhib_bar_plot(self, color: Iterable[float]) -> go.Bar:
        return self._bar_plot(
            "Inh. Targets", inhib_types, tuple(np.floor(255 * color).astype(int))
        )

    def _compute_tab_dat(self, direction):
        if direction == "pre":
            df = self.pre_targ_df()
            merge_column = "post_pt_root_id"
        elif direction == "post":
            df = self.post_targ_df()
            merge_column = "pre_pt_root_id"
        df[num_syn_col] = df.groupby(merge_column).transform("count")["ctr_pt_position"]
        df[net_size_col] = (
            df[[merge_column, "size"]].groupby(merge_column).transform("sum")["size"]
        )
        df[mean_size_col] = (
            df[[merge_column, "size"]].groupby(merge_column).transform("mean")["size"]
        ).astype(int)
        df_unique = df.drop_duplicates(subset=merge_column).drop(
            columns=["size", "ctr_pt_position"]
        )
        tab_dat = df_unique.sort_values(by=num_syn_col, ascending=False)
        tab_dat[merge_column] = tab_dat[merge_column].astype(
            str
        )  # Dash can't handle int64
        return tab_dat

    @lru_cache
    def pre_tab_dat(self) -> pd.DataFrame:
        return (
            self._compute_tab_dat("pre")
            .fillna(np.nan)
            .drop(columns=["pre_pt_root_id"])
            .rename(columns={"post_pt_root_id": "pt_root_id"})
        )

    @lru_cache
    def post_tab_dat(self) -> pd.DataFrame:
        return (
            self._compute_tab_dat("post")
            .fillna(np.nan)
            .drop(columns=["post_pt_root_id"])
            .rename(columns={"pre_pt_root_id": "pt_root_id"})
        )
