import pandas as pd
import numpy as np
import pcg_skel
import datetime

import plotly.graph_objects as go
from meshparty import meshwork
from .config import *

table_columns = [
    "nucleus_id",
    "post_pt_root_id",
    "num_syn",
    "net_size",
    "cell_type_pred",
    soma_depth_col,
    valence_col,
    "num_soma",
]


class NeuronCardData(object):
    def __init__(
        self,
        oid,
        client,
        timestamp=None,
        axon_only=False,
        split_threshold=0.7,
        synapse_table=synapse_table,
    ):
        self._oid = oid
        self._client = client
        if timestamp is None:
            timestamp = datetime.datetime.now()
        self._timestamp = timestamp
        self.axon_only = axon_only
        self.synapse_table = synapse_table
        self.split_threshold = split_threshold

        self._nrn = None
        self._vertex_df = None
        self._syn_df = None
        self._pre_targ_df = None
        self._tab_dat = None
        self._link_function = None

    @property
    def oid(self):
        return self._oid

    @property
    def client(self):
        return self._client

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def nrn(self):
        if self._nrn is None:
            nrn = pcg_skel.pcg_meshwork(
                self.oid,
                client=self.client,
                refine=None,
                synapses="all",
                live_query=True,
                timestamp=self.timestamp,
                synapse_table=self.synapse_table,
            )
            self._nrn = nrn
        return self._nrn

    @property
    def vertex_df(self):
        if self._vertex_df is None:
            if not self.axon_only:
                is_axon, q = meshwork.algorithms.split_axon_by_synapses(
                    self.nrn,
                    self.nrn.anno.pre_syn.mesh_index,
                    self.nrn.anno.post_syn.mesh_index,
                )
                if q > split_threshold:
                    df = pd.DataFrame(
                        {
                            "x": self.nrn.skeleton.vertices[:, 0] / 1000,
                            "y": self.nrn.skeleton.vertices[:, 1] / 1000,
                            "z": self.nrn.skeleton.vertices[:, 2] / 1000,
                            "is_axon": is_axon.to_skel_mask.astype(int),
                        }
                    )
                else:
                    df = pd.DataFrame(
                        {
                            "x": self.nrn.skeleton.vertices[:, 0] / 1000,
                            "y": self.nrn.skeleton.vertices[:, 1] / 1000,
                            "z": self.nrn.skeleton.vertices[:, 2] / 1000,
                            "is_axon": 1,
                        }
                    )
            else:
                df = pd.DataFrame(
                    {
                        "x": self.nrn.skeleton.vertices[:, 0] / 1000,
                        "y": self.nrn.skeleton.vertices[:, 1] / 1000,
                        "z": self.nrn.skeleton.vertices[:, 2] / 1000,
                        "is_axon": 0,
                    }
                )
            self._vertex_df = df
        return self._vertex_df

    def _dot_plot(self, color, is_axon, name, xaxis="x", yaxis="y"):
        dot = go.Scattergl(
            x=self.vertex_df.query("is_axon == @is_axon")["x"],
            y=self.vertex_df.query("is_axon == @is_axon")["y"],
            mode="markers",
            marker=dict(
                color=color,
                line_width=0,
                size=2.5,
            ),
            name=name,
            xaxis=xaxis,
            yaxis=yaxis,
        )
        return dot

    def axon_dot_plot(self, color, xaxis=None, yaxis=None):
        return self._dot_plot(color, 1, "Axon", xaxis=xaxis, yaxis=yaxis)

    def dendrite_dot_plot(self, color, xaxis=None, yaxis=None):
        return self._dot_plot(color, 0, "Dendrite", xaxis=xaxis, yaxis=yaxis)

    @property
    def syn_df(self):
        if self._syn_df is None:
            pre_df = self.client.materialize.live_query(
                synapse_table,
                filter_equal_dict={"pre_pt_root_id": self.oid},
                timestamp=self.timestamp,
            )
            pre_df["direction"] = "pre"
            post_df = self.client.materialize.live_query(
                synapse_table,
                filter_equal_dict={"post_pt_root_id": self.oid},
                timestamp=self.timestamp,
            )
            post_df["direction"] = "post"
            syn_df = pd.concat([pre_df, post_df])
            syn_df = syn_df.query("pre_pt_root_id != post_pt_root_id").reset_index(
                drop=True
            )
            syn_df["x"] = 0
            syn_df["syn_depth"] = syn_df["ctr_pt_position"].apply(
                lambda x: x[1] * 4 / 1000
            )
            self._syn_df = syn_df
        return self._syn_df

    def _violin_plot(self, direction, name, side, color, xaxis, yaxis):
        return go.Violin(
            x=self.syn_df.query("direction == @direction")["x"],
            y=self.syn_df.query("direction == @direction")["syn_depth"],
            side=side,
            scalegroup="syn",
            name=name,
            points=False,
            line_color=f"rgb{color}",
            fillcolor=f"rgb{color}",
            xaxis=xaxis,
            yaxis=yaxis,
        )

    def post_violin_plot(self, color, xaxis=None, yaxis=None):
        return self._violin_plot(
            "post", "Post", "negative", color, xaxis=xaxis, yaxis=yaxis
        )

    def pre_violin_plot(self, color, xaxis=None, yaxis=None):
        return self._violin_plot(
            "pre", "Post", "positive", color, xaxis=xaxis, yaxis=yaxis
        )

    def _get_targ_soma_df(self, soma_table, target_ids):
        targ_soma_df = self.client.materialize.live_query(
            soma_table,
            filter_in_dict={"pt_root_id": target_ids},
            timestamp=self.timestamp,
        )
        targ_soma_df["num_soma"] = (
            targ_soma_df.query('cell_type == "neuron"')
            .groupby("pt_root_id")
            .transform("count")["valid"]
        )
        return targ_soma_df

    def _compute_pre_targ_df(self, soma_table, ct_base_df):
        pre_df = self.syn_df.query('direction == "pre"')
        target_ids = np.unique(pre_df["post_pt_root_id"])
        targ_soma_df = self._get_targ_soma_df(soma_table, target_ids)

        ct_base_columns = ["nucleus_id", ct_col, soma_depth_col, valence_col]
        ct_df = ct_base_df[ct_base_columns].merge(
            targ_soma_df[["pt_root_id", "id", "num_soma"]],
            left_on="nucleus_id",
            right_on="id",
        )

        multisoma_ind = ct_df.query("num_soma>1").index
        ct_df.loc[multisoma_ind, "nucleus_id"] = np.nan
        ct_df = ct_df.drop_duplicates(subset=["pt_root_id"]).reset_index(drop=True)

        pre_targ_df = pre_df.merge(
            ct_df[
                [
                    "pt_root_id",
                    ct_col,
                    valence_col,
                    soma_depth_col,
                    "nucleus_id",
                    "num_soma",
                ]
            ],
            left_on="post_pt_root_id",
            right_on="pt_root_id",
            how="left",
        )
        pre_targ_df["num_soma"].fillna(0, inplace=True)
        pre_targ_df[syn_depth_col] = pre_targ_df["ctr_pt_position"].apply(
            lambda x: x[1] * 4 / 1000
        )
        return pre_targ_df

    @property
    def pre_targ_df(self):
        if self._pre_targ_df is None:
            self._pre_targ_df = self._compute_pre_targ_df(soma_table, ct_base_df)
        return self._pre_targ_df

    def synapse_soma_scatterplot(self, valence_colors, xaxis=None, yaxis=None):
        pre_targ_df = self.pre_targ_df.dropna()
        return go.Scattergl(
            x=pre_targ_df[soma_depth_col],
            y=pre_targ_df[syn_depth_col],
            mode="markers",
            marker=dict(
                color=val_colors[pre_targ_df[valence_col].astype(int)],
                line_width=0,
                size=5,
                opacity=0.5,
            ),
            xaxis=xaxis,
            yaxis=yaxis,
        )

    @property
    def bar_data(self):
        pre_targ_df = self.pre_targ_df.dropna()
        return pre_targ_df.groupby(ct_col).count()["valid"]

    def _bar_plot(self, name, indices, color):
        return go.Bar(
            name=name,
            y=self.bar_data.loc[indices].index,
            x=self.bar_data.loc[indices].values,
            marker_color=f"rgb{color}",
            orientation="h",
        )

    def excitatory_bar_plot(self, color):
        return self._bar_plot(
            "Exc. Targets", exc_types, tuple(np.floor(255 * color).astype(int))
        )

    def inhib_bar_plot(self, color):
        return self._bar_plot(
            "Inh. Targets", inhib_types, tuple(np.floor(255 * color).astype(int))
        )

    @property
    def tab_dat(self):
        if self._tab_dat is None:
            pre_targ_df = self.pre_targ_df
            pre_targ_df["num_syn"] = pre_targ_df.groupby("post_pt_root_id").transform(
                "count"
            )["valid"]
            pre_targ_df["net_size"] = (
                pre_targ_df[["post_pt_root_id", "size"]]
                .groupby("post_pt_root_id")
                .transform("sum")["size"]
            )
            pre_targ_unique_df = pre_targ_df.drop_duplicates(subset="post_pt_root_id")
            tab_dat = pre_targ_unique_df[table_columns].sort_values(
                by="num_syn", ascending=False
            )
            tab_dat["num_soma"] = tab_dat["num_soma"].astype(int)
            tab_dat["post_pt_root_id"] = tab_dat["post_pt_root_id"].astype(str)
            self._tab_dat = tab_dat
        return self._tab_dat