from src.dataframe_utilities import stringify_root_ids
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash import Dash
from jupyter_dash import JupyterDash

from .link_utilities import (
    generate_statebuilder,
    generate_statebuilder_pre,
    generate_statebuilder_post,
    generate_url_synapses,
)
from .dataframe_utilities import minimal_synapse_columns
from .neuron_data_base import NeuronData, table_columns
from .config import *
from .plots import *

dash_func_selector = {
    "jupyterdash": JupyterDash,
    "dash": Dash,
}


def generate_app(client, app_type="jupyterdash"):
    """Generate a viewer app

    Parameters
    ----------
    client : FrameworkClient
        Client for interacting with the CAVE framework
    appstyle : OneOf(['dash', 'jupyterdash']), optional
        String specifying what type of app to make. One of, by default 'jupyterdash'.

    Returns
    -------
    app
        Dash or JupyterDash app
    """

    DashFunc = dash_func_selector[app_type]
    header_text = html.H3(f"Neuron Target Info:")

    input_row = [
        html.Div("Root ID:"),
        dcc.Input(id="root_id", value="", type="text"),
        html.Button(id="submit-button", children="Submit"),
        html.Div(id="response-text", children=""),
    ]

    plot_header = [
        html.H4(id="plot-response-text", children=""),
    ]

    top_link = dbc.Row(
        [
            dbc.Col(
                [
                    html.A(
                        "Neuroglancer Link",
                        id="ngl_link",
                        href="",
                        target="_blank",
                        style={"font-size": "20px"},
                    ),
                ],
                width={"size": 2, "offset": 1},
            ),
            dbc.Col(
                html.Button(id="reset-selection", children="Reset Selection"),
                width={"size": 2, "offset": 3},
            ),
        ],
        justify="left",
    )

    data_table = html.Div(
        [
            dcc.Tabs(
                id="connectivity-tab",
                value="tab-pre",
                children=[
                    dcc.Tab(id="output-tab", label="Output", value="tab-pre"),
                    dcc.Tab(id="input-tab", label="Input", value="tab-post"),
                ],
            ),
            html.Div(
                dbc.Row(
                    [
                        dbc.Col(
                            dash_table.DataTable(
                                id="data-table",
                                columns=[{"name": i, "id": i} for i in table_columns],
                                data=[],
                                css=[
                                    {
                                        "selector": "table",
                                        "rule": "table-layout: fixed",
                                    }
                                ],
                                style_cell={
                                    "height": "auto",
                                    "width": "20%",
                                    "minWidth": "20%",
                                    "maxWidth": "20%",
                                    "whiteSpace": "normal",
                                },
                                sort_action="native",
                                sort_mode="multi",
                                filter_action="native",
                                row_selectable="multi",
                                page_current=0,
                                page_action="native",
                                page_size=50,
                            ),
                            width=10,
                        ),
                    ],
                    justify="center",
                )
            ),
        ]
    )

    external_stylesheets = [dbc.themes.FLATLY]

    app = DashFunc(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(
        children=[
            html.Div(header_text),
            html.Div(input_row),
            html.Hr(),
            html.Div(plot_header),
            html.Div(id="plots", children=None),
            html.Hr(),
            top_link,
            data_table,
            dcc.Store("target-synapse-json"),
            dcc.Store("source-synapse-json"),
            dcc.Store("target-table-json"),
            dcc.Store("source-table-json"),
        ]
    )

    @app.callback(
        Output("data-table", "selected_rows"),
        Input("reset-selection", "n_clicks"),
        Input("connectivity-tab", "value"),
    )
    def reset_selection(n_clicks, tab_value):
        return []

    @app.callback(
        Output("response-text", "children"),
        Input("submit-button", "n_clicks"),
        State("root_id", "value"),
    )
    def update_text(n_clicks, input_value):
        if len(input_value) == 0:
            return ""
        input_root_id = int(input_value)
        return f"  Running data for {input_root_id}..."

    @app.callback(
        Output("plots", "children"),
        Output("plot-response-text", "children"),
        Output("target-synapse-json", "data"),
        Output("source-synapse-json", "data"),
        Output("target-table-json", "data"),
        Output("source-table-json", "data"),
        Output("output-tab", "label"),
        Output("input-tab", "label"),
        Output("reset-selection", "n_clicks"),
        Input("submit-button", "n_clicks"),
        State("root_id", "value"),
    )
    def update_data(n_clicks, input_value):
        if len(input_value) == 0:
            return (
                html.Div("No plots to show yet"),
                "",
                [],
                [],
                [],
                [],
                "Output",
                "Input",
                1,
            )
        input_root_id = int(input_value)
        nrn_data = NeuronData(input_root_id, client=client)
        vfig = violin_fig(nrn_data, axon_color, dendrite_color, height=500, width=300)
        sfig = scatter_fig(nrn_data, valence_colors=val_colors, height=500)
        bfig = bar_fig(nrn_data, val_colors, height=500, width=500)

        pre_targ_df = nrn_data.pre_targ_df()[minimal_synapse_columns]
        pre_targ_df = stringify_root_ids(pre_targ_df)

        post_targ_df = nrn_data.post_targ_df()[minimal_synapse_columns]
        post_targ_df = stringify_root_ids(post_targ_df)

        return (
            dbc.Row(
                [
                    dcc.Graph(figure=vfig),
                    dcc.Graph(figure=sfig),
                    dcc.Graph(figure=bfig),
                ],
                justify="center",
                align="center",
                no_gutters=True,
            ),
            f"Data for {input_root_id}",
            pre_targ_df.to_dict("records"),
            post_targ_df.to_dict("records"),
            nrn_data.pre_tab_dat().to_dict("records"),
            nrn_data.post_tab_dat().to_dict("records"),
            f"Output (n = {pre_targ_df.shape[0]})",
            f"Input (n = {post_targ_df.shape[0]})",
            np.random.randint(10_000_000),
        )

    @app.callback(
        Output("data-table", "data"),
        Input("connectivity-tab", "value"),
        Input("target-table-json", "data"),
        Input("source-table-json", "data"),
    )
    def update_table(
        tab_value,
        pre_data,
        post_data,
    ):
        if tab_value == "tab-pre":
            return pre_data
        elif tab_value == "tab-post":
            return post_data
        else:
            return []

    @app.callback(
        Output("ngl_link", "href"),
        Input("connectivity-tab", "value"),
        Input("data-table", "derived_virtual_data"),
        Input("data-table", "derived_virtual_selected_rows"),
        Input("target-synapse-json", "data"),
        Input("source-synapse-json", "data"),
    )
    def update_link(
        tab_value, rows, selected_rows, syn_records_target, syn_records_source
    ):
        if rows is None or len(rows) == 0:
            rows = {}
            sb = generate_statebuilder(client)
            return sb.render_state(None, return_as="url")
        elif len(selected_rows) == 0:
            if tab_value == "tab-pre":
                syn_df = pd.DataFrame(syn_records_target)
                syn_df["pre_pt_root_id"] = syn_df["pre_pt_root_id"].astype(int)
                syn_df["post_pt_root_id"] = syn_df["post_pt_root_id"].astype(int)
                sb = generate_statebuilder_pre(client)
                return sb.render_state(syn_df, return_as="url")
            elif tab_value == "tab-post":
                syn_df = pd.DataFrame(syn_records_source)
                syn_df["pre_pt_root_id"] = syn_df["pre_pt_root_id"].astype(int)
                syn_df["post_pt_root_id"] = syn_df["post_pt_root_id"].astype(int)
                sb = generate_statebuilder_post(client)
            return sb.render_state(syn_df, return_as="url")
        else:
            dff = pd.DataFrame(rows)
            if tab_value == "tab-pre":
                return generate_url_synapses(
                    selected_rows, dff, pd.DataFrame(syn_records_target), "pre", client
                )
            elif tab_value == "tab-post":
                return generate_url_synapses(
                    selected_rows, dff, pd.DataFrame(syn_records_source), "post", client
                )

    return app