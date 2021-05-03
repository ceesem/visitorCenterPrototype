import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash import Dash
from jupyter_dash import JupyterDash

from .link_utilities import generate_statebuilder, generate_statebuilder_pre
from .neuron_data_base import NeuronCardData, table_columns
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

    sb = generate_statebuilder(client)
    sb_pre = generate_statebuilder_pre(client)

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
                        style={"font-size": "24px"},
                    ),
                ],
                width=10,
            )
        ],
        justify="center",
    )

    data_table = dbc.Row(
        [
            dbc.Col(
                dash_table.DataTable(
                    id="data-table",
                    columns=[{"name": i, "id": i} for i in table_columns],
                    data=[],
                    css=[{"selector": "table", "rule": "table-layout: fixed"}],
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
            dcc.Store("all-synapse-json"),
            dcc.Store("target-synapse-json"),
        ]
    )

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
        Output("data-table", "data"),
        Output("target-synapse-json", "data"),
        Output("all-synapse-json", "data"),
        Input("submit-button", "n_clicks"),
        State("root_id", "value"),
    )
    def update_data(n_clicks, input_value):
        if len(input_value) == 0:
            return html.Div("No plots to show yet"), "", [], [], []
        input_root_id = int(input_value)
        nrn_data = NeuronCardData(input_root_id, client=client)
        vfig = violin_fig(nrn_data, axon_color, dendrite_color, height=500, width=300)
        sfig = scatter_fig(nrn_data, valence_colors=val_colors, height=500)
        bfig = bar_fig(nrn_data, val_colors, height=500, width=500)

        pre_targ_df = nrn_data.pre_targ_df[
            ["pre_pt_root_id", "post_pt_root_id", "ctr_pt_position"]
        ]
        pre_targ_df["post_pt_root_id"] = pre_targ_df["post_pt_root_id"].astype(str)
        pre_targ_df["pre_pt_root_id"] = pre_targ_df["pre_pt_root_id"].astype(str)

        syn_df = nrn_data.syn_df.query('direction == "pre"')[
            ["pre_pt_root_id", "post_pt_root_id", "ctr_pt_position"]
        ]
        syn_df["post_pt_root_id"] = syn_df["post_pt_root_id"].astype(str)
        syn_df["pre_pt_root_id"] = syn_df["pre_pt_root_id"].astype(str)
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
            nrn_data.tab_dat.to_dict("records"),
            pre_targ_df.to_dict("records"),
            syn_df.to_dict("records"),
        )

    @app.callback(
        Output("ngl_link", "href"),
        Input("data-table", "derived_virtual_data"),
        Input("data-table", "derived_virtual_selected_rows"),
        Input("all-synapse-json", "data"),
        Input("target-synapse-json", "data"),
        prevent_initial_call=True,
    )
    def update_link(rows, selected_rows, syn_records_all, syn_records_target):
        if rows is None or len(rows) == 0:
            rows = {}
            return sb.render_state(None, return_as="url")
        elif len(selected_rows) == 0:
            syn_df = pd.DataFrame(syn_records_all)
            syn_df["pre_pt_root_id"] = syn_df["pre_pt_root_id"].astype(int)
            syn_df["post_pt_root_id"] = syn_df["post_pt_root_id"].astype(int)
            return sb_pre.render_state(syn_df, return_as="url")
        else:
            dff = pd.DataFrame(rows)
            dff["post_pt_root_id"] = dff["post_pt_root_id"].astype(np.int64)
            post_oids = dff.loc[selected_rows]["post_pt_root_id"].values

            pre_targ_df = pd.DataFrame(syn_records_target)
            pre_targ_df["pre_pt_root_id"] = pre_targ_df["pre_pt_root_id"].astype(int)
            pre_targ_df["post_pt_root_id"] = pre_targ_df["post_pt_root_id"].astype(int)
            return sb.render_state(
                pre_targ_df.query("post_pt_root_id in @post_oids"), return_as="url"
            )

    return app