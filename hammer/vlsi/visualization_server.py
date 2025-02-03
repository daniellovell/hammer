"""
Hammer Manual Placement Constraints Visualization Tool

This script visualizes manual placement constraints from a Hammer YAML configuration file.
It parses the YAML file, extracts constraint information, and creates an interactive
visualization using Plotly and Dash.

The tool supports various constraint types including toplevel, hierarchical, hardmacro,
overlap, and obstruction. It can parse sizes from hierarchical references and LEF files
for hardmacros.

For hardmacro constraints without a 'master' field, it attempts to assign the 'master' by
parsing the SystemVerilog source files specified in the 'verilog_synth' entries in the YAML file.

Usage:
    python visualize_hammer_constraints.py <yaml_file> [--config-file <config_file>] [--top-cell-name <name>]

Dependencies:
    - yaml
    - plotly
    - dash
"""

import yaml
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import argparse
import logging
import re
from power_strapping.power_strapping import PowerStrapping
# Set up logging
logging.basicConfig(level=logging.INFO, format='\033[93m%(asctime)s - %(levelname)s - %(message)s\033[0m')
logger = logging.getLogger(__name__)


def parse_power_strap_constraints(config_path: str) -> list:
    """
    Parse power strap definitions from a YAML file and convert them to visualization constraints.
    
    The function handles YAML aliases and merges them appropriately. It creates two types of
    constraints:
    1. manual_power_strap: Represents the power strap routing area
    2. manual_power_strap_blockage: Represents the blockage area around the power strap
    
    Args:
        config_path (str): Path to the power strap YAML configuration file

    Returns:
        list: List of constraint dictionaries for power straps and their blockages
        
    Example constraint format:
    {
        'type': 'manual_power_strap',
        'path': 'strap_name',
        'x': float,
        'y': float,
        'width': float,
        'height': float,
        'nets': ['VSS', 'VDD']
    }
    """
    try:
        power_strapping = PowerStrapping(config_path)
        constraints = []
        
        for strap in power_strapping.config.get('power_straps', []):
            # Extract bbox coordinates
            bbox = strap['bbox']
            x, y = bbox[0], bbox[1]
            width = bbox[4] - bbox[0]  # x2 - x1
            height = bbox[5] - bbox[1]  # y2 - y1
            
            # Get nets from the first add_stripes entry that has them
            # The PowerStrapping class already handles YAML aliases
            nets = []
            if 'add_stripes' in strap:
                for stripe in strap['add_stripes']:
                    if 'nets' in stripe:
                        nets = stripe['nets']
                        break
            
            # Create power strap constraint
            strap_constraint = {
                'type': 'manual_power_strap',
                'path': strap['name'],
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'nets': nets
            }
            constraints.append(strap_constraint)
            
            # Create blockage constraint if present
            if 'route_blockage' in strap:
                blockage = strap['route_blockage']
                size = blockage.get('size', 0)
                
                blockage_constraint = {
                    'type': 'manual_power_strap_blockage',
                    'path': f"{strap['name']}_blockage",
                    'x': x - size,
                    'y': y - size,
                    'width': width + (2 * size),
                    'height': height + (2 * size),
                    'layers': blockage.get('layers', [])
                }
                constraints.append(blockage_constraint)
        
        logger.info(f"Parsed {len(constraints)} power strap constraints")
        return constraints
        
    except Exception as e:
        logger.error(f"Failed to parse power strap definitions: {e}")
        return []


def parse_yaml(file_path, top_cell_name, config):
    """
    Parse the YAML file and extract constraint information.

    Args:
        file_path (str): Path to the YAML file.
        top_cell_name (str): Name of the top cell to look for.
        config (dict): Visualization configuration.

    Returns:
        list: List of constraint dictionaries.
    """
    logger.info(f"Parsing YAML file: {file_path}")
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    def find_constraints(data, target_name):
        """Recursively search for constraints in the YAML data."""
        if isinstance(data, dict):
            if 'manual_placement_constraints' in data:
                for item in data['manual_placement_constraints']:
                    if target_name in item:
                        logger.info(f"Found constraints for {target_name}")
                        return item[target_name]
            for key, value in data.items():
                result = find_constraints(value, target_name)
                if result:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = find_constraints(item, target_name)
                if result:
                    return result
        return None
    # Collect Verilog files from 'verilog_synth' entries
    def get_verilog_files(data):
        verilog_files = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'verilog_synth':
                    if isinstance(value, list):
                        verilog_files.extend(value)
                    else:
                        verilog_files.append(value)
                else:
                    verilog_files.extend(get_verilog_files(value))
        elif isinstance(data, list):
            for item in data:
                verilog_files.extend(get_verilog_files(item))
        return verilog_files
    
        # Function to get LEF file map including supplemental LEFs
    def get_lef_file_map(data, supplemental_lefs):
        """Create a set of LEF file paths."""
        lef_files = set()

        # Extract LEF files from extra libraries
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'vlsi.technology.extra_libraries':
                    for lib_entry in value:
                        lib = lib_entry.get('library', {})
                        lef_file = lib.get('lef_file')
                        if lef_file:
                            lef_files.add(lef_file)
                else:
                    lef_files.update(get_lef_file_map(value, supplemental_lefs))
        elif isinstance(data, list):
            for item in data:
                lef_files.update(get_lef_file_map(item, supplemental_lefs))

        # Add supplemental LEF files from config
        lef_files.update(supplemental_lefs)

        return lef_files
    # Function to assign masters to hardmacros without a 'master'
    def assign_masters(constraints, instance_map):
        logger.info(instance_map)
        for constraint in constraints:
            if constraint.get('type') == 'hardmacro' and 'master' not in constraint:
                # Use the full path as the instance name
                path = constraint.get('path', '')
                instance_name = path  # Full hierarchical path
                master_name = instance_map.get(instance_name)
                if master_name:
                    constraint['master'] = master_name
                    logger.info(f"Assigned master '{master_name}' to constraint '{path}'")
                else:
                    logger.warning(f"Could not find master for instance '{instance_name}' in constraint '{path}'")
    # Function to parse sizes for constraints with 'master'
    def parse_master_sizes(constraints, data, macro_db):
        """
        Parse sizes for constraints with a 'master'.

        Args:
            constraints (list): List of constraint dictionaries.
            data (dict): YAML data.
            macro_db (dict): Macro database from LEF files.
        """
        for constraint in constraints:
            if 'master' in constraint:
                master_name = constraint['master']
                # First, try to find master in YAML
                master_constraints = find_constraints(data, master_name)
                if master_constraints:
                    master_toplevel = next((c for c in master_constraints if c.get('type') == 'toplevel'), None)
                    if master_toplevel:
                        constraint['width'] = master_toplevel.get('width', 0)
                        constraint['height'] = master_toplevel.get('height', 0)
                        constraint['margins'] = master_toplevel.get('margins', {})
                        logger.info(f"Updated constraint size from YAML master {master_name}: {constraint['width']}x{constraint['height']}")
                        continue
                    else:
                        for mc in master_constraints:
                            if 'width' in mc and 'height' in mc:
                                constraint['width'] = mc['width']
                                constraint['height'] = mc['height']
                                constraint['margins'] = mc.get('margins', {})
                                logger.info(f"Updated constraint size from YAML master {master_name}: {constraint['width']}x{constraint['height']}")
                                break
                # If not found in YAML, try LEF database
                if master_name in macro_db:
                    width, height = macro_db[master_name]
                    constraint['width'] = width
                    constraint['height'] = height
                    logger.info(f"Updated constraint size from LEF master {master_name}: {width}x{height}")
                else:
                    # Warn if size can't be determined
                    logger.warning(f"Could not determine size for constraint with master '{master_name}'")


    constraints = find_constraints(data, top_cell_name)
    if not constraints:
        logger.warning(f"No constraints found for top cell '{top_cell_name}'")
        return None


    verilog_files = get_verilog_files(data)
    instance_map = parse_verilog_files(verilog_files, top_cell_name)

    # Build LEF file map including supplemental LEFs from config
    supplemental_lefs = config.get('supplemental_lefs', [])
    lef_files = get_lef_file_map(data, supplemental_lefs)

    # Build macro database from LEFs
    macro_db = build_macro_database(lef_files)

    # Assign masters to hardmacros without 'master'
    assign_masters(constraints, instance_map)

    # Parse sizes for constraints with 'master'
    parse_master_sizes(constraints, data, macro_db)

    if 'manual_power_strap_definitions' in config:
        power_strap_constraints = parse_power_strap_constraints(
            config['manual_power_strap_definitions']
        )
        constraints.extend(power_strap_constraints)

    return constraints

def parse_verilog_files(verilog_files, top_module_name):
    """
    Parse SystemVerilog files to map hierarchical instance paths to module names.

    Args:
        verilog_files (list): List of paths to SystemVerilog files.
        top_module_name (str): Name of the top module.

    Returns:
        dict: Mapping of full instance paths to module names.
    """
    def remove_comments(code):
        """
        Remove comments from Verilog code.

        Args:
            code (str): Verilog code as a string.

        Returns:
            str: Code without comments.
        """
        # Remove block comments (/* ... */)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove line comments (// ...)
        code = re.sub(r'//.*', '', code)
        return code
    instance_map = {}
    module_definitions = {}

    # First, parse all modules and store their contents
    for file_path in verilog_files:
        logger.info(f"Parsing Verilog file: {file_path}")
        try:
            with open(file_path, 'r') as verilog_file:
                content = verilog_file.read()
                # Remove comments
                content = remove_comments(content)
                # Find all module definitions
                modules = re.findall(r'module\s+(\w+)\s*(#\s*\([^)]*\)\s*)?\s*\([^)]*\)\s*;([\s\S]*?)endmodule', content, re.MULTILINE)
                for module in modules:
                    module_name = module[0]
                    module_body = module[2]
                    module_definitions[module_name] = module_body
                    logger.info(f"Found module '{module_name}'")
        except FileNotFoundError:
            logger.warning(f"Verilog file not found: {file_path}")

    # Now, build the instance hierarchy starting from the top module
    def build_instance_hierarchy(module_name, parent_path):
        if module_name not in module_definitions:
            logger.warning(f"Module definition for '{module_name}' not found")
            return
        module_body = module_definitions[module_name]
        # Find all instances in the module body
        instances = re.findall(r'(\w+)\s+(\w+)\s*\(', module_body)
        logger.info(f"Found {len(instances)} instances in module '{module_name}'")
        for instance in instances:
            inst_module, inst_name = instance
            full_instance_path = '/'.join([parent_path, inst_name]) if parent_path else inst_name
            instance_map[full_instance_path] = inst_module
            logger.info(f"Found instance '{full_instance_path}' of module '{inst_module}'")
            # Recursively parse the instantiated module
            build_instance_hierarchy(inst_module, full_instance_path)

    # Start building from the top module
    build_instance_hierarchy(top_module_name, top_module_name)

    return instance_map


def build_macro_database(lef_files):
    """
    Build a database of macros and their sizes from LEF files.

    Args:
        lef_files (set): Set of LEF file paths.

    Returns:
        dict: Mapping of macro names to (width, height).
    """
    macro_db = {}
    for lef_file_path in lef_files:
        logger.info(f"Parsing LEF file: {lef_file_path}")
        try:
            with open(lef_file_path, 'r') as lef_file:
                lines = lef_file.readlines()
                macro_name = None
                for line in lines:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    if tokens[0] == 'MACRO':
                        macro_name = tokens[1]
                    elif tokens[0] == 'SIZE' and macro_name:
                        width = float(tokens[1])
                        height = float(tokens[3])
                        macro_db[macro_name] = (width, height)
                        logger.info(f"Added macro '{macro_name}' with size {width}x{height} to database")
                        macro_name = None  # Reset for next macro
        except FileNotFoundError:
            logger.warning(f"LEF file not found: {lef_file_path}")
    logger.info(f"Total macros in database: {len(macro_db)}")
    return macro_db

def create_figure(constraints, visible_constraints, filter_layers=None, filter_types=None, config={}):
    """
    Create a Plotly figure from the parsed constraints.

    Args:
        constraints (list): List of constraint dictionaries.
        visible_constraints (set): Set of constraint paths that are currently visible.
        filter_layers (list): List of layers to include in the visualization.
        filter_types (list): List of constraint types to include in the visualization.
        config (dict): Visualization configuration.

    Returns:
        go.Figure: Plotly figure object.
    """
    color_map = {
        'toplevel': 'gray',
        'hierarchical': 'blue',
        'overlap': 'yellow',
        'hardmacro': 'green',
        'obstruction': 'red',
        'manual_power_strap_blockage': 'orange',
        'manual_power_strap': 'purple'
    }

    fig = go.Figure()

    # Explicitly clear annotations and shapes
    fig.update_layout(annotations=[], shapes=[])

    # Determine aspect ratio and max dimensions based on 'toplevel' constraint
    toplevel = next((c for c in constraints if c.get('type') == 'toplevel'), None)
    if toplevel:
        toplevel_width = toplevel.get('width', 1)
        toplevel_height = toplevel.get('height', 1)
        aspect_ratio = toplevel_width / toplevel_height
        max_dim = max(toplevel_width, toplevel_height)
    else:
        aspect_ratio = 1
        max_dim = 1000  # Default value if no toplevel is found

    # Set font size range from config or use defaults
    min_font_size = config.get('font_size', {}).get('min', 8)
    max_font_size = config.get('font_size', {}).get('max', 16)

    # Initialize lists for shapes and hover traces
    shapes = []
    hover_traces = []

    for constraint in constraints:
        constraint_type = constraint.get('type')
        path = constraint.get('path', '')
        name = path.split('/')[-1]

        if filter_types and constraint_type not in filter_types:
            continue

        if path not in visible_constraints:
            continue

        x = constraint.get('x', 0)
        y = constraint.get('y', 0)
        width = constraint.get('width', 0)
        height = constraint.get('height', 0)
        margins = constraint.get('margins', {})

        if constraint_type == 'obstruction' or constraint_type == 'manual_power_strap_blockage':
            layers = constraint.get('layers', [])
            if filter_layers and not any(layer in filter_layers for layer in layers):
                continue

        color = color_map.get(constraint_type, 'black')

        # Add shape
        shapes.append(dict(
            type="rect",
            x0=x, y0=y, x1=x + width, y1=y + height,
            line=dict(color="black", width=2),
            fillcolor=color,
            opacity=0.5,
            layer='below',
            name=name
        ))

        # Calculate font size based on object size
        obj_size = max(width, height)
        font_size = max(min_font_size, min((obj_size / max_dim) * max_font_size, max_font_size))

        # Add text label
        fig.add_annotation(
            x=x + width / 2, y=y + height / 2,
            text=name,
            showarrow=False,
            font=dict(size=font_size),
            xanchor='center',
            yanchor='middle'
        )

        # Construct hovertext with additional information
        hovertext = (
            f"Type: {constraint_type}<br>"
            f"Name: {name}<br>"
            f"Position: ({x:.2f}, {y:.2f})<br>"
            f"Size: {width:.2f} x {height:.2f}"
        )
        if 'top_layer' in constraint:
            hovertext += f"<br>Top Layer: {constraint['top_layer']}"
        if 'layers' in constraint:
            hovertext += f"<br>Layers: {', '.join(constraint['layers'])}"

        # Add a scatter trace for hover interaction over the shape
        hover_traces.append(go.Scatter(
            x=[x, x + width, x + width, x, x],
            y=[y, y, y + height, y + height, y],
            fill='toself',
            opacity=0.0,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='text',
            text=hovertext,
            showlegend=False
        ))

    # Add shapes and hover traces to the figure
    fig.update_layout(shapes=shapes)
    for trace in hover_traces:
        fig.add_trace(trace)

    # Update axes and layout
    if config.get('grid', {}).get('enable', False):
        grid_spacing = config['grid'].get('spacing', 10)
        grid_color = config['grid'].get('color', 'lightgray')
        grid_opacity = config['grid'].get('opacity', 0.5)  # Get opacity from config or use default 0.5
        # Apply opacity to grid color if it's a named color
        if isinstance(grid_color, str):
            grid_color = f'rgba(211, 211, 211, {grid_opacity})'  # Convert lightgray to rgba
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            tick0=0,
            dtick=grid_spacing
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            tick0=0,
            dtick=grid_spacing,
            scaleanchor="x",
            scaleratio=1
        )
    else:
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title="Manual Placement Constraints Visualization",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=False,
        hovermode="closest",
        hoverdistance=10,  # Adjust hover radius
        autosize=True,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    logger.info(f"Number of shapes being drawn: {len(shapes)}")
    logger.info(f"Number of hover traces being added: {len(hover_traces)}")

    return fig

def create_app(config):
    """Create the Dash application for constraint visualization."""
    app = dash.Dash(__name__)

    # Initialize constraints data in dcc.Store
    app.layout = html.Div([
        dcc.Store(id='constraints-data'),
        # Main content area
        html.Div([
            # Plot
            html.Div([
                dcc.Graph(
                    id='constraint-plot',
                    config={
                        'scrollZoom': True,
                        'editable': False,
                    },
                    style={'height': '80vh', 'width': '100%'}  # Increased plot size
                ),
            ], style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'top'}),

            # Control panel
            html.Div([
                html.Button(
                    'Reload YAML', 
                    id='reload-button',
                    n_clicks=0,
                    style={
                        'width': '100%',
                        'margin': '10px 0',
                        'padding': '10px',
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer'
                    }
                ),
                html.H4("Filter Layers"),
                dcc.Checklist(
                    id='layer-filter',
                    style={'maxHeight': '200px', 'overflowY': 'auto'}
                ),
                html.H4("Filter Constraint Types"),
                dcc.Checklist(
                    id='type-filter',
                    style={'maxHeight': '200px', 'overflowY': 'auto'}
                ),
            ], style={
                'width': '23%', 
                'display': 'inline-block', 
                'vertical-align': 'top',
                'marginLeft': '2%',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),
        ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}),

        # Constraint table
        html.Div([
            dash_table.DataTable(
                id='constraint-table',
                columns=[
                    {
                        'name': 'Visible',
                        'id': 'visible',
                        'type': 'text',
                        'presentation': 'dropdown'
                    },
                    {'name': 'Path', 'id': 'path'},
                    {'name': 'Type', 'id': 'type'},
                    {'name': 'Top Layer', 'id': 'top_layer'},
                    {'name': 'Size', 'id': 'size'},
                    {'name': 'Position', 'id': 'position'}
                ],
                dropdown={
                    'visible': {
                        'options': [
                            {'label': 'True', 'value': 'True'},
                            {'label': 'False', 'value': 'False'}
                        ]
                    }
                },
                style_table={
                    'height': '200px',
                    'overflowY': 'auto',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'marginTop': '20px',
                    'width': '100%'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'backgroundColor': 'white',
                    'minWidth': '0px',
                    'maxWidth': '180px',
                    'whiteSpace': 'normal'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': 'bold',
                    'border': '1px solid #ddd'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f9f9f9'
                    }
                ],
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
            )
        ], style={
            'padding': '20px',
            'backgroundColor': '#ffffff',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginTop': '20px',
            'width': '96%',
            'marginLeft': '2%',
            'marginRight': '2%'
        }),
    ])

    # Callback to load constraints initially and when Reload YAML is clicked
    @app.callback(
        Output('constraints-data', 'data'),
        Output('layer-filter', 'options'),
        Output('layer-filter', 'value'),
        Output('type-filter', 'options'),
        Output('type-filter', 'value'),
        Input('reload-button', 'n_clicks')
    )
    def reload_constraints(n_clicks):
        logger.info("Loading constraints...")
        constraints = parse_yaml(args.yaml_file, args.top_cell_name, config)
        if constraints:
            logger.info(f"Loaded {len(constraints)} constraints")
        else:
            logger.warning(f"No constraints found for top cell '{args.top_cell_name}'")
            constraints = []

        # Get available constraint types and layers
        available_types = sorted(list(set(c['type'] for c in constraints)))
        type_options = [{'label': t.capitalize(), 'value': t} for t in available_types]

        available_layers = set()
        for c in constraints:
            if 'layers' in c:
                available_layers.update(c['layers'])
            if 'top_layer' in c:
                available_layers.add(c['top_layer'])
        layer_options = [{'label': f'Layer {layer}', 'value': layer} for layer in sorted(available_layers)]

        # Initialize all constraints as visible
        initial_visible_constraints = {c['path'] for c in constraints}

        return constraints, layer_options, [option['value'] for option in layer_options], type_options, available_types

    # Callback to update table data based on filters
    @app.callback(
        Output('constraint-table', 'data'),
        Input('layer-filter', 'value'),
        Input('type-filter', 'value'),
        Input('constraints-data', 'data')
    )
    def update_table(selected_layers, selected_types, constraints_data):
        if not constraints_data:
            return []

        constraints = constraints_data

        new_data = []
        for c in constraints:
            if c['type'] not in selected_types:
                continue

            if 'layers' in c and selected_layers:
                if not any(layer in selected_layers for layer in c['layers']):
                    continue
            if 'top_layer' in c and selected_layers:
                if c['top_layer'] not in selected_layers:
                    continue

            new_data.append({
                'visible': 'True',
                'path': c['path'],
                'type': c['type'],
                'top_layer': c.get('top_layer', ''),
                'size': f"{c.get('width', 0):.2f} x {c.get('height', 0):.2f}",
                'position': f"({c.get('x', 0):.2f}, {c.get('y', 0):.2f})"
            })

        return new_data

    # Callback to update figure based on table data and filters
    @app.callback(
        Output('constraint-plot', 'figure'),
        Input('constraint-table', 'data'),
        Input('constraints-data', 'data'),
        State('layer-filter', 'value'),
        State('type-filter', 'value')
    )
    def update_figure(table_data, constraints_data, selected_layers, selected_types):
        if not constraints_data or not table_data:
            return go.Figure()

        constraints = constraints_data
        visible_constraints = {row['path'] for row in table_data if row['visible'] == 'True'}

        filtered_constraints = []
        for c in constraints:
            c_type = c['type']
            c_layers = c.get('layers', [])
            if c_type not in selected_types:
                continue
            if c_layers and selected_layers:
                if not any(layer in selected_layers for layer in c_layers):
                    continue
            if 'top_layer' in c and selected_layers:
                if c['top_layer'] not in selected_layers:
                    continue
            filtered_constraints.append(c)

        fig = create_figure(filtered_constraints, visible_constraints, selected_layers, selected_types, config)
        return fig

    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize manual placement constraints from YAML file.')
    parser.add_argument('yaml_file', help='Path to the YAML file')
    parser.add_argument('--config-file', default='visualization_config.yaml', help='Visualization configuration file')
    parser.add_argument('--top-cell-name', default='ScumVTop', help='Name of the top cell')

    args = parser.parse_args()

    # Load visualization config
    config = {}
    try:
        with open(args.config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)
            logger.info(f"Loaded visualization config from {args.config_file}")
    except FileNotFoundError:
        logger.warning(f"Visualization config file not found: {args.config_file}. Using defaults.")

    app = create_app(config)
    app.run_server(debug=True)