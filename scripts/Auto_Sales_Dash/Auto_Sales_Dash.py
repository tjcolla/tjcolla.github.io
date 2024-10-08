!pip install dash
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the data using pandas
data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# List of years
year_list = [i for i in range(1980, 2024, 1)]

# Create the layout of the app
app.layout = html.Div([
    # Title
    html.H1('Automobile Sales Statistics Dashboard', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 24}),
    
    # Dropdown menus
    html.Div([
        html.Label("Select Statistics:"),
        dcc.Dropdown(id='dropdown-statistics',
                     options=[
                         {'label': 'Yearly Statistics', 'value': 'Yearly Statistics'},
                         {'label': 'Recession Period Statistics', 'value': 'Recession Period Statistics'}
                     ],
                     value='Select Statistics', # Set default value
                     placeholder='Select a report type', # Set placeholder
                     style={
                         'width': '100%',
                         'padding': '3px',
                         'fontSize': '20px',
                         'textAlignLast': 'center'
                     })
    ]),
    
    html.Div(dcc.Dropdown(
        id='select-year',
        options=[{'label': i, 'value': i} for i in year_list],
        value='Select Year',
        placeholder='Select Year',
        style={
            'width': '100%',
            'padding': '3px',
            'fontSize': '20px',
            'textAlignLast': 'center'
        }
    )),
    
    # Output container
    html.Div(id='output-container', className='chart-grid',
             style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'overflowX': 'auto'})
])

# Callbacks for interactivity
@app.callback(
    Output(component_id='select-year', component_property='disabled'),
    Input(component_id='dropdown-statistics', component_property='value')
)
def update_input_container(selected_statistics):
    if selected_statistics == 'Yearly Statistics':
        return False
    else:
        return True

@app.callback(
    Output(component_id='output-container', component_property='children'),
    [Input(component_id='select-year', component_property='value'),
     Input(component_id='dropdown-statistics', component_property='value')]
)
def update_output_container(input_year, selected_statistics):
    if selected_statistics == 'Recession Period Statistics':
        recession_data = data[data['Recession'] == 1]

        # Plot 1: Automobile sales fluctuate over Recession Period (year wise)
        yearly_rec = recession_data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        R_chart1 = dcc.Graph(
            figure=px.line(yearly_rec, x='Year', y='Automobile_Sales',
                           title="Average Automobile Sales fluctuation over Recession Period")
            .update_yaxes(title_text='Automobile Sales'))

        # Plot 2: Average number of vehicles sold by vehicle type
        average_sales = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        R_chart2 = dcc.Graph(figure=px.bar(average_sales, x='Vehicle_Type', y='Automobile_Sales',
                                           title="Average Vehicles Sold by Vehicle Type during Recession Periods")
                             .update_yaxes(title_text='Automobile Sales')
                             .update_xaxes(title_text='Vehicle Type'))

        # Plot 3: Pie chart for total expenditure share by vehicle type during recessions
        exp_rec = recession_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        R_chart3 = dcc.Graph(figure=px.pie(exp_rec, values='Advertising_Expenditure', names='Vehicle_Type',
                                           title="Total Advertisement Expenditure for each vehicle during Recession Period"))

        # Plot 4: Bar chart for the effect of unemployment rate on vehicle type and sales
        unemployment_effect = recession_data.groupby(['Vehicle_Type', 'unemployment_rate'])['Automobile_Sales'].mean().reset_index()
        R_chart4 = dcc.Graph(
            figure=px.bar(unemployment_effect, x='unemployment_rate', y='Automobile_Sales', color='Vehicle_Type',
                          title="Effect of Unemployment Rate on Vehicle Type and Sales during Recessions")
            .update_xaxes(title_text='Unemployment Rate')
            .update_yaxes(title_text='Automobile Sales'))

        return [
            html.Div(className='chart-item', children=[html.Div(children=R_chart1), html.Div(children=R_chart2)], style={'display': 'flex'}),
            html.Div(className='chart-item', children=[html.Div(children=R_chart3), html.Div(children=R_chart4)], style={'display': 'flex'})
        ]

    elif input_year and selected_statistics == 'Yearly Statistics':
        yearly_data = data[data['Year'] == input_year]

        # Plot 1: Yearly Automobile sales using line chart for the whole period
        yas = data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        Y_chart1 = dcc.Graph(figure=px.line(yas, x='Year', y='Automobile_Sales',
                                            title="Yearly Automobile Sales")
                             .update_yaxes(title_text='Automobile Sales'))

        # Plot 2: Total Monthly Automobile sales using line chart
        monthly_sales = yearly_data.groupby('Month')['Automobile_Sales'].sum().reset_index()
        Y_chart2 = dcc.Graph(figure=px.line(monthly_sales, x='Month', y='Automobile_Sales',
                                            title=f'Total Monthly Automobile Sales in {input_year}')
                             .update_yaxes(title_text='Automobile Sales'))

        # Plot 3: Bar chart for average number of vehicles sold during the given year
        avr_vdata = yearly_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        Y_chart3 = dcc.Graph(figure=px.bar(avr_vdata, x='Vehicle_Type', y='Automobile_Sales',
                                           title=f'Average Vehicles Sold by Vehicle Type in {input_year}')
                             .update_xaxes(title_text='Vehicle Type')
                             .update_yaxes(title_text='Automobile Sales'))

        # Plot 4: Pie chart for Total Advertisement Expenditure for each vehicle
        exp_data = yearly_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        Y_chart4 = dcc.Graph(figure=px.pie(exp_data, values='Advertising_Expenditure', names='Vehicle_Type',
                                           title=f"Total Advertisement Expenditure for each vehicle in {input_year}"))

        return [
            html.Div(className='chart-item', children=[html.Div(children=Y_chart1), html.Div(children=Y_chart2)], style={'display': 'flex'}),
            html.Div(className='chart-item', children=[html.Div(children=Y_chart3), html.Div(children=Y_chart4)], style={'display': 'flex'})
        ]

    else:
        return None

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
