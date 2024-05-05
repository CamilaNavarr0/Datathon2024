import pandas as pd

def optimize_supplies(data): 
    max_capacity = {
        'Carne Seca Habanero': 5,
        'Cheetos': 2,
        'Ruffles Queso': 23,
        'Coca Sin Azucar': 10,
        'Jack And Coke': 8,
        'Sprite': 6,
        'Nissin Res': 6,
        'Coca Cola Regular': 44,
        'Ron Bacardi': 2,
        'Arcoiris': 2,
        'Cafe 19 Chiapas': 10,
        'Sabritas Originales': 23,
        'Jugo De Manzana': 3,
        'Agua Natural 600 Ml': 40,
        'Cafe Costa': 25,
        'Amstel Ultra': 10,
        'Fanta De Naranja': 2,
        'Nishikawa Japones': 2,
        'Sabritas Flamin Hot': 12,
        'Ciel Mineralizada': 4,
        'Jugo De Mango': 3,
        'Sidral Mundet': 2,
        'Coca Cola Dieta': 8,
        'Chokis': 3,
        'Tostitos': 2,
        'Doritos Nacho': 23,
        'Fritos Limon Y Sal': 23,
        'Corajillo Baileys ': 2,  # Espacio extra al final, verifica si es correcto.
        'Nutty Berry Mix': 1,
        'Heineken Original': 6,
        'Luxury Nut Mix': 1,
        'Salsa Botanera': 50,
        'Nissin Picante': 8,
        'Cheetos Flamin Hot': 12,
        'Emperador Chocolate': 2,
        'Nissin Dark Dragon': 11,
        'Nissin Fuego': 8,
        'Cafe 19 Capuchino': 10,
        'Te Manzanilla Jengibre': 4,
        'Go Nuts': 1,
        'Nishikawa Salado': 2,
        'Corajillo': 2,
        'Tequila 7 Leguas Reposado': 3,
        'Arandano Mango Mix': 1,
        'Topochico Seltzer Mango': 2,
        'Tequila 7 Leguas Blanco': 3,
        'Nueces De Arbol Mix': 1,
        'Frutos Secos Enchilados': 1,
        'Te Frutos Rojos': 4,
        'Vino Tinto Cria Cuervos': 3,
        'Carne Seca Original': 5,
        'Vino Blanco Cria Cuervos ': 2
    }

    supply_points = [
        "AO",  # General replenishment
        "AU",  # General replenishment
        "AW",  # General replenishment
        "BA",  # General replenishment
        "BM",  # General replenishment
        "AT",  # General replenishment
        "AK",  # General replenishment
        "AD"   # Only fresh food is replenished here
    ]

    pernoct_points = ['AO', 'AU', 'AW', 'BA', 'BM', 'AT', 'AD', 'AK']

    # Calculating average quantity sold per product per flight and by departure station

    # Group by ProductName and DepartureStation to calculate the average quantity sold
    average_demand = data.groupby(['ProductName', 'DepartureStation'])['Quantity'].mean().reset_index()

    # Display the result to understand the average demand per product per station
    average_demand.head(10)

    # Convert the STD (Scheduled Time of Departure) column to datetime format for analysis
    data['STD'] = pd.to_datetime(data['STD'])

    # Extract hour from the STD datetime column to analyze the distribution of flights over the day
    data['Hour_of_Day'] = data['STD'].dt.hour

    # To adjust the analysis, we'll first need to count unique flights per hour at each station

    # Count unique Flight_IDs by DepartureStation and Hour_of_Day
    unique_flights_schedule = data.groupby(['DepartureStation', 'Hour_of_Day'])['Flight_ID'].nunique().unstack(fill_value=0)

    # Display the unique flight count by hour for each station
    unique_flights_schedule.head()

    # Merge the average demand data with the unique flight schedule data
    # This requires a join between the average_demand table and the unique_flights_schedule table, based on the DepartureStation

    # We need to first transform the unique_flights_schedule back to a long format
    unique_flights_schedule_long = unique_flights_schedule.stack().reset_index()
    unique_flights_schedule_long.columns = ['DepartureStation', 'Hour_of_Day', 'UniqueFlights']

    # Join average_demand with the transformed unique flight schedule data
    replenishment_model_data = pd.merge(average_demand, unique_flights_schedule_long, on='DepartureStation', how='left')

    # Filter to match the hours (only keep rows where the product demand matches the hours flights are present)
    replenishment_model_data = replenishment_model_data[replenishment_model_data['Hour_of_Day'] == replenishment_model_data['Hour_of_Day']]

    # Calculate the required stock per product based on unique flights and average quantity
    replenishment_model_data['RequiredStock'] = replenishment_model_data['Quantity'] * replenishment_model_data['UniqueFlights']

    # Display the proposed replenishment model data
    replenishment_model_data.head(10)

    # Implementing the full replenishment model across all products and stations

    # Calculate the required stock for each product, for each unique flight in each station and hour
    full_replenishment_model = pd.merge(average_demand, unique_flights_schedule_long, on='DepartureStation', how='left')
    full_replenishment_model['RequiredStock'] = full_replenishment_model['Quantity'] * full_replenishment_model['UniqueFlights']

    # Filter to keep only the rows where flights are available at those hours (where UniqueFlights > 0)
    full_replenishment_model = full_replenishment_model[full_replenishment_model['UniqueFlights'] > 0]

    # Display a more comprehensive view of the full replenishment model
    full_replenishment_model.sort_values(by=['DepartureStation', 'Hour_of_Day']).head(20)


    # Adjust the replenishment model to include these capacity constraints
    full_replenishment_model['MaxAllowed'] = full_replenishment_model['ProductName'].map(max_capacity)
    full_replenishment_model['AdjustedRequiredStock'] = full_replenishment_model.apply(
        lambda x: min(x['RequiredStock'], x['MaxAllowed']), axis=1
    )

    full_replenishment_model = full_replenishment_model.dropna()

    return full_replenishment_model

