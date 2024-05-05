import pickle
from pathlib import Path
import pandas as pd
from prophet import Prophet
import holidays
import numpy as np

from optimizer import optimize_supplies

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/flights_prophet-{__version__}.pkl", "rb") as f:
    flights_model = pickle.load(f)

with open(f"{BASE_DIR}/sales_catboost-{__version__}.pkl", "rb") as f:
    sales_model = pickle.load(f)



mx_holidays = holidays.Mexico()

# Function to determine if a date is a holiday
def is_holiday(date):
    return date in mx_holidays


def process_data_flights(data):
    data['STD'] = pd.to_datetime(data['STD'])
    data['STA'] = pd.to_datetime(data['STA'])

    df_sorted_ = data.sort_values(by='STD')
    df_sorted = df_sorted_.reset_index(drop=True)

    df_sorted_filter = df_sorted[(df_sorted['STD'].dt.year == 2024) & (df_sorted['STD'].dt.month == 1)]
    df_output = df_sorted_filter.copy()

    df_sorted_filter = df_sorted_filter.rename(columns={'STD': 'ds'})
    df_sorted_filter = pd.get_dummies(df_sorted_filter, columns=['DepartureStation', 'ArrivalStation', 'Destination_Type', 'Origin_Type', 'tipo_vuelo'], drop_first=True)

    return df_output, df_sorted_filter

product_type = {'Carne Seca Habanero': 'Botanas',
 'Cheetos': 'Botanas',
 'Ruffles Queso': 'Botanas',
 'Coca Sin Azucar': 'Refrescos',
 'Jack And Coke': 'Licores',
 'Sprite': 'Refrescos',
 'Nissin Res': 'Sopas',
 'Tecate Light': 'Licores',
 'Mafer Sin Sal': 'Botanas',
 'Coca Cola Regular': 'Refrescos',
 'Ron Bacardi': 'Licores',
 'Arcoiris': 'Galletas',
 'Cafe 19 Chiapas': 'Bebidas Calientes',
 'Sabritas Originales': 'Botanas',
 'Xx Lager': 'Licores',
 'Jugo De Manzana': 'Refrescos',
 'Agua Natural 600 Ml': 'Refrescos',
 'Cafe Costa': 'Bebidas Calientes',
 'Amstel Ultra': 'Licores',
 'Panini Clasico': 'Perecederos',
 'Fanta De Naranja': 'Refrescos',
 'Nishikawa Japones': 'Botanas',
 'Sabritas Flamin Hot': 'Botanas',
 'Jw Red Label ': 'Licores',
 'Ciel Mineralizada': 'Refrescos',
 'Jugo De Mango': 'Refrescos',
 'Sidral Mundet': 'Refrescos',
 'Coca Cola Dieta': 'Refrescos',
 'Chokis': 'Galletas',
 'Tostitos': 'Botanas',
 'Mega Cuerno Clasico': 'Perecederos',
 'Doritos Nacho': 'Botanas',
 'Fritos Limon Y Sal': 'Botanas',
 'Corajillo Baileys ': 'Licores',
 'Quaker Avena Frutos Rojos': 'Galletas',
 'Nutty Berry Mix': 'Botanas',
 'Heineken Original': 'Licores',
 'Vino Tinto Sangre De Toro': 'Licores',
 'Luxury Nut Mix': 'Botanas',
 'Salsa Botanera': 'Botanas',
 'Jw Red Label': 'Licores',
 'Nissin Picante': 'Sopas',
 'Heineken Silver': 'Licores',
 'Leche De Fresa Sc': 'Lacteos',
 'Cheetos Flamin Hot': 'Botanas',
 'Emperador Chocolate': 'Galletas',
 'Cuerno Clasico De Pavo': 'Perecederos',
 'Nissin Dark Dragon': 'Sopas',
 'Nissin Fuego': 'Sopas',
 'Panini Integral': 'Perecederos',
 'Cafe 19 Capuchino': 'Bebidas Calientes',
 'Te Manzanilla Jengibre': 'Bebidas Calientes',
 'Xx Ultra': 'Licores',
 'Sol Clamato': 'Licores',
 'Go Nuts': 'Botanas',
 'Muffin Integral': 'Galletas',
 'Dip De Queso': 'Botanas',
 'Hazme Doble': 'OFERTAS ',
 'Baileys': 'Licores',
 'Nishikawa Salado': 'Botanas',
 'Corajillo': 'Licores',
 'Quaker Granola': 'Galletas',
 'Tequila 7 Leguas Reposado': 'Licores',
 'Emperador Vainilla': 'Galletas',
 'Leche De Chocolate Sc': 'Lacteos',
 'Arandano Mango Mix': 'Botanas',
 'Topochico Seltzer Mango': 'Licores',
 'Rancheritos': 'Botanas',
 'Baileys ': 'Licores',
 'Protein Adventure': 'Botanas',
 'Tequila 7 Leguas Blanco': 'Licores',
 'Nueces De Arbol Mix': 'Botanas',
 'Cafe De Olla': 'Bebidas Calientes',
 'Te Vainilla': 'Bebidas Calientes',
 'Tostitos Nachos Con Dip': 'Botanas',
 'Frutos Secos Enchilados': 'Botanas',
 'Hsbc-Viva': 'OFERTAS ',
 'Ultra Seltzer Frambuesa': 'Licores',
 'Arandano': 'Botanas',
 'Te Frutos Rojos': 'Bebidas Calientes',
 'Vino Tinto Cria Cuervos': 'Licores',
 'Carne Seca Original': 'Botanas',
 'Te Relax': 'Bebidas Calientes',
 'Vino Blanco Cria Cuervos ': 'Licores',
 'Topochico Seltzer Fresa-Guayaba': 'Licores',
 'Galleta De Arandano Relleno De Q/Crema': 'Galletas',
 'Galleta De Chispas De Chocolate': 'Galletas',
 'Promo Hsbc 1 Bebida Gratis': 'OFERTAS ',
 'Galleta De Chocolate': 'Galletas',
 'Cerveza Charter': 'Licores',
 'Eco Holder': 'Botanas',
 'Cafe 19 Cafe Clasico': 'Bebidas Calientes',
 'Gomita Enchilada La Cueva': 'Botanas',
 'Maxi Combo': 'Perecederos',
 'Heineken 0': 'Licores',
 'Combo Stl': 'Licores',
 'Kacang Flaming Hot': 'Botanas',
 'Licor Charter': 'Licores',
 'Quaker Avena Moras': 'Galletas',
 'Quaker Natural Balance': 'Galletas',
 'Nissin Limon Y Habanero': 'Sopas'}

products = ['Carne Seca Habanero', 'Cheetos', 'Ruffles Queso',
       'Coca Sin Azucar', 'Jack And Coke', 'Sprite', 'Nissin Res',
       'Tecate Light', 'Mafer Sin Sal', 'Coca Cola Regular',
       'Ron Bacardi', 'Arcoiris', 'Cafe 19 Chiapas',
       'Sabritas Originales', 'Xx Lager', 'Jugo De Manzana',
       'Agua Natural 600 Ml', 'Cafe Costa', 'Amstel Ultra',
       'Panini Clasico', 'Fanta De Naranja', 'Nishikawa Japones',
       'Sabritas Flamin Hot', 'Jw Red Label ', 'Ciel Mineralizada',
       'Jugo De Mango', 'Sidral Mundet', 'Coca Cola Dieta', 'Chokis',
       'Tostitos', 'Mega Cuerno Clasico', 'Doritos Nacho',
       'Fritos Limon Y Sal', 'Corajillo Baileys ',
       'Quaker Avena Frutos Rojos', 'Nutty Berry Mix',
       'Heineken Original', 'Vino Tinto Sangre De Toro', 'Luxury Nut Mix',
       'Salsa Botanera', 'Jw Red Label', 'Nissin Picante',
       'Heineken Silver', 'Leche De Fresa Sc', 'Cheetos Flamin Hot',
       'Emperador Chocolate', 'Cuerno Clasico De Pavo',
       'Nissin Dark Dragon', 'Nissin Fuego', 'Panini Integral',
       'Cafe 19 Capuchino', 'Te Manzanilla Jengibre', 'Xx Ultra',
       'Sol Clamato', 'Go Nuts', 'Muffin Integral', 'Dip De Queso',
       'Hazme Doble', 'Baileys', 'Nishikawa Salado', 'Corajillo',
       'Quaker Granola', 'Tequila 7 Leguas Reposado',
       'Emperador Vainilla', 'Leche De Chocolate Sc',
       'Arandano Mango Mix', 'Topochico Seltzer Mango', 'Rancheritos',
       'Baileys ', 'Protein Adventure', 'Tequila 7 Leguas Blanco',
       'Nueces De Arbol Mix', 'Cafe De Olla', 'Te Vainilla',
       'Tostitos Nachos Con Dip', 'Frutos Secos Enchilados', 'Hsbc-Viva',
       'Ultra Seltzer Frambuesa', 'Arandano', 'Te Frutos Rojos',
       'Vino Tinto Cria Cuervos', 'Carne Seca Original', 'Te Relax',
       'Vino Blanco Cria Cuervos ', 'Topochico Seltzer Fresa-Guayaba',
       'Galleta De Arandano Relleno De Q/Crema',
       'Galleta De Chispas De Chocolate', 'Promo Hsbc 1 Bebida Gratis',
       'Galleta De Chocolate', 'Cerveza Charter', 'Eco Holder',
       'Cafe 19 Cafe Clasico', 'Gomita Enchilada La Cueva', 'Maxi Combo',
       'Heineken 0', 'Combo Stl', 'Kacang Flaming Hot', 'Licor Charter',
       'Quaker Avena Moras', 'Quaker Natural Balance',
       'Nissin Limon Y Habanero']

def predict_sales(data):


    data = data[['Flight_ID', 'Aeronave', 'Capacity', 'DepartureStation',
       'ArrivalStation', 'Destination_Type', 'Origin_Type', 'STD', 'STA',
       'Passengers', 'tipo_vuelo']]
    
    data['STD'] = pd.to_datetime(data['STD'])  # Make sure the STD column is in datetime format
    # Create a new column 'holidays' based on whether the date in 'STD' is a holiday
    data['holidays'] = data['STD'].apply(is_holiday).astype(int)  # Convert boolean to int (1 for True, 0 for False)
    data['month'] = data['STD'].dt.month
    data['day_of_week'] = data['STD'].dt.dayofweek
    data['hour'] = data['STD'].dt.hour
    data['minute'] = data['STD'].dt.minute  # Extracting minute
    # Define categorical features for CatBoost
    # Splitting the data
    data = data.drop(['STA','Flight_ID', 'Aeronave'], axis=1)

    # Repetir cada fila len(products) veces
    repeated_df = data.loc[data.index.repeat(len(products))].reset_index(drop=True)

    # Asignar productos a cada fila repetida
    repeated_df['ProductName'] = products * len(data)
    repeated_df['ProductType'] = repeated_df['ProductName'].map(product_type)

    date_col = repeated_df['STD']
    repeated_df = repeated_df.drop(['STD'], axis=1)

    cat_features = ['DepartureStation', 'ArrivalStation', 'Destination_Type', 'Origin_Type', 'tipo_vuelo', 'ProductName', 'ProductType']
    for feature in cat_features:
        repeated_df[feature] = repeated_df[feature].astype('category')

    repeated_df = repeated_df[['ProductType', 'ProductName', 'Capacity', 'DepartureStation',
        'ArrivalStation', 'Destination_Type', 'Origin_Type', 'Passengers',
        'tipo_vuelo', 'holidays', 'month', 'day_of_week', 'hour', 'minute']]

    # Now, predict using the model
    predictions = sales_model.predict(repeated_df)
    predictions = np.floor(predictions)

    repeated_df['Quantity'] = predictions
    repeated_df['STD'] = date_col

    df = repeated_df.copy()  # Cambia esto por tu DataFrame
    # Supongamos que df es tu DataFrame y 'columna_especifica' es la columna a verificar
    columna_especifica = 'Quantity'  # Cambia esto por el nombre real de tu columna

    # Calcular el número total de registros a eliminar (60% del total del DataFrame)
    num_total_eliminar = int(len(df) * 0.60)

    # Filtrar los registros donde los valores son menores a 3
    datos_filtrados = df[df[columna_especifica] < 3]

    # Determinar cuántos registros eliminar del subconjunto filtrado
    if len(datos_filtrados) >= num_total_eliminar:
        # Si hay suficientes registros en el subconjunto, elimina el número necesario de ahí
        indices_a_eliminar = np.random.choice(datos_filtrados.index, num_total_eliminar, replace=False)
    else:
        # Si no hay suficientes registros, elimina todos del subconjunto y necesitas otro criterio para el resto
        indices_a_eliminar = datos_filtrados.index
        # Aquí podrías considerar otro criterio para seleccionar registros adicionales a eliminar

    # Eliminar esos registros del DataFrame original
    df = df.drop(indices_a_eliminar)

    return df



def predict_flights(data):
    data_output, data = process_data_flights(data)
    pred_df = flights_model.predict(data)

    data['Passengers'] = pred_df['yhat']
    data_output['Passengers'] = data['Passengers']

    data_output['Passengers'] = data_output['Passengers'].round(0).astype(int)

    return data_output

def predict(data):

    flights = predict_flights(data)
    sales = predict_sales(flights)

    return sales


def optimize_data(data):
    return optimize_supplies(data)


    
