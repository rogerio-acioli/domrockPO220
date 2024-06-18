import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


def clean_data(df):
    # Remove espaços em branco e caracteres invisíveis das strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def create_clusters(file_path):
    # Leitura da planilha
    df = pd.read_excel(file_path)

    # Limpeza dos dados
    df = clean_data(df)

    # Verificação de valores nulos
    if df.isnull().values.any():
        null_columns = df.columns[df.isnull().any()].tolist()
        null_rows = df[df.isnull().any(axis=1)].index.tolist()
        raise ValueError(f"A planilha contém valores nulos nas colunas: {null_columns} nas linhas: {null_rows}")

    # Verificação da existência da coluna 'ID'
    if 'ID' not in df.columns:
        raise KeyError("A coluna 'ID' não foi encontrada na planilha.")

    # Lista para armazenar resultados
    results = []

    # Contador global de clusters
    global_cluster_counter = 0

    # Loop através das cidades únicas
    for city in df['Cidade'].unique():
        city_df = df[df['Cidade'] == city].copy()
        print(f"Processando a cidade: {city}")

        if city_df.empty:
            print(f"Nenhuma loja encontrada na cidade: {city}")
            continue

        # Lojas que faturam >= 200,000 não têm clusters
        high_revenue_stores = city_df[city_df['Faturamento'] >= 200000].copy()
        high_revenue_stores['Cluster'] = ""

        # Lojas com faturamento < 200,000
        low_revenue_stores = city_df[city_df['Faturamento'] < 200000].copy()

        # Se não houver lojas de baixo faturamento, pula para a próxima cidade
        if low_revenue_stores.empty:
            results.append(high_revenue_stores)
            continue

        # Preparação para o KMeans
        low_revenue_stores = low_revenue_stores.sort_values(by='Faturamento', ascending=False).reset_index(drop=True)

        # Definir o número de clusters baseado no total do faturamento dividido por 200,000
        total_faturamento = low_revenue_stores['Faturamento'].sum()
        num_clusters = max(1, int(total_faturamento / 200000) + 1)

        # Coordenadas para KMeans
        coordinates = low_revenue_stores[['Latitude', 'Longitude']]

        # Aplicar KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(coordinates)

        # Atribuir clusters iniciais aos dados
        low_revenue_stores['Cluster'] = labels

        # Ajustar clusters para que a soma do faturamento não exceda R$200.000,00
        adjusted_clusters = []
        current_cluster = []
        current_sum = 0
        cluster_counter = 0

        for _, row in low_revenue_stores.iterrows():
            if current_sum + row['Faturamento'] > 200000:
                adjusted_clusters.append((global_cluster_counter + cluster_counter, pd.DataFrame(current_cluster)))
                cluster_counter += 1
                current_cluster = []
                current_sum = 0
            current_cluster.append(row)
            current_sum += row['Faturamento']

        if current_cluster:
            adjusted_clusters.append((global_cluster_counter + cluster_counter, pd.DataFrame(current_cluster)))

        global_cluster_counter += cluster_counter + 1

        # Concatenando resultados
        for cluster_id, cluster_df in adjusted_clusters:
            cluster_df['Cluster'] = cluster_id
            results.append(cluster_df)

        results.append(high_revenue_stores)

    # Concatenando todas as cidades
    final_df = pd.concat(results)

    # Salvando em uma nova planilha de Excel
    final_df.to_excel('C:/Users/roger/Downloads/clusters_output.xlsx', index=False)


# Caminho para a planilha de entrada
file_path = 'C:/Users/roger/Downloads/lojas_faturamento.xlsx'

# Chamando a função para criar clusters
create_clusters(file_path)
