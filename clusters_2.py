import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def ajustar_clusters(df, cluster_faturamento, limite=200000):
    while any(cluster_faturamento > limite):
        for cluster in cluster_faturamento.index:
            if cluster_faturamento[cluster] > limite:
                # Encontre lojas que podem ser movidas para outro cluster
                excess = cluster_faturamento[cluster] - limite
                lojas_excesso = df[df['cluster'] == cluster].sort_values(by='Faturamento', ascending=False)

                for index, row in lojas_excesso.iterrows():
                    # Tentar mover a loja para outro cluster
                    df.loc[index, 'cluster'] = encontrar_novo_cluster(df, row, cluster)
                    # Recalcular o faturamento do cluster
                    cluster_faturamento = df.groupby('cluster')['Faturamento'].sum()
                    if cluster_faturamento[cluster] <= limite:
                        break
    return df


def encontrar_novo_cluster(df, loja, cluster_atual):
    # Encontre o cluster mais próximo que não exceda o limite de faturamento
    for novo_cluster in df['cluster'].unique():
        if novo_cluster != cluster_atual:
            novo_faturamento = df[df['cluster'] == novo_cluster]['Faturamento'].sum() + loja['Faturamento']
            if novo_faturamento <= 200000:
                return novo_cluster
    # Se não houver clusters válidos, crie um novo cluster
    return df['cluster'].max() + 1


lojas = pd.read_excel('clusters.xlsx')

for i in lojas['Cidade'].drop_duplicates().to_list():
    df_lojas = lojas[(lojas['Cidade'] == i) & (lojas['Faturamento'] < 200000)]

    coordinates = df_lojas[['Latitude', 'Longitude']].values
    faturamento = df_lojas['Faturamento'].values

    kmeans = KMeans(n_clusters=int(len(df_lojas['id_loja'])/10), random_state=0).fit(coordinates)
    df_lojas['cluster'] = kmeans.labels_

    cluster_faturamento = df_lojas.groupby('cluster')['Faturamento'].sum()

    df_lojas = ajustar_clusters(df_lojas, cluster_faturamento)

    new_cluster_faturamento = df_lojas.groupby('cluster')['Faturamento'].sum()

    import matplotlib.pyplot as plt
    plt.hist(new_cluster_faturamento)
    plt.title('Histograma - Faturamento de Clusters - ' + i)
    plt.xlabel('Faturamento em R$')
    plt.ylabel('Frequência')
    plt.show()

    # with pd.ExcelWriter('clusters.xlsx', mode='a',
    #                     engine='openpyxl') as writer:
    #         df_lojas.to_excel(writer, sheet_name=i, index=False)