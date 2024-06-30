import numpy as np
from pulp import *
from collections import Counter
import pandas as pd
import signal
import time


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Tempo de execução excedido!")

# Carga de Dados
df_lojas = pd.read_csv('lojas.csv',delimiter='|')
atendimento = pd.read_csv('atendimento.csv', delimiter='|')
clusters = pd.read_excel('clusters.xlsx', sheet_name='CURITIBA')


clusters.rename(columns ={'Rua':'rua'}, inplace=True)
lojas_atendimento = df_lojas.merge(atendimento, how='inner', on = 'tipo_loja').drop_duplicates(subset='id_loja')
lojas_atendimento['tempo_visita_em_horas'] = lojas_atendimento['tempo_visita_em_horas'].str.replace(',', '.').astype('float')
lojas_atendimento_clusters = lojas_atendimento.merge(clusters, how = 'inner', on = 'rua')

#Conjuntos Iniciais

clusters = {ia:ib for ia, ib in Counter(lojas_atendimento_clusters["cluster"]).items() if ib > 1}

new_data = lojas_atendimento_clusters[(lojas_atendimento_clusters['cidade'] == 'CURITIBA') &
                           (lojas_atendimento_clusters['cluster'] == 13)].drop_duplicates(subset=['id_loja_x'])

lojas = new_data['id_loja_x'].to_list()
frequencia = new_data[['id_loja_x', 'freq_semanal']].rename(columns= {'id_loja_x':'id_loja'}).set_index('id_loja')
duracao = new_data[['id_loja_x', 'tempo_visita_em_horas']].rename(columns= {'id_loja_x':'id_loja'}).set_index('id_loja')
chd_max = 8
chs = 40

def math_model(agentes_num, relax=False):

    def reports(agentes, dias, x):
        solucao = pd.DataFrame([{'agente':ia, 'loja':ib, 'dias': ic, 'alocacao':x[(ia, ib, ic)].varValue} for ia in agentes for ib in lojas for ic in dias])
        agente_total = [{'agente': agente, 'chs':sum([duracao.loc[loja]['tempo_visita_em_horas']*x[(agente, loja, dia)].varValue for loja in lojas for dia in dias])} for agente in agentes]

        matriz_agente_dias = solucao.pivot_table(index='agente', columns='dias', values='alocacao', aggfunc=lambda x: sum(x))
        matriz_loja_dias = solucao.pivot_table(index='loja', columns='dias', values='alocacao', aggfunc=lambda x: sum(x))
        matriz_loja_agente = solucao.pivot_table(index='loja', columns= 'agente', values='alocacao', aggfunc=lambda x: sum(x))

        agentes_list = []
        for ia in agentes:
            agentes_set = {'agente':ia}
            for ib in dias:
                for ic in lojas:
                    if x[(ia, ic, ib)].varValue:
                        agentes_set[str(ib)] = str(ic)
                    elif (not x[(ia, ic, ib)].varValue) and (not agentes_set.get(str(ib))):
                        pass
            agentes_list.append(agentes_set)
            del agentes_set

        roteiro = pd.DataFrame(agentes_list)
        roteiro = roteiro.set_index('agente').fillna('-')

    # Conjuntos
    dias = list(range(1, 8))
    agentes = list(range(agentes_num))

    #Variáveis

    x = LpVariable.dicts('agente_loja_dia',((ia, ib, ic) for ia in agentes for ib in lojas for ic in dias),
                                  lowBound=0, upBound=1, cat='Integer')

    problem = LpProblem('domrock_problem', LpMaximize)

    problem += lpSum(x[(agente, loja, dia)] for agente in agentes for loja in lojas for dia in dias)/frequencia['freq_semanal'].sum()

    for dia in dias:
        for loja in lojas:
            problem += lpSum(x[(agente, loja, dia)] for agente in agentes) <= 1

    for agente in agentes:
            problem += lpSum(x[(agente, loja, dia)] for dia in dias for loja in lojas) <= 6

    for loja in lojas:
        if relax:
            problem += lpSum(x[(agente, loja, dia)] for agente in agentes for dia in dias) <= \
                       frequencia.loc[loja]['freq_semanal']
        else:
            problem += lpSum(x[(agente, loja, dia)] for agente in agentes for dia in dias) == \
                       frequencia.loc[loja]['freq_semanal']
    for agente in agentes:
        for dia in dias:
            problem += lpSum(duracao.loc[loja]['tempo_visita_em_horas']*x[(agente, loja, dia)] for loja in lojas) <= chd_max

    for agente in agentes:
        problem += lpSum(duracao.loc[loja]['tempo_visita_em_horas']*x[(agente, loja, dia)] for loja in lojas for dia in dias) <= chs

    for agente in agentes:
        for loja in lojas:
            for dia1, dia2 in [(ia, ia+1) for ia in dias if ia != 7]:
                problem += x[(agente, loja, dia1)] + x[(agente, loja, dia2)] <= 1

    # for agente in agentes:
    #     for loja in lojas:
    #         for dia0 , dia1, dia2 in [(ia-1, ia, ia+1) for ia in dias if ia != 7 and ia!= 1]:
    #             problem += x[(agente, loja, dia0)]+ x[(agente, loja, dia1)] + x[(agente, loja, dia2)] <= 1

    problem.solve()

    if problem.status == 1:
        reports(agentes, dias, x)
        return True, value(problem.objective)
    else:
        return False, 0


# Definir o manipulador de sinal
signal.signal(signal.SIGALRM, timeout_handler)

def executar_com_timeout(tempo, funcao, alternative_function):
    # Iniciar o alarme
    signal.alarm(tempo)

    try:
        resultado = funcao()
    except TimeoutException:
        print("Tempo de execução excedido!")
        print('Relaxando Modelo...')
        resultado = alternative_function(relax = True)
    finally:
        # Desativar o alarme
        signal.alarm(0)

    return resultado

def binary_search():
    inicio = 1

    while True:
        feasible, objective = math_model(inicio)
        if feasible:
            return inicio, objective
        else:
            inicio = inicio + 1

def binary_relaxed_search(relax):
    inicio = 1
    objective_vector = []
    while True:
        signal.alarm(200)
        try:
            feasible, objective = math_model(inicio, relax)
            objective_vector.append(objective)
            if objective >= 0.8:
                return inicio, objective
            else:
                inicio = inicio + 1
        except TimeoutException:
            print("Tempo de execução excedido!")
            return inicio, objective_vector[-1]
        finally:
            # Desativar o alarme
            signal.alarm(0)

tempo_limite = 200 #Tempo limite em segundos
numero_agentes, objetivo = executar_com_timeout(tempo_limite,
                                                binary_search,
                                                binary_relaxed_search)

print(numero_agentes, objetivo)

# report_agentes.append({'cidade': ix,
#                        'cluster': iy,
#                        'cobertura': objetivo,
#                        'total_lojas':len(lojas),
#                        'total_agentes':numero_agentes})
#
# df_report_agentes = pd.DataFrame(report_agentes)
# df_report_agentes_por_cidade = df_report_agentes.groupby(by='cidade').sum()[['total_lojas', 'total_agentes']]
#
# with pd.ExcelWriter('report_agentes_por_cidade.xlsx', mode='w',
#                     engine='openpyxl') as writer:
#     df_report_agentes.to_excel(writer, sheet_name='agentes_por_cluster')
#     df_report_agentes_por_cidade.to_excel(writer, sheet_name='agentes_por_cidade')