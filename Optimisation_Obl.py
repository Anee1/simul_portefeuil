import numpy as np
import numpy_financial as npf
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from itertools import combinations

def perioderest(df):
    format_date = '%Y-%m-%d'
    date_fin = datetime.strptime(df['DateFin'], format_date)
    today = datetime.today()
    diff_days = (date_fin - today).days

    periodicite = df['Periodicite']
    if periodicite == 'Annuel':
        return diff_days / 360
    elif periodicite == 'Semestriel':
        return diff_days / 182
    elif periodicite == 'Trimestriel':
        return diff_days / 90
    else:
        return diff_days / 30

def calc_diff(df):
    res = []
    for i in range(len(df)):
        differe = df.iloc[i]['NbreDifferes']
        periodicite = df.iloc[i]['Periodicite']

        if periodicite == 'Annuel':
            differe_j = differe
        elif periodicite == 'Semestriel':
            differe_j = differe / 2
        elif periodicite == 'Trimestriel':
            differe_j = differe / 4
        else:
            differe_j = differe / 12

        date_debut = datetime.strptime(df.iloc[i]['DateJouissance'], '%Y-%m-%d')
        annees_diff = datetime.today().year - (date_debut.year - int(differe_j))
        diff = max(0, differe - annees_diff)
        res.append(diff)
    return res

def taux_actuariel_annuel(df):
    taux_list = []
    nbrediffere = calc_diff(df)

    for i in range(len(df)):
        try:
            nominal = df.iloc[i]['ValeurNominale']
            taux = df.iloc[i]['TauxInteretsNet'] / 100
            periodicite = df.iloc[i]['Periodicite'].lower()
            diff = int(nbrediffere[i])

            if periodicite == 'annuel':
                freq = 1
            elif periodicite == 'semestriel':
                freq = 2
            elif periodicite == 'trimestriel':
                freq = 4
            elif periodicite == 'mensuel':
                freq = 12
            else:
                freq = 1

            if diff >= freq:
                flux = [-nominal] + [0] * freq
            else:
                paiements = freq - diff
                interet = taux * nominal / freq
                flux = [-nominal] + [0]*diff + [interet]*paiements

            tir = npf.irr(flux)
            taux_list.append(tir if not np.isnan(tir) else 0)

        except Exception as e:
            print(f"Erreur ligne {i} : {e}")
            taux_list.append(0)

    return taux_list

def contrainte_somme(x):
    return np.sum(x) - 1

def contrainte_capital(x, df, capital):
    total = np.sum(x * df['ValeurNominale'])
    return capital - total

def objectif_rendement(x, df):
    taux = df['Rendement1an'].values
    return -np.sum(x * taux)

def construire_portefeuille(df, periode_cible=1, nb_obligations=3, capital=1000000):
    # Étape 1 & 2 : filtrage selon la période restante
    obligations_filtrees = df[df.apply(lambda x: perioderest(x) >= periode_cible, axis=1)].copy()
    if obligations_filtrees.empty:
        print("Aucune obligation disponible pour la période cible.")
        return None

    # Étape 3 : calcul du rendement sur 1 an
    obligations_filtrees['Rendement1an'] = taux_actuariel_annuel(obligations_filtrees)

    # Étape 4 : top 10 des meilleures obligations
    top_obligations = obligations_filtrees.sort_values(by='Rendement1an', ascending=False).head(10)

    # Étape 5 : combinaison des meilleures obligations
    meilleures_combinaisons = list(combinations(top_obligations.index, nb_obligations))
    meilleur_score = -np.inf
    meilleur_portefeuille = None
    meilleure_alloc = None

    for combo in meilleures_combinaisons:
        portef = top_obligations.loc[list(combo)].copy()
        x0 = np.random.dirichlet(np.ones(nb_obligations), size=1)[0]

        cons = [
            {'type': 'eq', 'fun': contrainte_somme},
            {'type': 'ineq', 'fun': lambda x: contrainte_capital(x, portef, capital)}
        ]

        res = minimize(objectif_rendement, x0, args=(portef,), constraints=cons, bounds=[(0, 1)] * nb_obligations)

        if res.success and -res.fun > meilleur_score:
            meilleur_score = -res.fun
            meilleur_portefeuille = portef
            meilleure_alloc = res.x

    if meilleur_portefeuille is not None:
        return meilleur_portefeuille.reset_index(drop=True), meilleure_alloc, meilleur_score
    else:
        print("Échec d'optimisation.")
        return None
