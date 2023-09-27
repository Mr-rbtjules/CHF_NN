import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import copy


"""
on recupere le ical on le rentre dans agenda, on recupere le csv pour en faire un csv
mettre une date limite

peut gerer l'ensemble de notre calendrier avec des fonctions
genre un truc qui retire spécifiquement une partie de cours et tt les rev qui s'en suivent

et on a juste a enlever remettre le csv dans revisions
faire son syllametre !!!
"""

def get_new_date(origin, days):
    format = '%d/%m/%Y'
    date_format = datetime.strptime(origin, format)
    nouvelle_date = date_format + timedelta(days=days)
    nouvelle_date_str = nouvelle_date.strftime(format)
    return nouvelle_date_str

def addRev(agenda, new_date, days_range, info, line, repetition_range) -> pd.DataFrame:
    new_line = copy.deepcopy(line)
    if agenda.loc[agenda['Start Date'] == new_date].empty:
        df = pd.DataFrame(new_line, index=[0])
        agenda = pd.concat([agenda, df], ignore_index=True)
    else:
        row = agenda.loc[agenda['Start Date'] == new_date].index[0]
        nb = int(agenda.at[row, 'Subject'][-1])
        if nb <= 6:
            agenda.at[row, 'Description'] = agenda.at[row, 'Description'] + info            
            agenda.at[row, 'Subject'] = 'Rev ' + str(nb+1)
        elif repetition_range == 0:
            Exception("conflict essential rev but too many rev") #lister les cours regarderleur etape et si plus grand que 2 bouger a la place de celui ci
        else:
            return False
            #aller de + range à - range et check a chaque fois si y a la place et puis deposer => en faire une fonction
            pass  

    return agenda


horaire = pd.read_csv("horaire.csv", skiprows=[0,1,2])

#print(sort['Cours'])
horaire = horaire.loc[(horaire['Cours'] == 'PHYSH401') | (horaire['Cours'] == 'PHYSH406')]
horaire = horaire.loc[(horaire['Type de réservation'] == 'Théorie')]

space_repetition = [0,1,3,7,2*7,4*7, 7*7]
max = 6
repetition_range = [0,0,0,1,2,4, 7]
hdeb = '6:00 PM'
hfin = '7:00 PM'
colonnes = ['Subject', 'Start Date', 'Start Time', 'End Date', 'End Time', 'All Day Event', 'Description', 'Location', 'Private']
val = ['Rev 1', '', hdeb, '', hfin, False , '', 'bx', False]
line = dict(zip(colonnes, val))

agenda = pd.DataFrame(columns=colonnes)
count = {'PHYSH401': 0, 'PHYSH406': 0}
for i in range(len(horaire)):
    #pour chaque cours on etabli les repetitions espacées
    origin = horaire.iloc[i]['Date de début']
    cours = horaire.iloc[i]['Cours']
    count[cours] += 1
    for j in range(len(space_repetition)):
        #ajoute une revision
        new_date = get_new_date(origin, space_repetition[j])
        origin = new_date
        info = f'{cours} partie {count[cours]} étape {j} '

        while not addRev(agenda, new_date, repetition_range[j]):
            #if passed origin - range => error
            pass
        #arg: agenda, new date fct recusrive qui mette en place l'event retourne false si jour pas bon

        if agenda.loc[agenda['Start Date'] == new_date].empty:
            val = ['Rev 1', new_date, hdeb, new_date, hfin, False , info, 'bx', False]
            nouvelle_ligne = dict(zip(colonnes, val))
            df = pd.DataFrame(nouvelle_ligne, index=[0])
            agenda = pd.concat([agenda, df], ignore_index=True)
        else:
            row = agenda.loc[agenda['Start Date'] == '30/09/2023'].index[0]
            nb = int(agenda.at[row, 'Subject'][-1])
            if nb <= 6:
                agenda.at[row, 'Description'] = agenda.at[row, 'Description'] + info            
                agenda.at[row, 'Subject'] = 'Rev ' + str(nb+1)
            elif repetition_range[j] == 0:
                Exception("conflict essential rev but too many rev") #lister les cours regarderleur etape et si plus grand que 2 bouger a la place de celui ci
            else:

                #aller de + range à - range et check a chaque fois si y a la place et puis deposer => en faire une fonction
                pass            

    


    row = agenda.loc[agenda['Start Date'] == '30/09/2023'].index[0]
    agenda.at[row, 'Description'] = agenda.at[row, 'Description'] + info
    agenda.to_csv('./test.csv', index=False)
    print(agenda)
    exit()
    pass





print(agenda)
date = ''
if agenda.loc[agenda['Date de début'] == date].empty:
    nouvelle_ligne = {'Colonne1': 4, 'Colonne2': 'D'}

    agenda = agenda.append(nouvelle_ligne, ignore_index=True)
#puis ajoute à info
