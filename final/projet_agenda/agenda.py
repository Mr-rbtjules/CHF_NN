import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import copy

"""
penser a pt etendre le range car sketchy
"""


class Agenda:

    def __init__(
            self, 
            lessons:            list = None,
            origin_agenda_path: str = None,
            remove:             dict=None
    ) ->None:
        

        self.remove = remove
        self.lessons = lessons
        self.origin_agenda_path = origin_agenda_path
        self.limit_date = '01/01/2024'
        self.start_time = '6:00 PM'
        self.end_time = '7:00 PM'
        self.colomns = ['Subject', 'Start Date', 'Start Time', 
                        'End Date', 'End Time', 'All Day Event', 
                        'Description', 'Location', 'Private']
        self.end_path = './test.csv'
        self.space = [0,1,3,7,2*7,4*7, 7*7]
        self.range = [0,0,0,1,2,4, 7]
        self.max_rev = 6
        self.format = '%d/%m/%Y'
        self.count = dict(zip(self.lessons, [0]*len(self.lessons)))
        self.origin_agenda = pd.read_csv(
            origin_agenda_path, 
            skiprows=[0,1,2]
        )
        self.date = ''
        self.my_dict_agenda = {} #dict
        self.my_agenda = pd.DataFrame(columns=self.colomns) #dataframe
        self.sortOrigin() #check si on fait systematiquement

    def sortOrigin(self) -> None:
        self.origin_agenda = self.origin_agenda[
            self.origin_agenda['Cours'].isin(self.lessons)
            ]
        self.origin_agenda = self.origin_agenda.loc[
            (self.origin_agenda['Type de réservation'] == 'Théorie')
        ]
        return None
    
    def createDictAgenda(self) -> None:
        for lesson in range(len(self.origin_agenda)):
            origin_date = self.origin_agenda.iloc[lesson]['Date de début']
            lesson_name = self.origin_agenda.iloc[lesson]['Cours']
            self.count[lesson_name] += 1
            if self.count[lesson_name] not in self.remove[lesson_name]:
                self.spacedRepetition(origin_date, lesson_name)

        return None
    
    def spacedRepetition(self, origin_date, lesson_name):
        self.date = origin_date
        for step in range(len(self.space)):
            self.addRevision(self.date, lesson_name, step)
        return None



    def addRevision(self, origin_date, lesson_name, step) -> None:
        #no event at this date
        new_date = self.addDays(origin_date, self.space[step])
        self.date = new_date
        #on l'inscrit pas si on a dépassé la date limite
        if (datetime.strptime(new_date, self.format)
            >= datetime.strptime(self.limit_date, self.format)
        ):
            return None
        if new_date not in self.my_dict_agenda:
            self.my_dict_agenda[new_date] = [[
                lesson_name,
                self.count[lesson_name], 
                step
            ]]
        #already an event
        else:
            self.my_dict_agenda[new_date].append([
                lesson_name,
                self.count[lesson_name], 
                step
            ])
        return None
    
    def optimiseRev(self):
        optimised = copy.deepcopy(self.my_dict_agenda)

        today = "18/09/2023"
        #on passe sur tt les dates possible
        while today != self.limit_date:
            #celle ou y a un event
            if today in optimised:
                #on reduit la charge de travail la ou y a trop
                while len(optimised[today]) > self.max_rev:
                    #trouve le plus etaper
                    index = self.get_biggest_step(optimised[today])
                    if index == None:
                        raise Exception("tous le mm jour ?")
                    #on le pop
                    to_replace = optimised[today].pop(index)#reduit la taille donc la boucle tourne
                    #function qui va trouver dans le range de jours
                    #la où y a le moins de rev et qui est le + proche
                    #de today et plus loin donc alterne avant apres
                    #en se rapprochant de today
                    
                    #place la date range jour apres today
                    minimum = self.max_rev
                    date_mini = today
                    step = to_replace[2]
                    end = self.addDays(today, self.range[step] +1)
                    start = self.subsDays(today, self.range[step])
                    #tant que  != today , cherche le minimum dans le range
                    while (start != end):           #!!! prendre en compte qu'on a pop à today
                        #si le jour possede un event
                        if start in optimised:
                            rev_nb = len(optimised[start])
                            if rev_nb <= minimum:
                                minimum = rev_nb
                                date_mini = start
                        #si pas d'event place libre donc on place
                        else:
                            minimum = 0
                            date_mini = start

                        #date +1
                        start = self.addDays(start, 1)
                        

                    if date_mini == today:
                        raise Exception("nothing available in the range")
                    else:
                        if date_mini in optimised:
                            optimised[date_mini].append(to_replace)
                        else:
                            optimised[date_mini] = to_replace
            today = self.addDays(today, 1)
        
        self.my_dict_agenda = optimised
        
        
    
    def get_biggest_step(self, lis) -> int:
        """take a list of list and return the list and
        list index with the biggest third element"""
        maxi = 0
        index = None
        for i in range(len(lis)):
            if lis[i][2] > maxi:
                index = i
                maxi = lis[i][2]

        return index

    def createCsvAgenda(self):
        self.createDictAgenda()
        self.optimiseRev()
        #dict -> csv for calendar
        for date in self.my_dict_agenda:
            info = ''
            for event in self.my_dict_agenda[date]:
                info += f'{event[0]} partie {event[1]} étape {event[2]} '
            
            rev_name = 'Rev' + str(len(self.my_dict_agenda[date]))
            val = [rev_name, date, self.start_time, date, self.end_time, False , info, 'bx', False]
            line = pd.DataFrame(dict(zip(self.colomns, val)), index=[0])
            self.my_agenda = pd.concat([self.my_agenda, line])
        print(self.my_agenda)
        self.my_agenda.to_csv(self.end_path, index=False)
        print("saved new csv file in test.csv")
        return None

    
    def addDays(self, origin, days):
        date_format = datetime.strptime(origin, self.format)
        nouvelle_date = date_format + timedelta(days=days)
        nouvelle_date_str = nouvelle_date.strftime(self.format)
        return nouvelle_date_str
    
    def subsDays(self, origin, days):
        date_format = datetime.strptime(origin, self.format)
        nouvelle_date = date_format - timedelta(days=days)
        nouvelle_date_str = nouvelle_date.strftime(self.format)
        return nouvelle_date_str
    

    #cree une liste 
    def removePart(self):
        return None

lessons = ['PHYSH401','PHYSH406'] 
part_to_remove = [[],[1,2]]
remove = dict(zip(lessons, part_to_remove))
my_agenda = Agenda(
    lessons=lessons,
    origin_agenda_path="horaire.csv",
    remove=remove
    )



my_agenda.createCsvAgenda()