
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter 
import datetime as dt
from copy import copy
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from pytictoc import TicToc

#==============================================================================

# Funktionen

def to_percent(x,position): 
    '''
    berechnet den prozentualen Anteil einer absoluten Zahl an Käufern x
    in Relation zu allen Multishoppern (8188 Stück)     
    '''    
    return(str(int(x*100/8188)) + '%') 

#------------------------------------------------------------------------------
                                            
def axoptions(axnum,percent=False,
              xlabel=None,ylabel=None,
              xrange=None,yrange=None):
    '''
    Optionen für einen Plot     
    '''
    # y-Achse zu Prozent umwandeln    
    if percent:    
        axnum.yaxis.set_major_locator(plt.MaxNLocator(5))
        formatter = FuncFormatter(to_percent)
        axnum.yaxis.set_major_formatter(formatter)    

    # Achsenbeschriftung setzen
    if xlabel is not None: axnum.set_xlabel(xlabel)
    if ylabel is not None: axnum.set_ylabel(ylabel)
    
    # Frame teilweise entfernen
    axnum.spines['right'].set_visible(False)
    axnum.spines['top'].set_visible(False)
    axnum.tick_params(bottom='off', top='off', right='off', left='off')

    # Window setzen
    if xrange  is not None: axnum.set_xlim([xrange[0],xrange[1]])
    if yrange  is not None: axnum.set_ylim([yrange[0],yrange[1]])

#------------------------------------------------------------------------------          
          
def singleplot(variable,kind,axnum, figsiz=(10.8),
               percent=False,xlabel=None,ylabel=None,xrange=None,yrange=None,
               filename='unbenannt.png'):
    '''
    Vereinfachung der Optionen für einen Plot;
    Image wird größer, axoptions ausgeführt 
    und das figure automatisch gespeichert 
    (funktioniert noch nicht)
    '''
    fig,axnum = plt.subplots(figsize=figsiz)
    variable.plot(kind=kind)
    axoptions(axnum,percent,xlabel,ylabel,xrange,yrange)
    plt.savefig(filename)

#------------------------------------------------------------------------------

def dauer(personID):
    '''
    gibt die durchschnittliche Dauer bis zur nächsten Bestellung für einen
    Kunden mit gültiger Person_ID wieder   
    Input: person_ID wie 10, nicht '10'
    '''
    try:
        time   = allshoppers.Zeitdurchschnitt.loc[personID]
        if time == 'SingleShopper': string = 'Kunde hat nur einmal eingekauft'        
        else: string = 'Kunde mit Person_ID '  + \
                        str(personID) + \
                        ' kauft durchschnittlich nach '+ \
                        str(time) + \
                        ' Tagen wieder ein'
    except (KeyError,UnboundLocalError): 
        print('Kundennummer nicht vorhanden oder falsch eingegeben, korrekt: dauer(10)')
        string = None
    return(string)
    
#==============================================================================

# Dateien einlesen und durchgehende NAN-Zeilen entfernen
df1 = pd.read_csv("kunden.csv",sep=';').dropna(how='all') 
df2 = pd.read_csv("bestellungen.csv",sep=';').dropna(how='all') 
df3 = pd.read_csv("bestellpositionen.csv",sep=';').dropna(how='all') 

# Index ersetzen
df1.set_index("Person_ID", inplace=True)
df2copy = copy(df2.set_index("Person_ID")) # Indizes nicht unique!

#============================================================================== 
'''
Aufgabe 1:
Finde für jede Person_ID heraus, wie lange es im Durchschnitt dauert, 
bis ein bestimmter Kunde noch einmal in diesem Shop einkauft 
(dabei ist zu beachten, dass keine stornierten Bestellungen bei der 
Analyse berücksichtigt werden). 
'''
#==============================================================================

# Variablen

# Entferne Stornos und Rückzahlungen
df2 = df2[(df2.Bezahlstatus != 'storno') & (df2.Bezahlstatus != 'Rückzahlung')]
df2copy = df2copy[(df2copy.Bezahlstatus != 'storno') & \
                  (df2copy.Bezahlstatus != 'Rückzahlung')]                

# transformiere Datum, effizientere Lösung?
t = TicToc()
t.tic()
df2.Bestelldatum = pd.to_datetime(df2.Bestelldatum,dayfirst=True)   
df2copy.Bestelldatum = pd.to_datetime(df2copy.Bestelldatum,dayfirst=True)  

df2 = df2.sort_values(['Person_ID','Bestelldatum'])                 
df2['Zeitabstand'] = df2.groupby('Person_ID')['Bestelldatum'].diff().dt.days 
t.toc
# neues Dataframe mit Zeitmittelwert[days] pro Person_ID
allshoppers = pd.DataFrame(df2.groupby('Person_ID')['Zeitabstand'].mean()) 
allshoppers.columns = ['Zeitdurchschnitt']
# ersetze NaT durch 'Einzelkäufer'
allshoppers[pd.isnull(allshoppers.Zeitdurchschnitt)] = 'SingleShopper'                       

df2copy['Zeitabstand'] = allshoppers.Zeitdurchschnitt
df2 = df2copy
del df2copy

multi = allshoppers[allshoppers.Zeitdurchschnitt != 'SingleShopper']
# prozentualer Anteil der Einzelkäufer
einzel = allshoppers.Zeitdurchschnitt.value_counts\
         ('SingleShopper')['SingleShopper']

meanarray = multi.values            # Umwandeln in Array
meanofmeans = np.mean(meanarray)/30 # Durchschnitt aller Multikäufer in Monaten

#Antwort durch allgemeine Funktion:
###############################################################################
dauer('010')
dauer(89)
dauer(85)
dauer(10)
dauer(817)
###############################################################################

#------------------------------------------------------------------------------

# Plots

# Tortendiagramm
plt.figure(figsize=(10,8))
values = [einzel,1-einzel] 
labels = ['SingleShopper', 'MultiShopper']
pie = plt.pie(values, explode=(0.2, 0), labels=labels, autopct='%1.1f%%',
        startangle=-50,colors=['lightpink','c'],labeldistance=0.75)
for pie_wedge in pie[0]:
    pie_wedge.set_edgecolor('white')
plt.title('prozentualer Anteil aller Kunden nach Anzahl der Bestellungen k \
          \nSingle: $k=1$, Multi: $k>1$',fontsize=16)
plt.savefig('/home/guelguen/Desktop/pie.png')

# Multi-Histogram Überblick
fig1  = plt.figure(figsize=(10,12))
fig1.subplots_adjust(hspace=0.3) 
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212) 
ax1.hist(meanarray,bins=40)
axoptions(ax1,True,'Dauer bis zur nächsten Bestellung [Tage]','Anteil [%]')
ax2.hist(meanarray,bins=90,range=(0,90))
axoptions(ax2,True,'Dauer bis zur nächsten Bestellung [Tage]','Anteil [%]')
plt.suptitle('Überblick über die Anteile der Multi-Shopper in %',fontsize=16)
plt.savefig('/home/guelguen/Desktop/hist_überblick.png')

# Multi-Histogram kumuliert
fig = plt.figure()
fig  = plt.figure(figsize=(10,12))
fig.subplots_adjust(hspace=0.3)       
plt.suptitle('Kumulierter prozentualer Anteil aller Kunden in Abhängigkeit \
             \nder Dauer bis zur nächsten Bestellung',fontsize=18)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312) 
ax3 = fig.add_subplot(313) 
ylab = 'kumulierter Anteil [%]'
ax1.hist(meanarray,bins=np.arange(31)-0.5,edgecolor="w",color='c',
         cumulative=True)
axoptions(ax1,True,'Dauer bis zur nächsten Bestellung [Tage]',ylab)
ax2.hist(meanarray/30,bins=np.arange(7)-0.5,edgecolor="w",color='c',
         cumulative=True)
axoptions(ax2,True,'Dauer bis zur nächsten Bestellung [Monate]',ylab)
ax3.hist(meanarray/365,bins=np.arange(0,5,0.5)-0.25,edgecolor="w",color='c',
         cumulative=True)
axoptions(ax3,True,'Dauer bis zur nächsten Bestellung [Jahre]',ylab)
plt.savefig('/home/guelguen/Desktop/hist_kumuliert.png')


#==============================================================================
'''
Außerdem ist herauszufinden, ob Personen, die ein kurzes Kaufintervall 
aufweisen, auch im Durchschnitt günstigere Artikel einkaufen oder 
teurere einzelne Artikel einkaufen. Besteht hierbei auch ein Unterschied 
bei den Geschlechtern? 
'''
#==============================================================================

# SingleShopper-Analyse

single = allshoppers[allshoppers.Zeitdurchschnitt == 'SingleShopper']
single.insert(1, 'Geschlecht', df1.loc[single.index,'Geschlecht'])  
single.insert(2,'Bestellsumme',df2.loc[single.index,'Bestellsumme'])
single.drop('Zeitdurchschnitt', axis=1, inplace=True)   # lösche Spalte

# Tortendiagramm zu den Geschlechtern
plt.figure(figsize=(10,8))
values = single.Geschlecht.value_counts('Herr') 
labels = ['weiblich', 'männlich','keine Angabe']
pie = plt.pie(values, labels=labels, autopct='%1.1f%%',
              startangle=-50,colors=['hotpink','blue','k'],
              textprops={'fontsize': 16})
for wedge, autotext in zip(pie[0],pie[2]):
    wedge.set_edgecolor('white')
    autotext.set_color('white')
plt.title('Anteil der Geschlechter aller SingleShopper',fontsize=14)
plt.savefig('/home/guelguen/Desktop/pie_sex.png')

# Histogramm zu den Geschlechtern

sex = pd.DataFrame({'male': single[single.Geschlecht == 'Herr'].Bestellsumme,
                    'female': single[single.Geschlecht == 'Frau'].Bestellsumme})
fig,ax = plt.subplots(figsize=(10,8))
sex.female.hist(color='hotpink',bins=50,range=(0,1200),alpha=0.4,edgecolor='w')
sex.male.hist(color='c',grid=False,bins=50,range=(0,1200),edgecolor='w')
axoptions(ax,True,'Bestellbetrag [€]','Anteil [%]')
ax.legend(('weiblich','männlich'),fontsize=10)
plt.title('Geschlechteranteil der Singleshopper \nin Abhängigkeit des Kaufbetrages',
          fontsize=16)
plt.savefig('/home/guelguen/Desktop/hist_sex.png')

#------------------------------------------------------------------------------

# MultiShopper-Analyse
 
multi.insert(0,'Geschlecht',df1.loc[multi.index,'Geschlecht'])
multi.insert(2,'Gesamtsumme',pd.DataFrame(df2.groupby(df2.index)\
                             ['Bestellsumme'].sum()).loc[multi.index])

# Scatterplots (einzeln, um sie nachher flexibel anzuordnen)
color = copy(multi.Geschlecht.replace(['Herr','Frau','unbekannt'], 
                                      ['blue','hotpink','w'])) 
 # Punktgröße nach Häufigkeit der Bestellung
size  = pd.DataFrame({'Häufigkeit':df2.\
                                   groupby(df2.index)['Bestellsumme']\
                                   .count()})   
size  = size[size.Häufigkeit != 1]

fig,ax = plt.subplots(figsize=(10,8))
ax.scatter(multi.Zeitdurchschnitt,multi.Gesamtsumme,s=size*10,color=color)
axoptions(ax,False,'Dauer bis zur nächsten Bestellung [Tage]',
          'Gesamtbestellwert pro Kunde [€]')
plt.suptitle('Gesamtbestellwert pro Kunde in Abhängigkeit der Zeit bis \
             \nzur nächsten Bestellung',fontsize=12)
plt.legend([Circle((0, 0), fc="hotpink"),Circle((0, 0), fc="blue")],
            ['weiblich','männlich'],fontsize=10)        # Legende mit Kreisen?
plt.savefig('/home/guelguen/Desktop/scatter_multi.png')

# Betrachtung in den ersten drei Monaten
fig,ax = plt.subplots(figsize=(10,8))
ax.scatter(multi.Zeitdurchschnitt,multi.Gesamtsumme,s=size*10,color=color)
axoptions(ax,False,'Dauer bis zur nächsten Bestellung [Tage]',
          'Gesamtbestellwert pro Kunde [€]',[0,90])
plt.savefig('/home/guelguen/Desktop/scatter_multi_90.png')

fig,ax = plt.subplots(figsize=(10,8))
ax.scatter(multi.Zeitdurchschnitt,multi.Gesamtsumme,s=size*10,color=color)
axoptions(ax,False,'Dauer bis zur nächsten Bestellung [Tage]',
          'Gesamtbestellwert pro Kunde [€]',[0,90],[0,5000])
plt.savefig('/home/guelguen/Desktop/scatter_multi_90_5000.png')

# innerhalb eines Jahres Höchstwerte betrachten
fig,ax = plt.subplots(figsize=(10,8))
ax.scatter(multi.Zeitdurchschnitt,multi.Gesamtsumme,s=size*10,color=color)
axoptions(ax,False,'Dauer bis zur nächsten Bestellung [Tage]',
          'Gesamtbestellwert pro Kunde [€]',[0,400],[5000,40000])
plt.savefig('/home/guelguen/Desktop/scatter_multi_400.png')

#==============================================================================
'''
Aufgabe 2:
In welchen Postleitzahlen wird im Durchschnitt am meisten eingekauft? 
Findet man hierbei drastische saisonale Unterschiede? Beispiel:
In der 3-stelligen PLZ (263XX) wird verhältnismäßig im Sommer weniger 
eingekauft, als bei PLZ (256XX) im Sommer und Winter. 
Eigentlich geht man bei Einkaufszahlen im Jahresverlauf von einer doppelten 
Saisonalität aus (im Sommer und Winter ein Hoch). Welche 3-stelligen
PLZ fallen in dieses Muster und welche nicht? 
'''
#==============================================================================


# Alle folgenden Sortierungen der PLZ beziehen sich nur auf DE und sind 3-stellig 
# Natürlich könnte man noch eine flexible Funktion schreiben, 
# sodass die Analysen präziser werden

# speziell Reinigen der PLZ
df1clean    = df1.dropna()                                             
df1plz      = df1clean[df1clean.PLZ.str.isdigit() & (df1clean.PLZ.str.len()==5)]
df1plz.PLZ  = df1plz.PLZ.str.slice(stop=3)       # PLZ 3stellig nicht unique
P           = np.sort(df1plz.PLZ.unique())[10:]  # PLZ-Labels

df2plz              = copy(df2)
df2plz.Bestelldatum = df2.Bestelldatum.dt.month  # Monat herausfiltern

# Dictionary über alle PLZs abcxx (Monate ohne Einkauf fehlen)
# '010': { Monat_1 : Durchschnitt aller Einkäufe in diesem Monat über alle Jahre
#          Monat_2 : ...
#          ...
#          Monat_i : ...
#          ...
#          Monat_12: ...
#         }
# '011': ....
#  ....
# '999': ....

plzdict = {}
plzmean = pd.Series()
t.tic()
for p in P:
     try:
         data = df2.loc[df1plz[(df1plz.PLZ == p) & \
         (df1plz.Land == 'Deutschland')].index].\
         groupby('Bestelldatum')['Bestellsumme'].mean()     
         if len(data) == 0: plzdict[p] = 'keine deutsche PLZ'     
         else: 
             plzmean[p] = data.mean()
             plzdict[p] = dict(zip(data.index,data))  
     except KeyError: plzdict[p] = 'keine Bestellungen in dieser PLZ'
t.toc()  
# 38 Sekunden zur Berechnung des Dicts. Effizienterer Weg?
plzmean = plzmean.sort_values(ascending=False)     



plzseason = pd.DataFrame(columns=['Frühling','Sommer','Herbst','Winter'])
for p in P:
    try:
        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 0
        for k in plzdict[p].keys():        
            if  k in [2.0, 3.0, 4.0]:   s1 += plzdict[p][k]        
            if  k in [5.0, 6.0, 7.0]:   s2 += plzdict[p][k]        
            if  k in [8.0, 9.0, 10.0]:  s3 += plzdict[p][k]        
            if  k in [11.0, 12.0, 1.0]: s4 += plzdict[p][k]        
        plzseason.loc[p] = [s1,s2,s3,s4]
    except (TypeError,AttributeError): continue 
del s1,s2,s3,s4,k,p

#------------------------------------------------------------------------------
'''In welchen Postleitzahlen wird im Durchschnitt am meisten eingekauft? '''

print('In diesen PLZ wurde durchschnittlich am meisten gekauft:')
pd.DataFrame(plzmean[:5],columns=['Bestelldurchschnitt']).style

#------------------------------------------------------------------------------
'''Findet man hierbei drastische saisonale Unterschiede?'''

fig,ax = plt.subplots(figsize=(10,8))
plzseason.boxplot(grid=False,widths=0.8)
axoptions(ax,False,'Season',
          'durchschnittliche Bestellsumme über die Saison (3 Monate)[€]')
plt.savefig('/home/guelguen/Desktop/boxplot_gesamt.png')

fig,ax = plt.subplots(figsize=(10,8))
plzseason.boxplot(grid=False,widths=0.8)
axoptions(ax,False,'Season',
          'durchschnittliche Bestellsumme über die Saison (3 Monate)[€]',
          yrange=[0,2000])
plt.savefig('/home/guelguen/Desktop/boxplot_zoomin.png')

#------------------------------------------------------------------------------
'''
Eigentlich geht man bei Einkaufszahlen im Jahresverlauf von 
einer doppelten Saisonalität aus (im Sommer und Winter ein Hoch). 
Welche 3-stelligen PLZ fallen in dieses Muster und welche nicht? 
'''

plzmultisaison = pd.DataFrame(columns=['Sommer','Winter', 'Sommer+Winter'])
plzmultisaison.Sommer = (plzseason.Frühling < plzseason.Sommer) & \
                        (plzseason.Herbst < plzseason.Sommer) 
plzmultisaison.Winter = (plzseason.Frühling < plzseason.Winter) & \
                        (plzseason.Herbst < plzseason.Winter) 
plzmultisaison['Sommer+Winter'] = plzmultisaison.Sommer & plzmultisaison.Winter

result=plzmultisaison['Sommer+Winter']\
      [plzmultisaison['Sommer+Winter'] ==True].keys()

print('Diese PLzs fallen in das Muster der Doppelsaisonalität:\n' + str(result))
pd.DataFrame(result).style

