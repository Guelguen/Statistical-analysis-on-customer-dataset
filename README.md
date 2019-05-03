# Analysis of customer dataset


3 CSV-Dateien eines Outdoor-Shops wurden zur Analyse genutzt:

- kunden.csv
  - Person_ID: einzigartiger Identifier pro Kunde
  - Geschlecht
  - Geburtsdatum
  - PLZ
  - Stadt
  - Land

- bestellungen.csv
  - Person_ID
  - Bestellung_ID: einzigartiger Identifier pro Bestellung (Bestellung besteht aus mehreren Artikeln)
  - Bestelldatum
  - Bestellsumme: Summe aller Artikel einer Bestellung
  - Bezahlstatus

- bestellpositionen.csv
  - Bestellung_ID
  - Artikel_ID: einzigartiger Identifier pro Artikel
  - Artikeldeteil_ID: einzigartiger Identifier pro Artikel mit einer bestimmten Ausprägung (Farbe, Größe,...)
  - Anzahl
  - Einzelpreis
  - Gesamtpreis: setzt sich aus Anzahl*Einzelpreis zusammen (wenn der Gesamtpreis 0 beträgt, wurde der Artikel storniert)

---

Aufgabe 1:
Finde für jede Person_ID heraus, wie lange es im Durchschnitt dauert, bis ein bestimmter Kunde noch einmal in diesem Shop einkauft (dabei
ist zu beachten, dass keine stornierten Bestellungen bei der Analyse berücksichtigt werden). Außerdem ist herauszufinden, ob Personen,
die ein kurzes Kaufintervall aufweisen, auch im Durchschnitt mehrere oder eine einzige teure Bestellung aufgeben. Besteht
hierbei auch ein Unterschied bei den Geschlechtern?  

Aufgabe 2:
In welchen Postleitzahlen, wird im Durchschnitt am meisten eingekauft? Findet man hierbei drastische saisonale Unterschiede? Beispiel:
in der 3-stelligen PLZ (263XX) wird verhältnismäßig im Sommer weniger eingekauft, als bei PLZ (256XX) im Sommer und Winter. Eigentlich
geht man bei Einkaufszahlen im Jahresverlauf von einer doppelten Saisonalität aus (im Sommer und Winter ein Hoch). Welche 3-stelligen
PLZ fallen in dieses Muster und welche nicht? 

