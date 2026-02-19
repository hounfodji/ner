# Dataset NER Transport au Bénin : référentiel géographique complet

**Ce référentiel compile plus de 700 entités géographiques et lexicales** essentielles pour la reconnaissance d'entités nommées dans le domaine du transport béninois. Les données couvrent l'intégralité des **77 communes**, les quartiers des trois principales métropoles, les points de repère de transport, et la terminologie locale spécifique.

---

## Structure administrative du Bénin

Le Bénin compte **12 départements** subdivisés en **77 communes** et plus de **545 arrondissements**. Cette organisation territoriale, issue de la réforme de 1999 et finalisée en 2016, constitue le socle de toute annotation géographique.

### Les 12 départements et leurs chefs-lieux

| Département | Chef-lieu | Principales villes |
|-------------|-----------|-------------------|
| **Alibori** | Kandi | Malanville, Banikoara, Ségbana, Gogounou, Karimama |
| **Atacora** | Natitingou | Tanguiéta, Kouandé, Boukoumbé, Kérou, Matéri |
| **Atlantique** | Allada | Abomey-Calavi, Ouidah, Sô-Ava, Toffo, Zè |
| **Borgou** | Parakou | Nikki, Tchaourou, Kalalé, N'Dali, Bembèrèkè |
| **Collines** | Dassa-Zoumè | Savalou, Savè, Glazoué, Bantè, Ouèssè |
| **Couffo** | Aplahoué | Dogbo, Klouékanmè, Lalo, Djakotomey, Toviklin |
| **Donga** | Djougou | Bassila, Copargo, Ouaké |
| **Littoral** | Cotonou | (commune unique) |
| **Mono** | Lokossa | Grand-Popo, Comè, Bopa, Athiémé, Houéyogbé |
| **Ouémé** | Porto-Novo | Sèmè-Kpodji, Adjarra, Avrankou, Dangbo |
| **Plateau** | Pobè | Kétou, Sakété, Ifangni, Adja-Ouèrè |
| **Zou** | Abomey | Bohicon, Covè, Djidja, Zagnanado, Za-Kpota |

### Liste complète des 77 communes (pour annotation NER)

```
Abomey, Abomey-Calavi, Adja-Ouèrè, Adjarra, Adjohoun, Agbangnizoun, Aguégués, 
Akpro-Missérété, Allada, Aplahoué, Athiémé, Avrankou, Banikoara, Bantè, Bassila, 
Bembèrèkè, Bohicon, Bonou, Bopa, Boukoumbé, Cobly, Comè, Copargo, Cotonou, Covè, 
Dangbo, Dassa-Zoumè, Djakotomey, Djidja, Djougou, Dogbo, Glazoué, Gogounou, 
Grand-Popo, Houéyogbé, Ifangni, Kalalé, Kandi, Karimama, Kérou, Kétou, Klouékanmè, 
Kouandé, Kpomassè, Lalo, Lokossa, Malanville, Matéri, N'Dali, Natitingou, Nikki, 
Ouaké, Ouidah, Ouinhi, Ouèssè, Parakou, Pèrèrè, Pobè, Porto-Novo, Sakété, Savalou, 
Savè, Ségbana, Sèmè-Kpodji, Sinendé, Sô-Ava, Tanguiéta, Tchaourou, Toffo, 
Tori-Bossito, Toucountouna, Toviklin, Za-Kpota, Zagnanado, Zè, Zogbodomey
```

### Variantes orthographiques importantes

| Forme officielle | Variantes courantes |
|-----------------|---------------------|
| Dassa-Zoumè | Dassa-Zoumé, Dassa |
| Sèmè-Kpodji | Sèmè-Podji |
| Bembèrèkè | Bembéréké |
| Savè | Savé |
| Pobè | Pobé |
| Comè | Comé |
| Porto-Novo | Hogbonou (fon), Adjatchè (yoruba) |
| Ouidah | Gléhué (ancien nom) |

---

## Quartiers de Cotonou : 143 quartiers en 13 arrondissements

La capitale économique possède la nomenclature la plus dense, cruciale pour les annotations de transport urbain.

### 1er Arrondissement (Akpakpa Nord)
```
Dandji, Donaten, Finagnon, Tchanhounkpamè, Tokplegbe, Avotrou, N'vènamèdé, 
Suru-Léré, Tanto, Yagbé
```

### 2e Arrondissement (Akpakpa)
```
Irédé, Kpondéhou I, Kpondéhou II, Lom-Nava, Sènadé I, Sènadé II, Ahouassa, 
Gankpodo, Djèdjèlayé, Minontchou, Yénawa, Kowégbo
```

### 3e Arrondissement
```
Adjégounlè, Adogléta, Gbénonkpo, Hlakomey, Kpankpan, Midombo, Sègbèya Nord, 
Sègbèya Sud, Agbato, Agbodjedo, Ayélawadjè I, Ayélawadjè II, Fifatin
```

### 4e Arrondissement
```
Enagnon, Fifadji Houto, Sodjatinmè Centre, Sodjatinmè Est, Sodjatinmè Ouest, 
Abokicodji Centre, Abokicodji Lagune, Dédokpo, Gbèdjèwin, Missessin, Ohé
```

### 5e Arrondissement (Zone commerciale)
```
Guinkomè, Tokpa Hoho, Xwlacodji Kpodji, Xwlacodji Plage, Dota, Gbéto, 
Mifongu, Zongo Ehuzu, Zongo Nima, Jonquet, Bocossi Tokpa, Gbédokpo, 
Missèbo, Missité, Nouveau Pont
```

### 6e Arrondissement (Dantokpa)
```
Aïdjèdo I-IV, Ahouansori Agata, Ahouansori Towéta I-II, Gbèdjromèdé, 
Ladji, Dantokpa, Hindé I-II, Jéricho, Ahouansori Agué, Vossa, Djidjè I-II
```

### 7e Arrondissement
```
Gbèdomidji (Maro Militaire), Gbènan, Gbéwa (Batito), Sèdami, Saint-Michel, 
Todoté, Yévèdo, Dagbédji-Sikê, Fignon-Sikê, Sèhogan-Sikê
```

### 8e Arrondissement (Sainte-Rita)
```
Agbodjèdo, Agontinkon, Gbèdagba, Houéhoun, Houénoussou, Mèdédjro, 
Tonato, Minonkpo, Wologuèdè
```

### 9e Arrondissement
```
Fifadji, Vossa-Kpodji, Zogbo, Zogbohouè, Kindonou, Mènontin
```

### 10e Arrondissement
```
Gbénonkpo, Kouhounou, Midédji, Missèkplé, Missogbé, Vèdoko, Yénawa, 
Sètovi, Gounvocodji
```

### 11e Arrondissement (Gbégamey/Vodjè)
```
Gbèdiga I-II, Gbégamey I-IV, Saint-Jean, Allobatin, Ayidoté, Finagnon, 
Houéyiho I-II, Vodjè Centre
```

### 12e Arrondissement (Zone résidentielle/institutionnelle)
```
Aïbatin I, Cadjèhoun I-V, Cocotier, Fidjrossè Centre, Fidjrossè Kpota, 
Fiyégnon I-II, Ahouanlèko, Haie-Vive, Gbodjètin, Akogbato, Yémicodji
```

### 13e Arrondissement (Agla)
```
Agla (subdivisions : Agongbomey, Akplomey, Figaro, Finafa, Les Pylônes, 
Petit Château, Agla-Sud), Ahogbohouè, Aïbatin II, Gbèdégbé, Houénoussou, Missité
```

### Noms courants à l'oral (essentiels pour NER)
```
Akpakpa, Ganhi, Dantokpa, Tokpa, Missèbo, Zongo, Haie Vive, Fidjrossè, 
Cadjèhoun, Gbégamey, Saint-Michel, Sainte-Rita, Placodji, Étoile Rouge, 
PK3, PK6, Jonquet, Mènontin, Vodjè, Vèdoko
```

---

## Quartiers de Porto-Novo : 100 quartiers en 5 arrondissements

### 1er Arrondissement (29 quartiers)
```
Accron-Gogankomey, Adjègounlè, Adomey, Ahouantikomey, Akpassa, Avassa, 
Ayétoro, Ayimlonfidé, Déguèkomè, Dota-Attingbansa, Ganto, Gbassou-Itabodo, 
Gbêcon, Guévié-Zinkomey, Hondji-Honnou, Houègbo, Houéyogbé-Gbèdji, 
Houèzounmey, Idi-Araba, Iléfiè, Kpota, Lokossa, Oganla-Gare, Sadognon, 
Sagbo, Sokomey, Togoh-Adankomey, Vêkpa
```

### 2e Arrondissement (16 quartiers)
```
Agbokou (Aga, Bassodji, Centre social, Odo), Attakè (Olory-Togbé, Yidi), 
Djègan Daho, Donoukin, Gbèzounkpa, Guévié Djèganto, Hinkoudé, 
Kandévié Radio, Koutongbé, Sèdjèko, Tchinvié, Zounkpa Houèto
```

### 3e Arrondissement (22 quartiers)
```
Adjina Nord, Adjina Sud, Avakpa-Kpodji, Avakpa-Tokpa, Djassin Daho, 
Djassin Zounmè, Foun-Foun (Djaguidi, Gbègo, Sodji, Tokpa), Hassou Agué, 
Oganla (Atakpamè, Nord, Poste, Sokè, Sud), Ouinlinda, Zèbou (Aga, 
Ahouangbo, Itatigri, Massè)
```

### 4e Arrondissement (18 quartiers)
```
Anavié, Djègan Kpèvi, Dodji, Gbèdjromèdé, Gbodjè, Guévié, Hlogou, 
Houinmè (Château d'eau, Djaguidi, Ganto, Gbèdjromèdé), Hounsa, Hounsouko, 
Kandévié (Missogbé, Owodé), Kpogbonmè, Sèto-Gbodjè
```

### 5e Arrondissement (15 quartiers)
```
Akonaboè, Djlado, Dowa (Centre, Aliogbogo, Dédomè), Houinvié, Louho, 
Ouando (Centre, Clékanmè, Kotin), Tokpota (Dadjrougbé, Davo, Vèdo, Zèbè, Zinlivali)
```

---

## Quartiers d'Abomey-Calavi : 149 localités en 9 arrondissements

### Arrondissement Abomey-Calavi Centre
```
Agori, Kansoukpa, Agamadin, Gbodjo, Sèmè, Tokpa-Zoungo
```

### Arrondissement Godomey (49 subdivisions - zone la plus dense)
```
Cococodji, Cocotomey, Dèkoungbé, Godomey-Gare, Godomey-N'gbého, Houalaco, 
Salamey, Togbin, Godomey Togoudo, Ylomahouto, Tankpè, Atrokpo-Codji, 
Fignonhou, Maria-Gléta, Alègléta
```

### Arrondissement Akassato
```
Adjagbo, Agassa Godomey, Agonsoudja, Akassato Centre, Gbètagbo, Glotokpa, 
Houèkègbo, Houèkè Honou, Misséssinto, Kpodji-Les-Monts
```

### Arrondissement Togba
```
Ahossou Gbéta, Drabo, Houéto, Houéga-Agué, Houéga-Tokpa, Somè, Tokan, Zogbadjè
```

### Arrondissement Zinvié
```
Adjogansa, Dangbodji, Dokomey, Gbodjé, Gbodjoko, Kpotomey, Sokan, 
Wawata, Yévié, Zinvié Centre, Zinvié Zoumè
```

### Arrondissement Glo-Djigbé
```
Agongbé, Djissoukpa, Doméagbo, Glo-Djigbé, Glo Fanto, Lohoussa, Yékon (Aga, Do), Glo Missebo
```

### Arrondissement Ouèdo
```
Adjagbo, Ahouato, Allansankomè, Dassèkomè, Kpossidja, Ouèdo Centre
```

### Arrondissement Hêvié
```
Adovié, Akossavié, Dossounon, Houinmè, Zoungo
```

### Arrondissement Kpanroun
```
Anagbo, Avagbé, Wagnizoun, Djigbo, Avogniko, Kpanroun Centre, Kpaviedja, Zoungbo
```

### Autres localités notables
```
Womey, Houndodji, Lobozounkpa, Sèdégbé, Séminaire, Zèkanmey, Zoca, Zopah
```

---

## Points de repère pour le transport

### Gares routières principales

| Gare | Ville | Type |
|------|-------|------|
| **Gare de Jonquet** | Cotonou | Principale, taxis et bus |
| **Gare de Dantokpa** | Cotonou | Taxis-brousse |
| **Gare de l'Étoile Rouge** | Cotonou | Hub central |
| **Gare de Saint Michel** | Cotonou | Minibus Tokpa-tokpa |
| **Gare de Godomey** | Abomey-Calavi | Interurbain |
| **Gare de Ouando** | Porto-Novo | Principale |
| **Gare de Saint Pierre-Paul** | Porto-Novo | Transport régional |
| **Gare routière de Parakou** | Parakou | Nord du pays |
| **Gare de Bohicon** | Bohicon | Centre du pays |

### Marchés majeurs (repères de navigation)

| Marché | Ville | Importance |
|--------|-------|------------|
| **Marché Dantokpa** | Cotonou | Plus grand d'Afrique de l'Ouest, 20 ha |
| **Marché Missèbo** | Cotonou | 2e marché, friperies |
| **Marché Ganhi** | Cotonou | Zone commerciale |
| **Marché Saint Michel** | Cotonou | Quartier central |
| **Marché Arzèkè** | Parakou | 2e du Bénin, 10 530 m² |
| **Marché Dépôt** | Parakou | Près de la gare |
| **Marché Ouando** | Porto-Novo | Principal de la capitale |
| **Marché de Bohicon** | Bohicon | Pôle économique régional |
| **Marché de Malanville** | Malanville | Frontière Niger |

### Carrefours connus (repères essentiels)

```
Carrefour de l'Étoile Rouge, Carrefour Vèdoko, Carrefour Cica Toyota, 
Carrefour des Trois Banques, Carrefour Steinmetz, Carrefour Dantokpa, 
Carrefour Abomey Gare, Carrefour Le Bélier, Carrefour PK3, 
Carrefour Saint Michel, Carrefour SOBEBRA, Carrefour Notre-Dame, 
Carrefour Air Afrique, Carrefour Toyota, Carrefour Houéyiho, 
Carrefour Cadjèhoun, Carrefour Agla/Pylônes, Carrefour FUNAI, 
Carrefour La Vie, Carrefour Tankpè, Carrefour Bidossessi, 
Carrefour Sainte Rita, Carrefour Aigle, Échangeur de Godomey, 
Échangeur de Vèdoko, Place de l'Étoile Rouge, Place du Souvenir
```

### Autres repères urbains importants
```
Aéroport Cardinal Bernardin Gantin (Cadjèhoun), Port Autonome de Cotonou, 
Stade de l'Amitié (Stade GMK), CNHU-HKM, CHU-MEL (HOMEL), 
Université d'Abomey-Calavi (UAC), Camp Guézo, Ancien Pont, 
Troisième Pont, Passage supérieur Steinmetz, Passage supérieur Houéyiho
```

---

## Axes routiers principaux

### Routes Nationales Inter-États (RNIE)

| Route | Tracé | Distance |
|-------|-------|----------|
| **RNIE 1** | Frontière Togo → Cotonou → Porto-Novo → Frontière Nigeria | 177 km |
| **RNIE 2** | Cotonou → Bohicon → Parakou → Malanville → Frontière Niger | 785 km |
| **RNIE 3** | Dassa-Zoumè → Savalou → Djougou → Frontière Burkina | 456 km |
| **RNIE 7** | Frontière Burkina → Banikoara → Kandi → Ségbana → Frontière Nigeria | 222 km |

### Axes urbains de Cotonou
```
Boulevard de la Marina, Boulevard Saint Michel, Boulevard du Canada, 
Avenue Jean-Paul II, Avenue Steinmetz, Avenue Delorme, 
Avenue de la Francophonie, Avenue Clozel, Rue des Cheminots, 
Route des Pêches (Cotonou-Ouidah, 40 km), Corniche Est
```

### Routes spéciales
```
Route de l'Esclave (Ouidah), Corridor Abidjan-Lagos, 
Corridor Cotonou-Niamey, Voie de contournement de Parakou, 
Rocade de Porto-Novo
```

---

## Terminologie locale du transport

### Véhicules de transport

| Terme | Variantes | Définition |
|-------|-----------|------------|
| **Zémidjan** | Zem, Zemijan | Moto-taxi (du Fon « Zé mi djan » = emmène-moi vite) |
| **Kêkênon** | Kékéno | Conducteur de moto-taxi |
| **Kloboto** | Cloboto | Tricycle motorisé pour passagers |
| **Kèkèssi** | - | Tricycle (variante régionale) |
| **Tokpa-tokpa** | Tôkpa-tôkpa | Minibus 18 places vers marché Dantokpa |
| **Taxi-brousse** | - | Véhicule 5-9 places inter-villes |
| **Taxi-ville** | Taxi collectif | Taxis jaune-vert partagés |

### Codes couleur uniformes (Zémidjan)
- **Cotonou** : chemise jaune
- **Porto-Novo** : chemise bleue

### Expressions pour demander un transport

| Expression | Langue | Usage |
|------------|--------|-------|
| **« Kêkênon! »** | Fon/Français | Héler un moto-taxi |
| **« C'est combien? »** | Français béninois | Demander le prix |
| **« Nin-bi wè? »** | Fon | « Combien ça coûte? » |
| **« Zé min-yi... »** | Fon | « Emmène-moi à... » |
| **« Je descends à [lieu] »** | Français | Indiquer la destination |
| **« Je descends ici »** | Français | Demander l'arrêt |

### Tarification indicative (FCFA)

| Type | Tarif |
|------|-------|
| Zémidjan court trajet | 100-300 FCFA |
| Zémidjan trajet moyen | 300-500 FCFA |
| Zémidjan long/nuit | 500-1000 FCFA |
| Tokpa-tokpa urbain | 200-500 FCFA |
| Cotonou-Parakou (bus) | 6 500-12 000 FCFA |

### Compagnies de transport inter-villes
```
ATT (Ayina Transport et Tourisme), Baobab Express, La Poste Voyages, 
Confort Lines, BenAfrique, Pax Express, Nonvi Voyage, 
Intercity STC, Ghana Express
```

### Applications VTC actives
```
Gozem, Yango, Bénin Taxi
```

### Vocabulaire connexe
| Terme | Définition |
|-------|------------|
| **Kpayo** | Essence frelatée de contrebande nigériane |
| **Work and pay** | Location-achat de moto (paiement échelonné) |
| **FCFA** | Franc de la Communauté Financière Africaine |
| **Mobile Money** | Paiement mobile (MTN, Moov) |

---

## Récapitulatif des entités pour annotation NER

### Types d'entités recommandés

| Type NER | Exemples | Quantité approximative |
|----------|----------|----------------------|
| **LOC_VILLE** | Cotonou, Parakou, Porto-Novo | ~80 |
| **LOC_QUARTIER** | Akpakpa, Ganhi, Missèbo | ~400 |
| **LOC_COMMUNE** | Les 77 communes | 77 |
| **LOC_DEPT** | Alibori, Borgou, Zou... | 12 |
| **TRANSPORT_GARE** | Gare de Jonquet, Gare de Dantokpa | ~15 |
| **TRANSPORT_MARCHE** | Marché Dantokpa, Marché Arzèkè | ~20 |
| **TRANSPORT_CARREFOUR** | Étoile Rouge, Carrefour 3 Banques | ~40 |
| **TRANSPORT_ROUTE** | RNIE 1, Boulevard de la Marina | ~50 |
| **TRANSPORT_VEHICULE** | Zémidjan, Tokpa-tokpa, Kloboto | ~10 |
| **ORG_TRANSPORT** | ATT, Baobab Express, Gozem | ~15 |

### Considérations pour l'annotation

Les **suffixes toponymiques fon** récurrents facilitent la détection :
- **-codji/-kpodji** : vers l'eau/lagune
- **-kpota** : subdivision
- **-houé/-komey** : maison/quartier
- **-mey** : suffixe locatif

Les **variantes orthographiques** (avec/sans accents, tirets) doivent être normalisées : Dassa-Zoumè = Dassa-Zoumé = Dassa.

---

Ce référentiel constitue une base solide pour entraîner un modèle NER spécialisé dans le transport béninois, couvrant l'ensemble du territoire national avec une granularité adaptée aux usages quotidiens de navigation et de mobilité.