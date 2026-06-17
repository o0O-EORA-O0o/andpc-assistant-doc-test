# Assistant documentaire ANDPC — test Codespaces

Prototype de test pour valider une architecture d'assistant documentaire interne de type RAG, sans donnée réelle ANDPC.

## Objectif

Ce dépôt permet de tester :

- un environnement Codespaces basé sur Debian ;
- une application Streamlit ;
- une chaîne d'ingestion documentaire ;
- une base vectorielle locale Chroma ;
- des embeddings multilingues légers ;
- des réponses sourcées ;
- des cas d'usage métier simulés : OP, fiche de cadrage, procédure POOL, vue action/session fictive.

## Ce que ce prototype ne fait pas encore

- pas de données ANDPC réelles ;
- pas de connexion à une vraie BDD ;
- pas d'accès à des pièces jointes réelles ;
- pas de modèle Mistral local obligatoire ;
- pas de test de charge multi-utilisateurs.

## Démarrage dans GitHub Codespaces

1. Ouvrir le dépôt dans GitHub.
2. Cliquer sur **Code** > **Codespaces** > **Create codespace on main**.
3. Attendre l'installation automatique des dépendances.
4. Lancer l'application :

```bash
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

5. Ouvrir le port `8501` depuis l'onglet **Ports** de Codespaces.
6. Dans l'application, cliquer sur **Indexer les documents**.
7. Poser une question métier.

## Questions de test

```text
Quels sont les points de vigilance pour contrôler une action rattachée à l'OP-001 ?
```

```text
L'action A-001 est-elle cohérente avec la durée minimale attendue ?
```

```text
Que faire si une action n'a pas d'objectif pédagogique explicite ?
```

```text
Génère un rapport d'aide à l'évaluation pour l'action A-001.
```

## Architecture testée

```text
Documents fictifs
    ↓
Extraction / découpage
    ↓
Embeddings multilingues
    ↓
Base vectorielle Chroma
    ↓
Recherche documentaire
    ↓
Réponse sourcée / rapport de test
```

## Architecture cible agence

La version agence pourra remplacer les composants de test par :

- VM Debian interne ;
- Docker Compose ;
- Qdrant ou Chroma ;
- BGE-M3 pour les embeddings ;
- modèle local Mistral/Ministral via Ollama ;
- interface Open WebUI, AnythingLLM ou interface interne dédiée ;
- corpus documentaire réel validé ;
- vue SQL limitée en version ultérieure.

## Règle de sécurité

Ne jamais déposer dans ce dépôt :

- données PS ;
- données ODPC réelles ;
- exports préprod ;
- pièces jointes réelles ;
- données de facturation ;
- IBAN ou données bancaires ;
- documentation interne confidentielle.

Ce dépôt sert uniquement à tester l'architecture avec des données fictives.
