# Commandes utiles

## Lancer dans Codespaces

```bash
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

Puis ouvrir le port `8501` dans l'onglet **Ports** de Codespaces.

## Lancer avec Docker Compose

```bash
docker compose up
```

## Réinitialiser l'index

Depuis l'interface Streamlit :

1. cliquer sur **Réinitialiser l'index** ;
2. cliquer sur **Indexer les documents**.

## Tester Ollama en option

Codespaces n'est pas idéal pour tester un vrai modèle local. Cette étape est donc optionnelle.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

Dans un second terminal :

```bash
ollama pull llama3.2:1b
```

Puis, dans l'application, cocher **Utiliser Ollama si disponible**.

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
