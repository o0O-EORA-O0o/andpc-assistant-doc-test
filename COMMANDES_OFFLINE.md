# Lancement de la version offline

Cette version évite le téléchargement de modèles depuis Hugging Face.

## 1. Récupérer les derniers fichiers

```bash
git pull
```

## 2. Lancer l'application offline

```bash
streamlit run app/main_offline.py --server.port 8501 --server.address 0.0.0.0
```

## 3. Dans l'interface

1. Cliquer sur **Indexer les documents**.
2. Lancer une question de test.

## Questions de test

```text
Quels sont les points de vigilance pour contrôler une action rattachée à l'OP-001 ?
```

```text
L'action A-001 est-elle cohérente avec la durée minimale attendue ?
```

```text
Génère un rapport d'aide à l'évaluation pour l'action A-001.
```

## Limite

Cette version utilise un embedding local déterministe. Elle est suffisante pour valider l'architecture et l'interface, mais elle ne remplace pas BGE-M3 ou un vrai modèle d'embedding sur la future VM interne.
