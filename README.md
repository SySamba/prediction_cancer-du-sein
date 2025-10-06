"# Prédiction Cancer du Sein – Octobre Rose

Application web Flask pour prédire (démonstration) le risque bénin/malin à partir de 10 caractéristiques et sensibiliser à Octobre Rose. UI stylée en rose (#f5bbca + blanc), formulaires, page de résultats et mini-assistante de chat côté front.

## Fonctionnalités
- **Formulaire** de 10 champs numéraux essentiels.
- **Prédiction** via modèle TensorFlow (`.h5`) si présent, sinon fallback scikit-learn (`.pkl`) ou modèle de démonstration.
- **Pages**: `index` (formulaire), `result` (résultat + étapes suivantes).
- **Assistant**: widget de chat front avec infos Octobre Rose.

## Stack technique
- Python 3.11, Flask
- scikit-learn, numpy, joblib
- Gunicorn (prod)
- Dockerfile prêt pour Azure (App Service Linux / ACR)

## Structure du projet
```
predection-main/
├─ app.py                 # Application Flask (routes '/', '/predict_cancer', '/chat')
├─ requirements.txt       # Dépendances Python (inclut gunicorn)
├─ Dockerfile             # Image prod (Gunicorn, PORT)
├─ .dockerignore          # Réduit le contexte de build
├─ templates/
│  ├─ index.html          # Formulaire
│  └─ result.html         # Résultats, actions
└─ static/
   └─ style.css           # Thème #f5bbca + blanc
```

## Lancer en local (sans Docker)
1) Créer un venv et installer les deps:
```bash
python -m venv .venv
. .venv/bin/activate    # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2) Optionnel: déposer votre modèle à la racine:
- `model_breast_cancer.h5` (TensorFlow) ou
- `model_breast_cancer.pkl` (scikit-learn)

3) Démarrer:
```bash
python app.py
```
Ouvrez http://localhost:5000

## Exécuter avec Docker (local)
Build:
```bash
docker build -t breast-app:latest .
```
Run (port 8000):
```bash
docker run -e PORT=8000 -p 8000:8000 breast-app:latest
```
Ouvrez http://localhost:8000

## Image Docker publique
- Docker Hub: `papasy/breast-app:latest`

Push (rappel):
```bash
docker tag breast-app:latest papasy/breast-app:latest
docker push papasy/breast-app:latest
```

## Déploiement Azure – Option 1 (Web App + Docker Hub)
Prérequis: `az login`
```powershell
$RG="rg-breast"
$LOC="westeurope"
$PLAN="plan-breast"
$APP="web-breast-app"
$IMAGE="papasy/breast-app:latest"

az group create -n $RG -l $LOC
az appservice plan create -n $PLAN -g $RG --is-linux --sku B1
az webapp create -n $APP -g $RG --plan $PLAN `
  --deployment-container-image-name $IMAGE

az webapp config appsettings set -n $APP -g $RG --settings PORT=8000 WEBSITES_PORT=8000
az webapp restart -n $APP -g $RG
az webapp show -n $APP -g $RG --query defaultHostName -o tsv
```

Si le repo Docker Hub est privé, ajoutez les identifiants:
```powershell
az webapp config container set -n $APP -g $RG `
  --docker-custom-image-name $IMAGE `
  --docker-registry-server-url https://index.docker.io/v1/ `
  --docker-registry-server-user papasy `
  --docker-registry-server-password "VOTRE_TOKEN_OU_MDP"
```

## Déploiement Azure – Option 2 (ACR build sans Docker local)
```powershell
az login
$RG="rg-breast"
$LOC="westeurope"
$ACR="monacrunique"

az group create -n $RG -l $LOC
az acr create -n $ACR -g $RG --sku Basic
az acr build -r $ACR -t breast-app:latest .

$FQDN=$(az acr show -n $ACR --query loginServer -o tsv)
az appservice plan create -n plan-breast -g $RG --is-linux --sku B1
az webapp create -n web-breast-app -g $RG --plan plan-breast `
  --deployment-container-image-name "$FQDN/breast-app:latest"

az webapp config appsettings set -n web-breast-app -g $RG --settings PORT=8000 WEBSITES_PORT=8000
```

## Variables d’environnement
- `PORT`: port d’écoute Gunicorn (défaut 8000 dans le conteneur)
- `WEBSITES_PORT`: requis par Azure App Service Linux pour router (doit matcher `PORT`)
- `GUNICORN_WORKERS` (optionnel, défaut 4)
- `GUNICORN_THREADS` (optionnel, défaut 8)

## Endpoints
- `GET /` – formulaire d’entrée
- `POST /predict_cancer` – prédiction (retourne la page résultat)
- `POST /chat` – endpoint JSON pour le widget de chat

## Notes modèles
Placez `model_breast_cancer.h5` ou `model_breast_cancer.pkl` à la racine. Sans modèle, un **modèle de démo** scikit-learn est entraîné en mémoire.

## Licence
Projet à but pédagogique/démonstration. Adaptez avant usage clinique.
