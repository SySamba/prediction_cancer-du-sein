import os
import numpy as np
from flask import Flask, render_template, request
import joblib

# Initialiser l'application Flask
app = Flask(__name__)

# Fonction pour charger le modèle avec gestion d'erreur
def load_cancer_model():
    try:
        # Essayer de charger le modèle TensorFlow d'abord
        try:
            from tensorflow.keras.models import load_model
            model_cancer = load_model('model_breast_cancer.h5')
            return model_cancer, 'tensorflow'
        except ImportError:
            print("TensorFlow non disponible, recherche de modèle scikit-learn...")
            
        # Essayer de charger un modèle scikit-learn
        try:
            model_cancer = joblib.load('model_breast_cancer.pkl')
            return model_cancer, 'sklearn'
        except FileNotFoundError:
            print("Modèle scikit-learn non trouvé, création d'un modèle de démonstration...")
            
        # Créer un modèle de démonstration
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Modèle de démonstration pour le cancer (10 features)
        X_cancer, y_cancer = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        model_cancer = RandomForestClassifier(random_state=42)
        model_cancer.fit(X_cancer, y_cancer)
        
        return model_cancer, 'demo'
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, 'error'

# Charger le modèle
model_cancer, model_type = load_cancer_model()

# Routes de l'application
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    if model_cancer is None:
        return "Erreur : Le modèle de prédiction du cancer n'a pas pu être chargé."
    
    try:
        # Liste des 10 champs les plus essentiels
        field_names = [
            'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 
            'mean_compactness', 'mean_concavity', 'worst_radius', 'worst_texture',
            'worst_perimeter', 'worst_concavity'
        ]
        
        # Récupérer et valider les données du formulaire
        features = []
        for field_name in field_names:
            if field_name not in request.form:
                return f"Erreur : Le champ '{field_name}' est manquant."
            
            try:
                value = float(request.form[field_name])
                if value < 0:
                    return f"Erreur : Le champ '{field_name}' ne peut pas être négatif."
                features.append(value)
            except ValueError:
                return f"Erreur : Le champ '{field_name}' doit être un nombre valide."

        # Vérifier que nous avons exactement 10 caractéristiques
        if len(features) != 10:
            return f"Erreur : Nombre incorrect de caractéristiques. Attendu: 10, Reçu: {len(features)}"

        # Prédire avec le modèle
        features_array = np.array(features).reshape(1, -1)
        
        if model_type == 'tensorflow':
            prediction = model_cancer.predict(features_array)
            probability = prediction[0][0]
        else:  # sklearn ou demo
            prediction = model_cancer.predict(features_array)
            prediction_proba = model_cancer.predict_proba(features_array)
            probability = prediction_proba[0][1]  # Probabilité de la classe positive
        
        # Interpréter le résultat
        result = "Malin" if probability > 0.5 else "Bénin"
        confidence = f"{probability * 100:.1f}%" if probability > 0.5 else f"{(1 - probability) * 100:.1f}%"

        return render_template('result.html', 
                             result=f"{result} (Confiance: {confidence})", 
                             model="Cancer du Sein")
    
    except Exception as e:
        return f"Une erreur s'est produite lors de la prédiction du cancer: {str(e)}"

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    try:
        from flask import jsonify
        data = request.get_json()
        user_message = data.get('message', '').lower().strip()
        
        # Base de connaissances étendue sur le cancer du sein
        responses = {
            # Salutations
            'bonjour': "🎗️ Bonjour ! Je suis Rosa, votre assistante IA spécialisée dans la prévention du cancer du sein. Comment puis-je vous aider aujourd'hui ?",
            'salut': "🎗️ Salut ! Je suis Rosa, votre assistante pour Octobre Rose. Posez-moi vos questions sur le cancer du sein !",
            'hello': "🎗️ Hello ! I'm Rosa, your breast cancer awareness AI assistant. How can I help you today?",
            'bonsoir': "🌙 Bonsoir ! Je suis Rosa, disponible 24h/7j pour répondre à vos questions sur le cancer du sein.",
            'hi': "👋 Hi ! Je suis Rosa, votre guide pour la prévention du cancer du sein. Que voulez-vous savoir ?",
            
            # Symptômes détaillés
            'symptômes': "🔍 Les symptômes du cancer du sein peuvent inclure :\n• Une bosse dans le sein ou l'aisselle\n• Changement de taille ou forme du sein\n• Écoulement du mamelon\n• Changement de la peau (rougeur, capitonnage)\n• Douleur persistante\n• Rétraction du mamelon\n• Peau d'orange\n\n⚠️ Consultez un médecin si vous remarquez ces signes.",
            'symptome': "🔍 Les symptômes du cancer du sein peuvent inclure :\n• Une bosse dans le sein ou l'aisselle\n• Changement de taille ou forme du sein\n• Écoulement du mamelon\n• Changement de la peau (rougeur, capitonnage)\n• Douleur persistante\n• Rétraction du mamelon\n• Peau d'orange\n\n⚠️ Consultez un médecin si vous remarquez ces signes.",
            'bosse': "🔍 Une bosse peut être :\n• Dure et fixe (plus préoccupant)\n• Mobile et souple (souvent bénigne)\n• Douloureuse ou indolore\n\n⚠️ Toute nouvelle bosse doit être examinée par un médecin, même si elle semble bénigne.",
            'douleur': "💔 La douleur mammaire :\n• N'est PAS toujours un signe de cancer\n• Peut être liée au cycle hormonal\n• Doit être évaluée si persistante\n• Accompagnée d'autres symptômes = consultation urgente",
            
            # Causes et facteurs de risque étendus
            'causes': "🧬 Les causes du cancer du sein sont multiples :\n• Facteurs génétiques (5-10% des cas)\n• Hormones (œstrogènes, progestérone)\n• Âge (80% après 50 ans)\n• Antécédents familiaux\n• Mode de vie (alcool, tabac, sédentarité)\n• Radiations\n• Traitement hormonal substitutif\n\n💡 La plupart des cancers surviennent sans cause identifiable.",
            'cause': "🧬 Les causes du cancer du sein sont multiples :\n• Facteurs génétiques (5-10% des cas)\n• Hormones (œstrogènes, progestérone)\n• Âge (80% après 50 ans)\n• Antécédents familiaux\n• Mode de vie (alcool, tabac, sédentarité)\n• Radiations\n• Traitement hormonal substitutif\n\n💡 La plupart des cancers surviennent sans cause identifiable.",
            'génétique': "🧬 Facteurs génétiques :\n• Mutations BRCA1 et BRCA2 (risque 50-85%)\n• Syndrome de Li-Fraumeni\n• Mutation du gène TP53\n• Antécédents familiaux directs\n\n🔬 Test génétique recommandé si :\n• Plusieurs cas familiaux\n• Cancer avant 40 ans\n• Cancer bilatéral",
            'brca': "🧬 Mutations BRCA1/BRCA2 :\n• BRCA1 : 55-65% de risque de cancer du sein\n• BRCA2 : 45% de risque de cancer du sein\n• Aussi risque de cancer ovarien\n• Hérédité autosomique dominante\n\n💡 Options : surveillance renforcée ou chirurgie préventive",
            'hormones': "🌸 Impact hormonal :\n• Œstrogènes : stimulent certains cancers\n• Règles précoces (avant 12 ans)\n• Ménopause tardive (après 55 ans)\n• Nulliparité (pas d'enfants)\n• Premier enfant après 30 ans\n• Traitement hormonal substitutif\n\n⚖️ L'allaitement est protecteur !",
            'alcool': "🍷 Alcool et cancer du sein :\n• Augmente le risque de 7% par verre/jour\n• Métabolisme de l'alcool produit des toxines\n• Augmente les œstrogènes\n• Réduit l'absorption de folates\n\n🚫 Recommandation : maximum 1 verre/jour pour les femmes",
            'tabac': "🚬 Tabac et cancer du sein :\n• Risque augmenté de 10-20%\n• Particulièrement avant la première grossesse\n• Tabagisme passif aussi dangereux\n• Retarde la guérison après traitement\n\n🚭 Arrêter à tout âge est bénéfique !",
            
            # Prévention étendue
            'prévention': "🛡️ Pour prévenir le cancer du sein :\n• Maintenez un poids santé (IMC < 25)\n• Faites 150min d'exercice/semaine\n• Limitez l'alcool (max 1 verre/jour)\n• Évitez le tabac\n• Allaitez si possible (6+ mois)\n• Alimentation riche en fruits/légumes\n• Faites des auto-examens mensuels\n• Suivez les recommandations de dépistage",
            'prevention': "🛡️ Pour prévenir le cancer du sein :\n• Maintenez un poids santé (IMC < 25)\n• Faites 150min d'exercice/semaine\n• Limitez l'alcool (max 1 verre/jour)\n• Évitez le tabac\n• Allaitez si possible (6+ mois)\n• Alimentation riche en fruits/légumes\n• Faites des auto-examens mensuels\n• Suivez les recommandations de dépistage",
            'alimentation': "🥗 Alimentation protectrice :\n• Fruits et légumes (5 portions/jour)\n• Poissons gras (oméga-3)\n• Légumineuses et céréales complètes\n• Thé vert (antioxydants)\n• Curcuma et brocolis\n\n❌ Limitez :\n• Viandes rouges et charcuteries\n• Graisses saturées\n• Sucres raffinés\n• Aliments ultra-transformés",
            'exercice': "🏃‍♀️ Activité physique :\n• Réduit le risque de 20-30%\n• 150min d'activité modérée/semaine\n• Ou 75min d'activité intense/semaine\n• Marche rapide, natation, vélo\n• Renforce le système immunitaire\n• Régule les hormones\n\n💪 Commencez progressivement !",
            'sport': "🏃‍♀️ Activité physique :\n• Réduit le risque de 20-30%\n• 150min d'activité modérée/semaine\n• Ou 75min d'activité intense/semaine\n• Marche rapide, natation, vélo\n• Renforce le système immunitaire\n• Régule les hormones\n\n💪 Commencez progressivement !",
            
            # Dépistage détaillé
            'dépistage': "🏥 Recommandations de dépistage :\n• Auto-examen mensuel dès 20 ans\n• Examen clinique annuel dès 25 ans\n• Mammographie tous les 2 ans de 50-74 ans\n• IRM si haut risque génétique\n• Échographie complémentaire si seins denses\n\n📅 Parlez-en à votre médecin !",
            'depistage': "🏥 Recommandations de dépistage :\n• Auto-examen mensuel dès 20 ans\n• Examen clinique annuel dès 25 ans\n• Mammographie tous les 2 ans de 50-74 ans\n• IRM si haut risque génétique\n• Échographie complémentaire si seins denses\n\n📅 Parlez-en à votre médecin !",
            'mammographie': "📸 La mammographie :\n• Examen de référence après 50 ans\n• Détecte 85-90% des cancers\n• Rayons X faible dose\n• Peut être inconfortable mais rapide\n• Permet de détecter des lésions de 2-3mm\n\n⏰ Durée : 10-15 minutes",
            'irm': "🧲 IRM mammaire :\n• Réservée aux femmes à haut risque\n• Très sensible (détecte 95% des cancers)\n• Pas de rayons X\n• Nécessite injection de produit de contraste\n• Examen long (30-45 min)\n\n💡 Complément de la mammographie",
            
            # Auto-examen détaillé
            'auto-examen': "🤲 Auto-examen des seins :\n1. Devant un miroir, bras le long du corps\n2. Bras levés, observez les changements\n3. Allongée, palpez avec la pulpe des doigts\n4. Mouvements circulaires de l'extérieur vers le mamelon\n5. Vérifiez aussi les aisselles et clavicules\n6. Pressez délicatement le mamelon\n\n📅 À faire chaque mois, 7 jours après les règles",
            'palpation': "✋ Technique de palpation :\n• Utilisez la pulpe des 3 doigts du milieu\n• Mouvements circulaires, pression variable\n• Couvrez tout le sein (jusqu'aux côtes)\n• 3 niveaux de pression : léger, moyen, ferme\n• N'oubliez pas les aisselles\n\n🎯 Cherchez : bosses, épaississements, zones dures",
            
            # Facteurs de risque
            'facteurs de risque': "⚠️ Facteurs de risque :\n• Âge (risque augmente avec l'âge)\n• Antécédents familiaux (mère, sœur)\n• Mutations génétiques (BRCA1, BRCA2)\n• Antécédents personnels de cancer\n• Densité mammaire élevée\n• Exposition aux radiations\n• Facteurs hormonaux\n\n💡 80% des femmes avec un cancer du sein n'ont aucun facteur de risque familial !",
            'age': "👵 Âge et cancer du sein :\n• 80% des cas après 50 ans\n• Risque double tous les 10 ans\n• Pic d'incidence : 65-70 ans\n• Possible à tout âge (même rare avant 30 ans)\n\n📈 Âge = facteur de risque principal",
            'famille': "👨‍👩‍👧‍👦 Antécédents familiaux :\n• Mère ou sœur : risque x2\n• Plusieurs parentes : risque x3-4\n• Cancer avant 50 ans : plus préoccupant\n• Côté paternel aussi important\n• Cancer de l'ovaire dans la famille\n\n🧬 Pensez au conseil génétique si nécessaire",
            
            # Traitements
            'traitement': "💊 Traitements du cancer du sein :\n• Chirurgie (tumorectomie, mastectomie)\n• Chimiothérapie\n• Radiothérapie\n• Hormonothérapie\n• Thérapies ciblées (Herceptin)\n• Immunothérapie\n\n🎯 Traitement personnalisé selon le type de cancer",
            'chirurgie': "🏥 Chirurgie du cancer du sein :\n• Tumorectomie : conservation du sein\n• Mastectomie : ablation totale\n• Ganglion sentinelle\n• Curage axillaire si nécessaire\n• Reconstruction possible\n\n💡 85% des femmes peuvent conserver leur sein",
            
            # Informations générales
            'octobre rose': "🎗️ Octobre Rose est le mois de sensibilisation au cancer du sein !\n• Campagne mondiale de prévention\n• Encourager le dépistage précoce\n• Soutenir la recherche\n• Accompagner les patientes\n• Monuments illuminés en rose\n• Courses et événements solidaires\n\n💪 Ensemble, luttons contre le cancer du sein !",
            
            'statistiques': "📊 Statistiques importantes :\n• 1 femme sur 8 développera un cancer du sein\n• 59,000 nouveaux cas/an en France\n• 2ème cancer le plus fréquent chez la femme\n• Détecté tôt : 99% de survie à 5 ans\n• 87% de survie globale à 5 ans\n• Âge moyen au diagnostic : 63 ans\n\n🎯 Le dépistage précoce sauve des vies !",
            
            'survie': "💪 Taux de survie :\n• Stade 0-1 : 99% à 5 ans\n• Stade 2 : 93% à 5 ans\n• Stade 3 : 72% à 5 ans\n• Tous stades confondus : 87% à 5 ans\n\n🎯 Diagnostic précoce = meilleur pronostic !",
            
            # Aide et navigation
            'aide': "🤖 Je peux vous aider avec :\n• Symptômes et signes d'alerte\n• Causes et facteurs de risque\n• Prévention et mode de vie\n• Dépistage et examens\n• Auto-examen des seins\n• Traitements disponibles\n• Statistiques et pronostic\n• Octobre Rose\n\n💬 Tapez simplement votre question !",
            'help': "🤖 Je peux vous aider avec :\n• Symptômes et signes d'alerte\n• Causes et facteurs de risque\n• Prévention et mode de vie\n• Dépistage et examens\n• Auto-examen des seins\n• Traitements disponibles\n• Statistiques et pronostic\n• Octobre Rose\n\n💬 Tapez simplement votre question !",
            
            # Remerciements et au revoir
            'merci': "🌸 De rien ! Je suis là pour vous accompagner dans votre démarche de prévention. N'hésitez pas à me poser d'autres questions !",
            'thank you': "🌸 You're welcome! I'm here to support your breast cancer prevention journey. Feel free to ask more questions!",
            'au revoir': "👋 Au revoir ! Prenez soin de vous et n'oubliez pas : la prévention est votre meilleure alliée ! 🎗️",
            'bye': "👋 Goodbye! Take care and remember: prevention is your best ally! 🎗️",
            'à bientôt': "👋 À bientôt ! N'hésitez pas à revenir me voir pour toute question sur la prévention du cancer du sein ! 🎗️"
        }
        
        # Recherche de réponse
        response = "🤔 Je ne suis pas sûre de comprendre votre question. Essayez de me demander des informations sur :\n• Les symptômes\n• La prévention\n• L'auto-examen\n• Le dépistage\n• Les facteurs de risque\n• Octobre Rose\n\nOu tapez 'aide' pour voir toutes mes fonctionnalités ! 💕"
        
        for keyword, answer in responses.items():
            if keyword in user_message:
                response = answer
                break
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'response': f"😔 Désolée, j'ai rencontré une erreur : {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
