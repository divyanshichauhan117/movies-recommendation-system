from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objs as go
import plotly.utils
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class MovieRecommendationSystem:
    def __init__(self):
        self.df = None
        self.scaler = None
        self.kmeans = None
        self.pca = None
        self.X_scaled = None
        self.X_pca = None
        self.cluster_labels = None
        self.optimal_k = 5
        
    def create_dataset(self):
        """Create the movie dataset"""
        np.random.seed(42)
        
        movie_titles = [
            "The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction", "Forrest Gump",
            "Inception", "The Matrix", "Goodfellas", "The Lord of the Rings", "Fight Club",
            "Star Wars", "Casablanca", "Schindler's List", "The Godfather Part II", "12 Angry Men",
            "One Flew Over the Cuckoo's Nest", "The Good, the Bad and the Ugly", "The Lord of the Rings: The Return of the King",
            "The Lord of the Rings: The Fellowship of the Ring", "The Lord of the Rings: The Two Towers",
            "Titanic", "Avatar", "The Avengers", "Jurassic Park", "Terminator 2", "Alien", "The Shining",
            "Apocalypse Now", "Gladiator", "Saving Private Ryan", "Interstellar", "The Prestige",
            "The Departed", "The Wolf of Wall Street", "Django Unchained", "Inglourious Basterds",
            "Kill Bill", "Reservoir Dogs", "The Silence of the Lambs", "Se7en", "The Usual Suspects",
            "Memento", "Eternal Sunshine of the Spotless Mind", "Her", "Lost in Translation", "The Grand Budapest Hotel",
            "Moonrise Kingdom", "The Royal Tenenbaums", "Rushmore", "Bottle Rocket", "The Life Aquatic",
            "Iron Man", "Captain America", "Thor", "Black Panther", "Spider-Man", "Batman Begins",
            "Man of Steel", "Wonder Woman", "Aquaman", "The Flash", "Green Lantern", "Suicide Squad",
            "Joker", "Deadpool", "X-Men", "Fantastic Four", "The Incredible Hulk", "Ant-Man",
            "Doctor Strange", "Guardians of the Galaxy", "Captain Marvel", "Black Widow", "Eternals",
            "Shang-Chi", "No Time to Die", "Skyfall", "Casino Royale", "Quantum of Solace",
            "GoldenEye", "Tomorrow Never Dies", "The World Is Not Enough", "Die Another Day",
            "Fast & Furious", "The Fast and the Furious", "2 Fast 2 Furious", "Fast Five",
            "Fast & Furious 6", "Furious 7", "The Fate of the Furious", "Hobbs & Shaw",
            "Mission: Impossible", "Mission: Impossible 2", "Mission: Impossible III", "Ghost Protocol",
            "Rogue Nation", "Fallout", "Top Gun", "Top Gun: Maverick", "Jerry Maguire",
            "A Few Good Men", "Rain Man", "Born on the Fourth of July", "The Color of Money",
            "The Pianist", "La La Land", "Whiplash", "Birdman", "The Revenant",
            "Mad Max: Fury Road", "Blade Runner 2049", "Dune", "Tenet", "Dunkirk"
        ]
        
        movies_data = {
            'movie_id': range(1, 101),
            'title': movie_titles,
            'genre_action': np.random.randint(0, 2, 100),
            'genre_comedy': np.random.randint(0, 2, 100),
            'genre_drama': np.random.randint(0, 2, 100),
            'genre_horror': np.random.randint(0, 2, 100),
            'genre_romance': np.random.randint(0, 2, 100),
            'rating': np.random.uniform(6.0, 9.5, 100),
            'year': np.random.randint(1990, 2024, 100),
            'duration': np.random.randint(90, 180, 100),
            'budget': np.random.uniform(5, 300, 100)
        }
        
        self.df = pd.DataFrame(movies_data)
        return self.df
    
    def prepare_data(self):
        """Prepare data for clustering"""
        features_for_clustering = ['genre_action', 'genre_comedy', 'genre_drama', 
                                  'genre_horror', 'genre_romance', 'rating', 
                                  'year', 'duration', 'budget']
        
        X = self.df[features_for_clustering].copy()
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        return self.X_scaled
    
    def find_optimal_clusters(self):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
        
        return K_range, inertias
    
    def apply_clustering(self):
        """Apply K-means clustering"""
        self.kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.X_scaled)
        self.df['cluster'] = self.cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.X_scaled, self.cluster_labels)
        
        return silhouette_avg
    
    def create_pca_visualization(self):
        """Create PCA visualization for clusters"""
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        return self.X_pca
    
    def get_cluster_summary(self, cluster_id):
        """Get summary of a specific cluster"""
        cluster_movies = self.df[self.df['cluster'] == cluster_id]
        
        summary = {
            'cluster_id': cluster_id,
            'movie_count': len(cluster_movies),
            'avg_rating': round(cluster_movies['rating'].mean(), 1),
            'avg_year': int(cluster_movies['year'].mean()),
            'avg_duration': int(cluster_movies['duration'].mean()),
            'avg_budget': round(cluster_movies['budget'].mean(), 1),
            'genres': {}
        }
        
        # Genre analysis
        genre_cols = ['genre_action', 'genre_comedy', 'genre_drama', 'genre_horror', 'genre_romance']
        for genre in genre_cols:
            pct = (cluster_movies[genre].sum() / len(cluster_movies)) * 100
            summary['genres'][genre.replace('genre_', '').title()] = round(pct, 1)
        
        return summary
    
    def recommend_movies(self, movie_title, n_recommendations=5):
        """Recommend movies based on cluster similarity"""
        try:
            movie_row = self.df[self.df['title'] == movie_title].iloc[0]
            movie_cluster = movie_row['cluster']
            
            # Find other movies in the same cluster
            same_cluster_movies = self.df[self.df['cluster'] == movie_cluster]
            recommendations = same_cluster_movies[same_cluster_movies['title'] != movie_title]
            
            # Sort by rating and get top N recommendations
            recommendations = recommendations.sort_values('rating', ascending=False)
            top_recommendations = recommendations.head(n_recommendations)
            
            result = {
                'input_movie': {
                    'title': movie_title,
                    'rating': round(movie_row['rating'], 1),
                    'year': int(movie_row['year']),
                    'cluster': int(movie_cluster),
                    'duration': int(movie_row['duration']),
                    'genres': self._get_movie_genres(movie_row)
                },
                'recommendations': []
            }
            
            for _, movie in top_recommendations.iterrows():
                rec = {
                    'title': movie['title'],
                    'rating': round(movie['rating'], 1),
                    'year': int(movie['year']),
                    'duration': int(movie['duration']),
                    'genres': self._get_movie_genres(movie)
                }
                result['recommendations'].append(rec)
            
            return result
            
        except IndexError:
            return None
    
    def _get_movie_genres(self, movie_row):
        """Extract genres for a movie"""
        genres = []
        genre_cols = ['genre_action', 'genre_comedy', 'genre_drama', 'genre_horror', 'genre_romance']
        for genre in genre_cols:
            if movie_row[genre] == 1:
                genres.append(genre.replace('genre_', '').title())
        return genres if genres else ['None']
    
    def get_all_movies(self):
        """Get list of all movies"""
        return sorted(self.df['title'].tolist())
    
    def get_cluster_stats(self):
        """Get statistics for all clusters"""
        stats = []
        for i in range(self.optimal_k):
            stats.append(self.get_cluster_summary(i))
        return stats

# Initialize the recommendation system
recommender = MovieRecommendationSystem()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize_system():
    """Initialize the recommendation system"""
    try:
        # Create dataset
        recommender.create_dataset()
        
        # Prepare data
        recommender.prepare_data()
        
        # Apply clustering
        silhouette_score = recommender.apply_clustering()
        
        # Create PCA visualization
        recommender.create_pca_visualization()
        
        # Get cluster statistics
        cluster_stats = recommender.get_cluster_stats()
        
        # Get all movies for dropdown
        all_movies = recommender.get_all_movies()
        
        return jsonify({
            'success': True,
            'message': 'System initialized successfully!',
            'silhouette_score': round(silhouette_score, 3),
            'cluster_stats': cluster_stats,
            'movies': all_movies,
            'total_movies': len(all_movies)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error initializing system: {str(e)}'
        })

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get movie recommendations"""
    try:
        data = request.get_json()
        movie_title = data.get('movie_title')
        n_recommendations = data.get('n_recommendations', 5)
        
        if not movie_title:
            return jsonify({
                'success': False,
                'message': 'Please provide a movie title'
            })
        
        recommendations = recommender.recommend_movies(movie_title, n_recommendations)
        
        if recommendations is None:
            return jsonify({
                'success': False,
                'message': f'Movie "{movie_title}" not found in database'
            })
        
        return jsonify({
            'success': True,
            'data': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting recommendations: {str(e)}'
        })

@app.route('/cluster-visualization')
def cluster_visualization():
    """Generate cluster visualization"""
    try:
        if recommender.X_pca is None:
            return jsonify({
                'success': False,
                'message': 'System not initialized'
            })
        
        # Create Plotly visualization
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        traces = []
        
        for i in range(recommender.optimal_k):
            cluster_points = recommender.X_pca[recommender.cluster_labels == i]
            traces.append(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(color=colors[i], size=8, opacity=0.6)
            ))
        
        layout = go.Layout(
            title='Movie Clusters Visualization (PCA)',
            xaxis=dict(title=f'PC1 ({recommender.pca.explained_variance_ratio_[0]:.1%} variance)'),
            yaxis=dict(title=f'PC2 ({recommender.pca.explained_variance_ratio_[1]:.1%} variance)'),
            hovermode='closest'
        )
        
        fig = go.Figure(data=traces, layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'plot': graphJSON
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error creating visualization: {str(e)}'
        })

@app.route('/elbow-curve')
def elbow_curve():
    """Generate elbow curve for optimal K"""
    try:
        if recommender.X_scaled is None:
            return jsonify({
                'success': False,
                'message': 'System not initialized'
            })
        
        K_range, inertias = recommender.find_optimal_clusters()
        
        trace = go.Scatter(
            x=list(K_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        )
        
        layout = go.Layout(
            title='Elbow Method for Optimal K',
            xaxis=dict(title='Number of Clusters (K)'),
            yaxis=dict(title='Inertia (Within-cluster sum of squares)'),
            hovermode='closest'
        )
        
        fig = go.Figure(data=[trace], layout=layout)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'plot': graphJSON
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error creating elbow curve: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)