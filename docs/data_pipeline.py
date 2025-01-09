"""Script to generate data pipeline visualization."""
import graphviz

# Create a new directed graph
dot = graphviz.Digraph(comment='Movie Recommender Data Pipeline')
dot.attr(rankdir='LR')

# Add nodes
dot.node('A', 'MovieLens\nDataset')
dot.node('B', 'Data\nLoader')
dot.node('C', 'User-Item\nMatrix')
dot.node('D', 'Collaborative\nFilter')
dot.node('E', 'Item-Item\nSimilarity')
dot.node('F', 'Top-N\nRecommendations')
dot.node('G', 'FastAPI\nEndpoint')

# Add edges
dot.edge('A', 'B', 'CSV files')
dot.edge('B', 'C', 'transform')
dot.edge('C', 'D', 'train')
dot.edge('D', 'E', 'compute\nsimilarity')
dot.edge('E', 'F', 'rank\nsimilar items')
dot.edge('F', 'G', 'serve\nvia API')

# Save the visualization
dot.render('docs/data_pipeline', format='png', cleanup=True)
