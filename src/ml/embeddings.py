"""Graph Embeddings Module for Lexigraph

Uses Node2Vec to create embeddings for nodes in the knowledge graph,
enabling similarity search and intelligent recommendations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

try:
    import networkx as nx
    from node2vec import Node2Vec
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    HAS_GRAPH_ML = True
except ImportError:
    HAS_GRAPH_ML = False


class GraphEmbeddings:
    """Generate and query Node2Vec embeddings for the knowledge graph"""
    
    def __init__(self, neo4j_client):
        """Initialize with a Neo4j client connection"""
        if not HAS_GRAPH_ML:
            raise ImportError(
                "Graph ML libraries not installed. Run: pip install node2vec networkx scikit-learn"
            )
        self.client = neo4j_client
        self.graph: Optional[nx.Graph] = None
        self.model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.node_labels: Dict[str, str] = {}  # node_id -> label
        
    def build_networkx_graph(self) -> nx.Graph:
        """Extract graph structure from Neo4j into NetworkX"""
        G = nx.Graph()
        
        # Get all nodes with their labels and properties
        nodes_query = """
        MATCH (n)
        WHERE n:Person OR n:Topic OR n:ActionItem OR n:Decision OR n:Meeting
        RETURN elementId(n) as id, labels(n) as labels, 
               COALESCE(n.name, n.title, n.description) as name
        """
        nodes = self.client.run_query(nodes_query)
        
        for node in nodes:
            node_id = node['id']
            label = node['labels'][0] if node['labels'] else 'Unknown'
            name = node.get('name', 'Unknown')
            G.add_node(node_id, label=label, name=name)
            self.node_labels[node_id] = label
            
        # Get all relationships
        rels_query = """
        MATCH (a)-[r]->(b)
        WHERE (a:Person OR a:Topic OR a:ActionItem OR a:Decision OR a:Meeting)
          AND (b:Person OR b:Topic OR b:ActionItem OR b:Decision OR b:Meeting)
        RETURN elementId(a) as source, elementId(b) as target, type(r) as rel_type
        """
        rels = self.client.run_query(rels_query)
        
        for rel in rels:
            G.add_edge(rel['source'], rel['target'], rel_type=rel['rel_type'])
            
        self.graph = G
        return G
    
    def generate_embeddings(
        self, 
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 100,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1
    ) -> Dict[str, np.ndarray]:
        """Generate Node2Vec embeddings for all nodes"""
        if self.graph is None:
            self.build_networkx_graph()
            
        if len(self.graph.nodes()) == 0:
            return {}
            
        # Handle disconnected nodes by creating a minimal connected graph
        if not nx.is_connected(self.graph):
            # Add edges between disconnected components
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                self.graph.add_edge(node1, node2, rel_type='_EMBEDDING_LINK')
        
        # Generate embeddings using Node2Vec
        node2vec = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers,
            quiet=True
        )
        
        # Train the model
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # Store embeddings
        for node in self.graph.nodes():
            try:
                self.embeddings[node] = self.model.wv[node]
            except KeyError:
                continue
                
        return self.embeddings
    
    def find_similar_nodes(
        self, 
        node_id: str, 
        top_k: int = 5,
        filter_label: Optional[str] = None
    ) -> List[Tuple[str, str, float]]:
        """Find nodes most similar to the given node
        
        Returns list of (node_id, node_name, similarity_score)
        """
        if node_id not in self.embeddings:
            return []
            
        target_embedding = self.embeddings[node_id].reshape(1, -1)
        similarities = []
        
        for other_id, other_embedding in self.embeddings.items():
            if other_id == node_id:
                continue
            if filter_label and self.node_labels.get(other_id) != filter_label:
                continue
                
            sim = cosine_similarity(
                target_embedding, 
                other_embedding.reshape(1, -1)
            )[0][0]
            
            # Get node name
            name = self.graph.nodes[other_id].get('name', 'Unknown')
            similarities.append((other_id, name, float(sim)))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def find_similar_people(self, person_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find people who collaborate similarly to the given person"""
        # Find the person's node ID
        query = """
        MATCH (p:Person)
        WHERE toLower(p.name) CONTAINS toLower($name)
        RETURN elementId(p) as id, p.name as name
        LIMIT 1
        """
        result = self.client.run_query(query, {"name": person_name})
        
        if not result:
            return []
            
        person_id = result[0]['id']
        similar = self.find_similar_nodes(person_id, top_k + 1, filter_label='Person')
        
        # Return just name and similarity (exclude the person themselves)
        return [(name, sim) for _, name, sim in similar if name.lower() != person_name.lower()][:top_k]
    
    def cluster_topics(self, n_clusters: int = 3) -> Dict[int, List[str]]:
        """Cluster topics based on their embeddings"""
        # Get topic embeddings only
        topic_embeddings = []
        topic_ids = []
        
        for node_id, label in self.node_labels.items():
            if label == 'Topic' and node_id in self.embeddings:
                topic_embeddings.append(self.embeddings[node_id])
                topic_ids.append(node_id)
                
        if len(topic_embeddings) < n_clusters:
            return {}
            
        # Cluster using K-means
        X = np.array(topic_embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Group topics by cluster
        clusters = defaultdict(list)
        for i, cluster_label in enumerate(labels):
            node_id = topic_ids[i]
            topic_name = self.graph.nodes[node_id].get('name', 'Unknown')
            clusters[int(cluster_label)].append(topic_name)
            
        return dict(clusters)
    
    def suggest_task_owner(self, task_description: str) -> List[Tuple[str, float, str]]:
        """Suggest who should own a task based on similar past assignments
        
        Returns list of (person_name, confidence, reason)
        """
        # Extract meaningful keywords (skip common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been',
                      'set', 'up', 'get', 'do', 'make', 'create', 'add', 'update', 'who', 'should'}
        
        words = task_description.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2][:5]
        
        if not keywords:
            keywords = words[:3]  # Fallback to first 3 words
        
        # Find people who own similar tasks
        person_scores = defaultdict(float)
        person_tasks = defaultdict(list)
        person_topics = defaultdict(set)
        person_roles = {}
        
        for keyword in keywords:
            # Search by person role (e.g., "mobile" finds "Mobile Lead")
            role_query = """
            MATCH (p:Person)
            WHERE toLower(p.role) CONTAINS toLower($keyword)
            RETURN p.name as person, p.role as role
            """
            role_results = self.client.run_query(role_query, {"keyword": keyword})
            
            for r in role_results:
                person_scores[r['person']] += 5.0  # Highest weight for role match
                person_roles[r['person']] = r['role']
            
            # Search action items
            action_query = """
            MATCH (p:Person)-[:OWNS]->(a:ActionItem)
            WHERE toLower(a.description) CONTAINS toLower($keyword)
            RETURN p.name as person, a.description as task
            """
            action_results = self.client.run_query(action_query, {"keyword": keyword})
            
            for r in action_results:
                person_scores[r['person']] += 2.0  # Higher weight for action items
                if r['task'] not in person_tasks[r['person']]:
                    person_tasks[r['person']].append(r['task'])
            
            # Also search decisions and topics they're involved with
            topic_query = """
            MATCH (p:Person)-[:ATTENDED]->(m:Meeting)-[:DISCUSSED]->(t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($keyword)
            RETURN DISTINCT p.name as person, t.name as topic
            """
            topic_results = self.client.run_query(topic_query, {"keyword": keyword})
            
            for r in topic_results:
                person_scores[r['person']] += 1.0  # Lower weight for topic involvement
                person_topics[r['person']].add(r['topic'])
        
        # If no matches found, show people with most action items as fallback
        if not person_scores:
            fallback_query = """
            MATCH (p:Person)-[:OWNS]->(a:ActionItem)
            WITH p.name as person, count(a) as task_count, collect(a.description)[0..2] as tasks
            ORDER BY task_count DESC
            LIMIT 3
            RETURN person, task_count, tasks
            """
            fallback = self.client.run_query(fallback_query)
            return [
                (r['person'], r['task_count'] / 10.0, f"Active contributor with {r['task_count']} tasks")
                for r in fallback
            ]
        
        # Convert to list and sort
        suggestions = []
        max_score = max(person_scores.values()) if person_scores else 1
        
        for person, score in person_scores.items():
            normalized_score = score / max_score
            
            # Build reason - prioritize role match
            if person in person_roles:
                reason = f"Role: {person_roles[person]}"
            elif person_tasks[person]:
                reason = f"Owns similar: {person_tasks[person][0][:40]}..."
            elif person_topics[person]:
                reason = f"Works on: {', '.join(list(person_topics[person])[:2])}"
            else:
                reason = "Active team member"
                
            suggestions.append((person, normalized_score, reason))
            
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:3]
    
    def get_collaboration_strength(self) -> List[Tuple[str, str, int]]:
        """Get collaboration strength between people based on shared meetings/topics"""
        query = """
        MATCH (p1:Person)-[:ATTENDED]->(m:Meeting)<-[:ATTENDED]-(p2:Person)
        WHERE id(p1) < id(p2)
        WITH p1.name as person1, p2.name as person2, count(m) as meetings
        RETURN person1, person2, meetings
        ORDER BY meetings DESC
        LIMIT 10
        """
        results = self.client.run_query(query)
        return [(r['person1'], r['person2'], r['meetings']) for r in results]
    
    def get_stats(self) -> Dict:
        """Get statistics about the embeddings"""
        return {
            "total_nodes": len(self.graph.nodes()) if self.graph else 0,
            "total_edges": len(self.graph.edges()) if self.graph else 0,
            "nodes_with_embeddings": len(self.embeddings),
            "embedding_dimensions": len(next(iter(self.embeddings.values()))) if self.embeddings else 0,
            "node_types": dict(
                (label, sum(1 for l in self.node_labels.values() if l == label))
                for label in set(self.node_labels.values())
            )
        }
