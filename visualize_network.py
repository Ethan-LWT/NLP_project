import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import tensorflow as tf
from manim import Scene, Circle, Square, Arrow, Text, Create, FadeIn, Write
from manim import FadeOut, VGroup, MoveToTarget, RED, BLUE, GREEN, YELLOW, WHITE
from manim import config, ReplacementTransform

# Add project directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from text_vectorization import TextVectorizer
from model_builder import EncoderDecoder

class NeuralNetworkVisualization:
    """Main class for neural network visualization interfaces with Manim"""
    
    def __init__(self, model=None, vectorizer=None):
        self.model = model
        self.vectorizer = vectorizer
        self.layer_outputs = {}
        self.activation_models = {}
        self._build_activation_models()
        
    def load_model_and_vectorizer(self, model_path, vectorizer_path):
        """Load model and vectorizer from files"""
        self.model = tf.keras.models.load_model(model_path)
        self.vectorizer = TextVectorizer.load(vectorizer_path)
        self._build_activation_models()
        
    def _build_activation_models(self):
        """Build models to extract activations from each layer"""
        if self.model is None:
            return
            
        # Create models that output each layer's activations
        for layer in self.model.layers:
            if hasattr(layer, 'output'):
                # Create a model that will return the layer's output on given input
                self.activation_models[layer.name] = tf.keras.Model(
                    inputs=self.model.input, 
                    outputs=layer.output
                )
                
    def get_layer_activations(self, input_text):
        """Get activations for all layers given input text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first")
            
        # Vectorize the input text
        sequence = self.vectorizer.texts_to_sequences([input_text])
        
        # Get activations for each layer
        activations = {}
        for layer_name, activation_model in self.activation_models.items():
            activations[layer_name] = activation_model.predict(sequence)
            
        return activations
    
    def get_attention_weights(self, input_text):
        """Extract attention weights if the model has an attention layer"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first")
            
        # Find attention layers
        attention_layers = [layer for layer in self.model.layers 
                           if 'attention' in layer.name.lower()]
        
        if not attention_layers:
            return None
            
        # Vectorize the input text
        sequence = self.vectorizer.texts_to_sequences([input_text])
        
        # Create a model that outputs attention weights
        # This is specific to the attention mechanism used
        # Might need to adjust based on your actual model architecture
        attention_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[layer.output for layer in attention_layers]
        )
        
        attention_weights = attention_model.predict(sequence)
        return attention_weights
    
    def get_model_structure(self):
        """Get the structure of the model for visualization"""
        if self.model is None:
            raise ValueError("Model must be loaded first")
            
        structure = []
        for i, layer in enumerate(self.model.layers):
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'shape': str(layer.output_shape),
                'params': layer.count_params(),
                'index': i
            }
            structure.append(layer_info)
            
        return structure
    
    def get_word_embeddings(self, words, n_components=2):
        """Get the embeddings for given words"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first")
            
        # Find embedding layer
        embedding_layers = [layer for layer in self.model.layers 
                           if isinstance(layer, tf.keras.layers.Embedding)]
        
        if not embedding_layers:
            return None
            
        embedding_layer = embedding_layers[0]
        embeddings = embedding_layer.get_weights()[0]
        
        # Get indices for words
        word_indices = []
        found_words = []
        for word in words:
            if word.lower() in self.vectorizer.word_to_idx:
                word_indices.append(self.vectorizer.word_to_idx[word.lower()])
                found_words.append(word)
        
        if not word_indices:
            return None
            
        # Get embeddings for these words
        word_embeddings = embeddings[word_indices]
        
        # Reduce dimensions for visualization if needed
        if n_components < word_embeddings.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            word_embeddings = pca.fit_transform(word_embeddings)
            
        return found_words, word_embeddings


class ManimNeuralNetworkVisualization(Scene):
    """Manim scene for visualizing neural networks"""
    
    def __init__(self, network_vis=None, **kwargs):
        super().__init__(**kwargs)
        self.network_vis = network_vis or NeuralNetworkVisualization()
        self.layer_groups = {}
        self.attention_groups = {}
        
    def create_layer_representation(self, layer_info, position):
        """Create visual representation of a layer"""
        layer_type = layer_info['type']
        
        if 'Input' in layer_type:
            shape = Circle(radius=0.5, color=BLUE)
        elif 'Embedding' in layer_type:
            shape = Square(side_length=0.5, color=GREEN)
        elif 'LSTM' in layer_type or 'RNN' in layer_type or 'GRU' in layer_type:
            shape = Square(side_length=0.8, color=RED)
        elif 'Dense' in layer_type:
            shape = Circle(radius=0.5, color=YELLOW)
        elif 'Dropout' in layer_type:
            shape = Circle(radius=0.4, color=WHITE, fill_opacity=0.3)
        else:
            shape = Square(side_length=0.5, color=WHITE)
            
        # Add text with layer name
        label = Text(layer_info['name'], font_size=24)
        label.next_to(shape, direction=np.array([0, -1, 0]))
        
        # Group the shape and label
        group = VGroup(shape, label)
        group.move_to(position)
        
        return group
    
    def visualize_model_structure(self):
        """Visualize the structure of the neural network model"""
        if self.network_vis.model is None:
            self.add(Text("No model loaded", font_size=36))
            return
            
        structure = self.network_vis.get_model_structure()
        
        # Calculate positions for each layer
        n_layers = len(structure)
        x_spacing = 12.0 / (n_layers + 1)
        x_start = -6.0 + x_spacing
        
        # Create a VGroup for the whole network
        network_group = VGroup()
        
        # Create representations for each layer
        for i, layer_info in enumerate(structure):
            position = np.array([x_start + i * x_spacing, 0, 0])
            layer_group = self.create_layer_representation(layer_info, position)
            network_group.add(layer_group)
            self.layer_groups[layer_info['name']] = layer_group
            
        # Add connections between layers
        for i in range(len(structure) - 1):
            layer1 = self.layer_groups[structure[i]['name']][0]  # Get the shape part
            layer2 = self.layer_groups[structure[i+1]['name']][0]  # Get the shape part
            arrow = Arrow(start=layer1.get_right(), end=layer2.get_left(), buff=0.1)
            network_group.add(arrow)
            
        # Animate the creation of the network
        self.play(Create(network_group))
        self.wait(2)
        
        return network_group
    
    def visualize_text_processing(self, input_text):
        """Visualize how text is processed through the network"""
        if self.network_vis.model is None or self.network_vis.vectorizer is None:
            self.add(Text("Model or vectorizer not loaded", font_size=36))
            return
            
        # Tokenize the text
        words = input_text.lower().split()
        
        # Display input words
        word_texts = VGroup()
        for i, word in enumerate(words):
            word_text = Text(word, font_size=24)
            word_text.shift(np.array([-5, 3 - i * 0.5, 0]))
            word_texts.add(word_text)
            
        self.play(Write(word_texts))
        self.wait(1)
        
        # Vectorize the text
        sequence = self.network_vis.vectorizer.texts_to_sequences([input_text])[0]
        
        # Display vectorized sequence
        sequence_texts = VGroup()
        for i, idx in enumerate(sequence):
            if idx == 0:  # PAD token
                continue
            word = words[i] if i < len(words) else "<PAD>"
            idx_text = Text(f"{word} → {idx}", font_size=20)
            idx_text.shift(np.array([0, 3 - i * 0.5, 0]))
            sequence_texts.add(idx_text)
            
        self.play(
            ReplacementTransform(word_texts, sequence_texts)
        )
        self.wait(1)
        
        # Get layer activations
        activations = self.network_vis.get_layer_activations(input_text)
        
        # Visualize activations flowing through the network
        self.visualize_activations_flow(activations)
        
        return sequence_texts
    
    def visualize_activations_flow(self, activations):
        """Visualize activations flowing through each layer"""
        if not activations:
            return
            
        # For each layer with activations
        for layer_name, activation in activations.items():
            if layer_name not in self.layer_groups:
                continue
                
            layer_group = self.layer_groups[layer_name]
            
            # Create a visual representation of the activation
            act_shape = layer_group[0].copy()  # Copy the shape
            act_shape.scale(1.2)  # Make it slightly larger
            act_shape.set_color(RED)  # Set a different color
            
            # Animate the activation flowing through the layer
            self.play(
                FadeIn(act_shape),
                FadeOut(act_shape, rate_func=lambda t: min(1, 1.5*t))
            )
    
    def visualize_attention(self, input_text):
        """Visualize attention mechanism if available"""
        attention_weights = self.network_vis.get_attention_weights(input_text)
        if attention_weights is None:
            return
            
        # Tokenize the text
        words = input_text.lower().split()
        
        # For simplicity, visualize the first attention mechanism's weights
        attn_weights = attention_weights[0][0]  # Shape could be [batch, heads, seq_len, seq_len]
        
        # Create a heatmap of attention weights
        plt.figure(figsize=(8, 6))
        plt.imshow(attn_weights, cmap='viridis')
        plt.colorbar()
        plt.xticks(range(len(words)), words, rotation=90)
        plt.yticks(range(len(words)), words)
        plt.title("Attention Weights")
        plt.tight_layout()
        
        # Save to a temporary file and display
        temp_file = "temp_attention.png"
        plt.savefig(temp_file)
        plt.close()
        
        # Create and display the image in Manim
        from manim import ImageMobject
        attention_img = ImageMobject(temp_file)
        attention_img.scale(3)
        attention_img.to_edge(np.array([0, 1, 0]))
        
        self.play(FadeIn(attention_img))
        self.wait(2)
        self.play(FadeOut(attention_img))
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    def visualize_word_embeddings(self, words):
        """Visualize word embeddings in 2D space"""
        result = self.network_vis.get_word_embeddings(words, n_components=2)
        if result is None:
            return
            
        found_words, embeddings = result
        
        # Create dots for each word in the embedding space
        word_dots = VGroup()
        
        for i, (word, embedding) in enumerate(zip(found_words, embeddings)):
            # Scale the embedding to fit the screen
            pos = np.array([embedding[0] * 3, embedding[1] * 3, 0])
            
            dot = Circle(radius=0.1, color=BLUE, fill_opacity=1)
            dot.move_to(pos)
            
            label = Text(word, font_size=20)
            label.next_to(dot, np.array([0, -0.5, 0]))
            
            word_group = VGroup(dot, label)
            word_dots.add(word_group)
            
        self.play(Create(word_dots))
        self.wait(2)
        
        return word_dots
    
    def construct(self):
        """Main construction method for the scene"""
        title = Text("Neural Network Visualization", font_size=48)
        title.to_edge(np.array([0, 1, 0]))
        self.play(Write(title))
        self.wait(1)
        
        # Load model and vectorizer if not already done
        if self.network_vis.model is None:
            model_path = os.getenv("MODEL_PATH", "models/final_model.h5")
            vectorizer_path = os.getenv("VECTORIZER_PATH", "models/vectorizer.pkl")
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                try:
                    self.network_vis.load_model_and_vectorizer(model_path, vectorizer_path)
                    self.play(Write(Text("Model and vectorizer loaded", font_size=36).shift(np.array([0, 1, 0]))))
                    self.wait(1)
                except Exception as e:
                    self.play(Write(Text(f"Error loading model: {str(e)}", font_size=36).shift(np.array([0, 1, 0]))))
                    self.wait(2)
                    return
            else:
                self.play(Write(Text("Model files not found", font_size=36).shift(np.array([0, 1, 0]))))
                self.wait(2)
                return
        
        # Visualize model structure
        network_group = self.visualize_model_structure()
        
        # Sample text for processing visualization
        input_text = "Natural language processing is fascinating"
        
        # Shrink network visualization to make room
        self.play(network_group.animate.scale(0.7).to_edge(np.array([0, -1, 0])))
        
        # Add text that shows what we're visualizing
        process_title = Text("Text Processing Visualization", font_size=36)
        process_title.to_edge(np.array([0, 1, 0]))
        
        self.play(
            FadeOut(title),
            Write(process_title)
        )
        
        # Visualize text processing
        sequence_texts = self.visualize_text_processing(input_text)
        
        # Visualize attention if available
        self.visualize_attention(input_text)
        
        # Visualize word embeddings
        embedding_title = Text("Word Embedding Visualization", font_size=36)
        embedding_title.to_edge(np.array([0, 1, 0]))
        
        self.play(
            FadeOut(process_title),
            FadeOut(sequence_texts),
            Write(embedding_title)
        )
        
        # Sample words for embedding visualization
        sample_words = ["natural", "language", "processing", "is", "fascinating", 
                        "data", "science", "neural", "network"]
        
        word_dots = self.visualize_word_embeddings(sample_words)
        
        # Final animation
        self.play(
            FadeOut(embedding_title),
            FadeOut(word_dots),
            FadeOut(network_group)
        )
        
        final_text = Text("Neural Network Visualization Complete", font_size=48)
        self.play(Write(final_text))
        self.wait(2)


def create_visualization(model_path=None, vectorizer_path=None, text=None, 
                         output_file=None, words=None, quality='medium_quality'):
    """Create visualization video using Manim"""
    # Set Manim configuration
    config.output_file = output_file or "neural_network_visualization"
    config.quality = quality
    
    # Initialize visualization
    network_vis = NeuralNetworkVisualization()
    
    # Load model and vectorizer if paths provided
    if model_path and vectorizer_path:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            network_vis.load_model_and_vectorizer(model_path, vectorizer_path)
    
    # Set environment variables for Manim scene
    if model_path:
        os.environ["MODEL_PATH"] = model_path
    if vectorizer_path:
        os.environ["VECTORIZER_PATH"] = vectorizer_path
    if text:
        os.environ["VISUALIZATION_TEXT"] = text
    if words:
        os.environ["VISUALIZATION_WORDS"] = ",".join(words)
    
    # Create and render the scene
    scene = ManimNeuralNetworkVisualization(network_vis=network_vis)
    scene.render()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Neural Network with Manim")
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    parser.add_argument('--vectorizer_path', type=str, help='Path to the vectorizer file')
    parser.add_argument('--text', type=str, default="Natural language processing is fascinating",
                        help='Text to visualize processing for')
    parser.add_argument('--output_file', type=str, default="neural_network_visualization",
                        help='Output filename (without extension)')
    parser.add_argument('--words', type=str, default="natural,language,processing,neural,network",
                        help='Comma-separated words for embedding visualization')
    parser.add_argument('--quality', type=str, default='medium_quality',
                        choices=['low_quality', 'medium_quality', 'high_quality'],
                        help='Rendering quality')
    
    args = parser.parse_args()
    
    # Process words into a list
    words_list = args.words.split(',') if args.words else None
    
    create_visualization(
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        text=args.text,
        output_file=args.output_file,
        words=words_list,
        quality=args.quality
    )
