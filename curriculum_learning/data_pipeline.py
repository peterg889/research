"""
Unified Data Pipeline for Curriculum Learning

Combines real data loading with curriculum strategies and statistical labeling.
Optimized for production use with comprehensive caching and parallel processing.
"""

import os
import json
import hashlib
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from transformers import AutoTokenizer
from bertopic import BERTopic
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import textstat
from tqdm import tqdm
import warnings

from config import Config


class CurriculumDataset(Dataset):
    """
    PyTorch Dataset with curriculum learning capabilities.
    
    Supports all 9 curriculum strategies with efficient data loading.
    """
    
    def __init__(self, texts: List[str], reading_levels: List[float], 
                 topics: List[int], tokenizer: AutoTokenizer, max_length: int = 256):
        
        self.texts = texts
        self.reading_levels = np.array(reading_levels)
        self.topics = np.array(topics)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-compute for efficiency
        self.difficulty_tiers = self._compute_difficulty_tiers()
        self.topic_sizes = self._compute_topic_sizes()
        
        # Define epoch-interleaving strategies
        self.EPOCH_INTERLEAVING_STRATEGIES = {
            "reading_topic_by_epoch",
            "reading_levels_by_epoch",
            "all_strategies_by_epoch"
        }
        
        print(f"Dataset: {len(texts)} samples")
        print(f"Reading levels: {self.reading_levels.min():.1f} - {self.reading_levels.max():.1f}")
        print(f"Topics: {len(set(topics))} unique")
    
    def is_epoch_interleaving_strategy(self, strategy: str) -> bool:
        """Check if a strategy requires epoch-based curriculum switching"""
        return strategy in self.EPOCH_INTERLEAVING_STRATEGIES
    
    def _compute_difficulty_tiers(self) -> np.ndarray:
        """Pre-compute difficulty tiers (quartiles)"""
        if len(self.reading_levels) == 0:
            return np.array([])
        
        quartiles = np.percentile(self.reading_levels, [25, 50, 75])
        tiers = np.zeros(len(self.reading_levels), dtype=int)
        
        tiers[self.reading_levels <= quartiles[0]] = 0  # Easy
        tiers[(self.reading_levels > quartiles[0]) & (self.reading_levels <= quartiles[1])] = 1
        tiers[(self.reading_levels > quartiles[1]) & (self.reading_levels <= quartiles[2])] = 2  
        tiers[self.reading_levels > quartiles[2]] = 3  # Hard
        
        return tiers
    
    def _compute_topic_sizes(self) -> Dict[int, int]:
        """Pre-compute topic sizes"""
        unique_topics, counts = np.unique(self.topics, return_counts=True)
        return dict(zip(unique_topics, counts))
    
    def get_curriculum_order(self, strategy: str, epoch: int = None) -> List[int]:
        """Get sample indices ordered by curriculum strategy
        
        Args:
            strategy: Curriculum strategy name
            epoch: Current epoch (required for epoch-interleaving strategies)
        """
        
        if strategy == "random":
            indices = list(range(len(self.texts)))
            np.random.shuffle(indices)
            return indices
        
        elif strategy == "reading_level_easy_to_hard":
            return np.argsort(self.reading_levels).tolist()
        
        elif strategy == "reading_level_hard_to_easy":
            return np.argsort(self.reading_levels)[::-1].tolist()
        
        elif strategy == "reading_level_staged":
            # Present each difficulty tier completely
            indices = []
            for tier in range(4):
                tier_indices = np.where(self.difficulty_tiers == tier)[0]
                np.random.shuffle(tier_indices)
                indices.extend(tier_indices.tolist())
            return indices
        
        elif strategy == "topic_sequential":
            # Present each topic completely
            indices = []
            for topic in sorted(set(self.topics)):
                topic_indices = np.where(self.topics == topic)[0]
                np.random.shuffle(topic_indices)
                indices.extend(topic_indices.tolist())
            return indices
        
        elif strategy == "topic_interleaved":
            # Interleave topics evenly
            topics = sorted(set(self.topics))
            topic_indices = {topic: np.where(self.topics == topic)[0].tolist() for topic in topics}
            
            # Shuffle within each topic
            for topic in topics:
                np.random.shuffle(topic_indices[topic])
            
            indices = []
            max_topic_size = max(len(topic_indices[topic]) for topic in topics)
            
            for i in range(max_topic_size):
                for topic in topics:
                    if i < len(topic_indices[topic]):
                        indices.append(topic_indices[topic][i])
            
            return indices
        
        elif strategy == "topic_largest_first":
            # Present largest topics first
            sorted_topics = sorted(self.topic_sizes.keys(), 
                                 key=lambda t: self.topic_sizes[t], reverse=True)
            indices = []
            for topic in sorted_topics:
                topic_indices = np.where(self.topics == topic)[0]
                np.random.shuffle(topic_indices)
                indices.extend(topic_indices.tolist())
            return indices
        
        elif strategy == "hybrid_reading_topic":
            # Sort by reading level first, then by topic
            level_topic_indices = []
            for idx in range(len(self.texts)):
                level_topic_indices.append((self.reading_levels[idx], self.topics[idx], idx))
            level_topic_indices.sort(key=lambda x: (x[0], x[1]))
            return [x[2] for x in level_topic_indices]
        
        elif strategy == "hybrid_topic_reading":
            # Sort by topic first, then by reading level
            topic_level_indices = []
            for idx in range(len(self.texts)):
                topic_level_indices.append((self.topics[idx], self.reading_levels[idx], idx))
            topic_level_indices.sort(key=lambda x: (x[0], x[1]))
            return [x[2] for x in topic_level_indices]
        
        # ===== EPOCH-INTERLEAVING STRATEGIES =====
        
        elif strategy == "reading_topic_by_epoch":
            # Alternate between reading level easy-to-hard and topic sequential by epoch
            if epoch is None:
                raise ValueError("Epoch-interleaving strategies require epoch parameter")
            
            if epoch % 2 == 0:
                # Even epochs: reading level easy-to-hard
                return np.argsort(self.reading_levels).tolist()
            else:
                # Odd epochs: topic sequential
                indices = []
                for topic in sorted(set(self.topics)):
                    topic_indices = np.where(self.topics == topic)[0]
                    np.random.shuffle(topic_indices)
                    indices.extend(topic_indices.tolist())
                return indices
        
        elif strategy == "reading_levels_by_epoch":
            # Cycle through different reading level strategies by epoch
            if epoch is None:
                raise ValueError("Epoch-interleaving strategies require epoch parameter")
                
            reading_strategies = [
                "reading_level_easy_to_hard",
                "reading_level_hard_to_easy", 
                "reading_level_staged"
            ]
            
            chosen_strategy = reading_strategies[epoch % len(reading_strategies)]
            return self.get_curriculum_order(chosen_strategy, epoch)
        
        elif strategy == "all_strategies_by_epoch":
            # Cycle through ALL non-interleaving strategies by epoch
            if epoch is None:
                raise ValueError("Epoch-interleaving strategies require epoch parameter")
                
            base_strategies = [
                "random",
                "reading_level_easy_to_hard",
                "reading_level_hard_to_easy",
                "reading_level_staged", 
                "topic_sequential",
                "topic_interleaved",
                "topic_largest_first",
                "hybrid_reading_topic",
                "hybrid_topic_reading"
            ]
            
            chosen_strategy = base_strategies[epoch % len(base_strategies)]
            return self.get_curriculum_order(chosen_strategy, epoch)
        
        else:
            raise ValueError(f"Unknown curriculum strategy: {strategy}")
    
    def create_dataloader(self, strategy: str, batch_size: int, epoch: int = None) -> DataLoader:
        """Create DataLoader with curriculum ordering
        
        Args:
            strategy: Curriculum strategy name
            batch_size: Batch size for DataLoader
            epoch: Current epoch (required for epoch-interleaving strategies)
        """
        indices = self.get_curriculum_order(strategy, epoch)
        subset = Subset(self, indices)
        
        # Optimize for CPU/GPU
        use_cuda = torch.cuda.is_available()
        num_workers = 4 if use_cuda else 0
        
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,  # Already ordered by curriculum
            num_workers=num_workers,
            pin_memory=use_cuda,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for MLM"""
        texts = [item['text'] for item in batch]
        
        # Tokenize batch
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple['CurriculumDataset', 'CurriculumDataset', 'CurriculumDataset']:
        """Split into train/val/test sets"""
        test_ratio = 1.0 - train_ratio - val_ratio
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        indices = list(range(len(self.texts)))
        train_indices, temp_indices = train_test_split(indices, test_size=1-train_ratio, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)
        
        def create_subset(indices):
            return CurriculumDataset(
                texts=[self.texts[i] for i in indices],
                reading_levels=[self.reading_levels[i] for i in indices],
                topics=[self.topics[i] for i in indices],
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
        
        return create_subset(train_indices), create_subset(val_indices), create_subset(test_indices)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            'num_samples': len(self.texts),
            'avg_text_length': np.mean([len(text) for text in self.texts]),
            'reading_level_stats': {
                'mean': float(np.mean(self.reading_levels)),
                'std': float(np.std(self.reading_levels)),
                'min': float(np.min(self.reading_levels)),
                'max': float(np.max(self.reading_levels))
            },
            'topic_stats': {
                'num_topics': len(set(self.topics)),
                'largest_topic_size': max(self.topic_sizes.values()) if self.topic_sizes else 0,
                'smallest_topic_size': min(self.topic_sizes.values()) if self.topic_sizes else 0,
                'avg_topic_size': np.mean(list(self.topic_sizes.values())) if self.topic_sizes else 0
            }
        }
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'reading_level': self.reading_levels[idx],
            'topic': self.topics[idx]
        }


class DataPipeline:
    """
    Unified data pipeline for curriculum learning.
    
    Handles data loading, processing, labeling, and caching.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # Generate cache key
        self.cache_key = self._generate_cache_key()
        self.cache_file = self.cache_dir / f"dataset_{self.cache_key}.pkl"
        
        print(f"Data Pipeline initialized")
        print(f"Cache: {self.cache_dir}")
        print(f"🔑 Cache key: {self.cache_key}")
    
    def _generate_cache_key(self) -> str:
        """Generate cache key from config"""
        key_data = {
            'num_samples': self.config.num_samples,
            'max_seq_length': self.config.max_seq_length,
            'use_bertopic': self.config.use_bertopic,
            'min_topic_size': self.config.min_topic_size
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def load(self) -> CurriculumDataset:
        """Load or create curriculum dataset"""
        
        # Check cache first
        if self.cache_file.exists():
            print("⚡ Loading from cache...")
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            return CurriculumDataset(
                texts=cached_data['texts'],
                reading_levels=cached_data['reading_levels'],
                topics=cached_data['topics'],
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length
            )
        
        # Create from scratch
        print("Creating dataset from scratch...")
        
        # Load raw data
        texts = self._load_raw_data()
        
        # Compute reading levels
        reading_levels = self._compute_reading_levels(texts)
        
        # Discover topics
        topics = self._discover_topics(texts)
        
        # Cache results
        cache_data = {
            'texts': texts,
            'reading_levels': reading_levels,
            'topics': topics
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Dataset created: {len(texts)} samples, {len(set(topics))} topics")
        
        return CurriculumDataset(
            texts=texts,
            reading_levels=reading_levels,
            topics=topics,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length
        )
    
    def _load_raw_data(self) -> List[str]:
        """Load raw text data from multiple sources"""
        print("📚 Loading raw datasets...")
        
        all_texts = []
        target_samples = self.config.num_samples
        
        # Calculate samples per source
        wikitext_samples = int(target_samples * 0.6)  # 60% WikiText
        agnews_samples = int(target_samples * 0.2)    # 20% AG News
        imdb_samples = target_samples - wikitext_samples - agnews_samples  # Remainder
        
        # Load WikiText
        try:
            print(f"  📖 WikiText ({wikitext_samples} samples)...")
            wiki_dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
            wiki_texts = [item['text'] for item in wiki_dataset if len(item['text'].strip()) > 50]
            
            if len(wiki_texts) > wikitext_samples:
                wiki_texts = np.random.choice(wiki_texts, wikitext_samples, replace=False).tolist()
            
            all_texts.extend(wiki_texts)
            print(f"    {len(wiki_texts)} samples")
            
        except Exception as e:
            print(f"    WikiText failed: {e}")
        
        # Load AG News
        try:
            print(f"  📰 AG News ({agnews_samples} samples)...")
            ag_dataset = load_dataset('ag_news', split='train')
            ag_texts = [item['text'] for item in ag_dataset if len(item['text'].strip()) > 20]
            
            if len(ag_texts) > agnews_samples:
                ag_texts = np.random.choice(ag_texts, agnews_samples, replace=False).tolist()
            
            all_texts.extend(ag_texts)
            print(f"    {len(ag_texts)} samples")
            
        except Exception as e:
            print(f"    AG News failed: {e}")
        
        # Load IMDB
        try:
            print(f"  🎬 IMDB ({imdb_samples} samples)...")
            imdb_dataset = load_dataset('imdb', split='train')
            imdb_texts = [item['text'] for item in imdb_dataset if len(item['text'].strip()) > 20]
            
            if len(imdb_texts) > imdb_samples:
                imdb_texts = np.random.choice(imdb_texts, imdb_samples, replace=False).tolist()
            
            all_texts.extend(imdb_texts)
            print(f"    {len(imdb_texts)} samples")
            
        except Exception as e:
            print(f"    IMDB failed: {e}")
        
        # Shuffle and trim to exact size
        np.random.shuffle(all_texts)
        all_texts = all_texts[:target_samples]
        
        print(f"Total: {len(all_texts)} samples")
        return all_texts
    
    def _compute_reading_levels(self, texts: List[str]) -> List[float]:
        """Compute Flesch-Kincaid reading levels with parallel processing"""
        print("Computing reading levels...")
        
        def compute_level(text):
            try:
                level = textstat.flesch_kincaid_grade(text)
                return max(0.0, min(20.0, level))  # Clamp to reasonable range
            except:
                return 8.0  # Default middle school level
        
        # Use parallel processing for large datasets
        if len(texts) > 1000:
            with ThreadPoolExecutor(max_workers=4) as executor:
                reading_levels = list(tqdm(
                    executor.map(compute_level, texts),
                    total=len(texts),
                    desc="Reading levels"
                ))
        else:
            reading_levels = [compute_level(text) for text in tqdm(texts, desc="Reading levels")]
        
        return reading_levels
    
    def _discover_topics(self, texts: List[str]) -> List[int]:
        """Discover semantic topics using BERTopic with scalability for large datasets"""
        
        if not self.config.use_bertopic:
            print("BERTopic disabled - using dummy topics")
            return [0] * len(texts)
        
        print(f"Discovering topics for {len(texts)} texts...")
        
        # For large datasets, use a sampling-based approach
        # Reduced sample sizes for memory efficiency
        SAMPLE_THRESHOLD = 30000  # Above this, we sample
        MAX_SAMPLE_SIZE = 10000   # Maximum texts to use for topic discovery (reduced from 25K)
        
        try:
            if len(texts) > SAMPLE_THRESHOLD:
                print(f"Large dataset detected. Using sampling-based topic discovery...")
                
                # Sample texts stratified by reading level for better representation
                sample_indices = self._stratified_sample(texts, MAX_SAMPLE_SIZE)
                sample_texts = [texts[i] for i in sample_indices]
                
                # Discover topics on sample
                print(f"  1. Discovering topics on {len(sample_texts)} sampled texts...")
                sample_topics = self._run_bertopic(sample_texts)
                
                # Clear memory after BERTopic
                del sample_texts
                gc.collect()
                
                # Train a lightweight classifier to extend topics to full dataset
                print(f"  2. Extending topics to full dataset...")
                all_topics = self._extend_topics_to_full_dataset(
                    texts, sample_indices, sample_topics
                )
                
                # Clear memory after extension
                del sample_indices, sample_topics
                gc.collect()
                
                print(f"  ✓ {len(set(all_topics))} topics discovered via sampling")
                return all_topics
            
            else:
                # Small dataset - run BERTopic directly
                topics = self._run_bertopic(texts)
                print(f"  ✓ {len(set(topics))} topics discovered")
                return topics
                
        except Exception as e:
            print(f"WARNING: BERTopic failed: {e}")
            print("Falling back to TF-IDF based topic clustering...")
            
            # Theoretically sound fallback: TF-IDF + K-means clustering
            try:
                topics = self._tfidf_topic_clustering(texts)
                print(f"  ✓ {len(set(topics))} topics discovered via TF-IDF clustering")
                return topics
            except Exception as e2:
                print(f"ERROR: Fallback also failed: {e2}")
                print("Using random topic assignment as last resort...")
                # Last resort: random topics (still maintains topic structure for curriculum)
                num_topics = max(10, len(texts) // 1000)  # Reasonable number of topics
                topics = np.random.randint(0, num_topics, size=len(texts)).tolist()
                return topics
    
    def _run_bertopic(self, texts: List[str]) -> List[int]:
        """Run BERTopic on a set of texts"""
        from sentence_transformers import SentenceTransformer
        import gc
        
        # Always use the better quality model
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Set lower batch size for embeddings to save memory in memory-efficient mode
        if self.config.memory_efficient:
            # Reduce batch size to save memory
            original_encode = embedding_model.encode
            embedding_model.encode = lambda x: original_encode(x, batch_size=32)
            print("  Using reduced embedding batch size (32) for memory efficiency")
        
        # Configure BERTopic with optimizations
        topic_model = BERTopic(
            language="english",
            calculate_probabilities=False,  # Much faster
            min_topic_size=self.config.min_topic_size,
            n_gram_range=(1, 2),
            embedding_model=embedding_model,
            verbose=False  # Less verbose for large runs
        )
        
        # Clear memory before fitting
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Fit and transform
        print("  Fitting BERTopic model...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            topics, _ = topic_model.fit_transform(texts)
        
        # Clear memory after fitting
        del topic_model, embedding_model
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Convert outliers (-1) to nearest topic
        topics = self._handle_outlier_topics(topics)
        
        # Final cleanup
        gc.collect()
        
        return topics
    
    def _stratified_sample(self, texts: List[str], sample_size: int) -> List[int]:
        """Stratified sampling to ensure representative sample"""
        # Calculate reading levels if not already done
        reading_levels = [textstat.flesch_kincaid_grade(text) for text in texts]
        
        # Stratify by reading level quartiles
        rl_array = np.array(reading_levels)
        quartiles = np.percentile(rl_array, [25, 50, 75])
        
        indices = []
        for i in range(4):
            if i == 0:
                mask = rl_array <= quartiles[0]
            elif i == 3:
                mask = rl_array > quartiles[2]
            else:
                mask = (rl_array > quartiles[i-1]) & (rl_array <= quartiles[i])
            
            quartile_indices = np.where(mask)[0]
            sample_per_quartile = sample_size // 4
            
            if len(quartile_indices) > sample_per_quartile:
                sampled = np.random.choice(quartile_indices, sample_per_quartile, replace=False)
            else:
                sampled = quartile_indices
            
            indices.extend(sampled)
        
        return indices
    
    def _extend_topics_to_full_dataset(self, texts: List[str], 
                                      sample_indices: List[int], 
                                      sample_topics: List[int]) -> List[int]:
        """Extend topics from sample to full dataset using efficient classification"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        import gc
        
        # Prepare training data
        sample_texts = [texts[i] for i in sample_indices]
        
        # Use TF-IDF for features (much faster than embeddings for large scale)
        print("    Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_sample = vectorizer.fit_transform(sample_texts)
        
        # Clear sample texts from memory
        del sample_texts
        gc.collect()
        
        # Train a fast classifier
        print("    Training topic classifier...")
        classifier = LogisticRegression(max_iter=1000, n_jobs=-1)  # Use all cores
        classifier.fit(X_sample, sample_topics)
        
        # Clear training data
        del X_sample
        gc.collect()
        
        # Predict topics for all texts in batches
        print("    Predicting topics for full dataset...")
        batch_size = 10000
        all_topics = np.zeros(len(texts), dtype=int)
        
        # Set known topics from sample
        for idx, topic in zip(sample_indices, sample_topics):
            all_topics[idx] = topic
        
        # Predict unknown topics
        unknown_indices = list(set(range(len(texts))) - set(sample_indices))
        
        for i in range(0, len(unknown_indices), batch_size):
            batch_indices = unknown_indices[i:i+batch_size]
            batch_texts = [texts[idx] for idx in batch_indices]
            
            X_batch = vectorizer.transform(batch_texts)
            batch_topics = classifier.predict(X_batch)
            
            for idx, topic in zip(batch_indices, batch_topics):
                all_topics[idx] = topic
            
            # Clear batch data to save memory
            del batch_texts, X_batch, batch_topics
            if i % (batch_size * 5) == 0:  # Every 5 batches
                gc.collect()
        
        # Final cleanup
        del vectorizer, classifier, unknown_indices
        gc.collect()
        
        return all_topics.tolist()
    
    def _tfidf_topic_clustering(self, texts: List[str]) -> List[int]:
        """Fallback: TF-IDF based topic clustering"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans
        
        print("  Using TF-IDF + MiniBatchKMeans clustering...")
        
        # TF-IDF with reasonable limits for large datasets
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            max_df=0.95,
            min_df=2
        )
        
        # Transform texts
        X = vectorizer.fit_transform(texts)
        
        # Determine number of topics
        num_topics = min(100, max(10, len(texts) // 1000))
        
        # Use MiniBatchKMeans for scalability
        kmeans = MiniBatchKMeans(
            n_clusters=num_topics,
            batch_size=1000,
            n_init=3,
            random_state=42
        )
        
        topics = kmeans.fit_predict(X)
        return topics.tolist()
    
    def _handle_outlier_topics(self, topics: List[int]) -> List[int]:
        """Handle outlier topics (-1) by assigning them to a separate outlier topic"""
        topics = np.array(topics)
        
        if -1 not in topics:
            return topics.tolist()
        
        # Find outliers
        outlier_mask = topics == -1
        
        if outlier_mask.sum() == len(topics):
            # All are outliers - assign to single topic 0
            return [0] * len(topics)
        
        # Assign outliers to a new topic number (max_topic + 1)
        # This preserves the information that these are outliers
        max_topic = topics[~outlier_mask].max()
        outlier_topic = max_topic + 1
        
        topics[outlier_mask] = outlier_topic
        
        print(f"    Assigned {outlier_mask.sum()} outliers to topic {outlier_topic}")
        
        return topics.tolist()


if __name__ == "__main__":
    # Test the data pipeline
    from config import debug_config
    
    print("🧪 Testing Data Pipeline...\n")
    
    # Debug configuration
    config = debug_config(num_samples=100)
    pipeline = DataPipeline(config)
    
    # Load dataset
    dataset = pipeline.load()
    
    # Test statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Samples: {stats['num_samples']}")
    print(f"  Reading level: {stats['reading_level_stats']['mean']:.1f} ± {stats['reading_level_stats']['std']:.1f}")
    print(f"  Topics: {stats['topic_stats']['num_topics']}")
    
    # Test curriculum strategies
    print(f"\nTesting Curriculum Strategies:")
    for strategy in ['random', 'reading_level_easy_to_hard', 'topic_sequential']:
        order = dataset.get_curriculum_order(strategy)
        print(f"  {strategy}: {len(order)} samples ordered")
    
    # Test data splitting
    train_ds, val_ds, test_ds = dataset.split()
    print(f"\nData Splits:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples") 
    print(f"  Test: {len(test_ds)} samples")
    
    print(f"\nData Pipeline test completed!")