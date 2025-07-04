import pandas as pd
import os

# Import all feature extractors
from features.basic_features import BasicFeatureExtractor
from features.char_diversity_features import CharDiversityFeatureExtractor
from features.pos_features import POSFeatureExtractor
from features.syntax_features import SyntaxFeatureExtractor
from features.dialogue_features import DialogueFeatureExtractor
from features.pronoun_features import PronounFeatureExtractor
from features.temporal_features import TemporalFeatureExtractor
from features.narrative_features import NarrativeFeatureExtractor
from features.academic_features import AcademicFeatureExtractor
from features.punctuation_features import PunctuationFeatureExtractor
from features.discourse_features import DiscourseFeatureExtractor
from features.readability_features import ReadabilityFeatureExtractor


class FeatureExtractor:
    """Main feature extractor that combines all feature groups"""
    
    def __init__(self, feature_groups='all'):
        """
        Initialize the feature extractor with specified feature groups.
        
        Parameters:
        -----------
        feature_groups : list or str
            List of feature groups to extract. Use 'all' for all features.
            Available groups: ['basic', 'char_diversity', 'pos', 'syntax', 
                             'dialogue', 'pronouns', 'temporal', 'narrative',
                             'academic', 'punctuation', 'discourse', 'readability']
        """
        self.available_groups = {
            'basic': BasicFeatureExtractor,
            'char_diversity': CharDiversityFeatureExtractor,
            'pos': POSFeatureExtractor,
            'syntax': SyntaxFeatureExtractor,
            'dialogue': DialogueFeatureExtractor,
            'pronouns': PronounFeatureExtractor,
            'temporal': TemporalFeatureExtractor,
            'narrative': NarrativeFeatureExtractor,
            'academic': AcademicFeatureExtractor,
            'punctuation': PunctuationFeatureExtractor,
            'discourse': DiscourseFeatureExtractor,
            'readability': ReadabilityFeatureExtractor
        }
        
        # Set feature groups to use
        if feature_groups == 'all':
            self.feature_groups = list(self.available_groups.keys())
        else:
            self.feature_groups = [g for g in feature_groups if g in self.available_groups]
        
        # Initialize selected extractors
        self.extractors = {}
        for group in self.feature_groups:
            self.extractors[group] = self.available_groups[group]()
    
    def extract_features(self, text):
        """Extract all selected features from a single text"""
        features = {}
        
        for group_name, extractor in self.extractors.items():
            try:
                group_features = extractor.extract(text)
                features.update(group_features)
            except Exception as e:
                print(f"Error extracting {group_name} features: {e}")
        
        return features
    
    def extract_features_from_dataframe(self, df, text_column, label_column=None, cache_file=None):
        """
        Extract features from a DataFrame
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing text data
        text_column : str
            Name of the column containing text
        label_column : str, optional
            Name of the label column
        cache_file : str, optional
            Path to cache extracted features
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with extracted features
        """
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            return pd.read_csv(cache_file)
        
        print(f"Extracting features from {len(df)} texts...")
        print(f"Selected feature groups: {self.feature_groups}")
        
        features_list = []
        
        for i, text in enumerate(df[text_column]):
            if i % 10 == 0:
                print(f"Processing text {i+1}/{len(df)}")
            
            try:
                features = self.extract_features(text)
                if label_column and label_column in df.columns:
                    features['label'] = df.iloc[i][label_column]
                features_list.append(features)
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                features_list.append({})
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        
        if cache_file:
            features_df.to_csv(cache_file, index=False)
            print(f"Saved features to {cache_file}")
        
        print(f"Extracted {features_df.shape[1]} features from {features_df.shape[0]} texts")
        
        return features_df
    
    def get_feature_names(self, group=None):
        """Get names of features for a specific group or all selected groups"""
        if group and group in self.extractors:
            dummy_text = "This is a sample text. It has two sentences."
            features = self.extractors[group].extract(dummy_text)
            return list(features.keys())
        else:
            # Return all features for selected groups
            dummy_text = "This is a sample text. It has two sentences."
            features = self.extract_features(dummy_text)
            return list(features.keys())