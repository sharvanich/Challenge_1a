import fitz
import re
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from skimage.filters import threshold_otsu
from collections import defaultdict
import joblib
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")


def filter_meaningful_lines(lines):
    filtered = []
    for line in lines:
        text = line["text"]

        # Basic filter: Remove empty or too short lines
        if len(text.strip()) < 2:
            continue

        # Remove standalone numbers (page numbers, list numbers)
        if re.fullmatch(r'\s*\d{1,3}\s*', text):
            continue

        # Remove URLs, emails, and similar patterns
        if re.search(r'(https?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+\.\w+)', text.lower()):
            continue

        # Remove lines with only special characters or whitespace
        if re.fullmatch(r'[\W\s]+', text):
            continue

        # Remove very short lines unless they resemble section numbers like "1.", "1.2", "1.2.3"
        if len(text.strip()) < 3 and not re.fullmatch(r'\d+(\.\d+)*\.?', text.strip()):
            continue

        filtered.append(line)

    return filtered


def extract_text(pdf_path):
    """Extract text with comprehensive metadata and layout features"""
    doc = fitz.open(pdf_path)
    lines = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height
        page_width = page.rect.width

        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue

            for line_idx, line in enumerate(block["lines"]):
                text_line = " ".join([span["text"] for span in line["spans"]])
                if not text_line.strip():
                    continue

                span = line["spans"][0] if line["spans"] else {}

                x0, y0, x1, y1 = line["bbox"]
                width = x1 - x0
                height = y1 - y0

                relative_y = y0 / page_height if page_height else 0

                line_data = {
                    "text": text_line.strip(),
                    "page": page_num,
                    "block_idx": block_idx,
                    "line_idx": line_idx,
                    "font_size": span.get("size", 12),
                    "font_name": span.get("font", ""),
                    "is_bold": "Bold" in span.get("font", ""),
                    "is_italic": "Italic" in span.get("font", ""),
                    "bbox": line["bbox"],
                    "bbox_x1": x0,
                    "bbox_y1": y0,
                    "bbox_x2": x1,
                    "bbox_y2": y1,
                    "content_length": len(text_line.strip()),
                    "relative_y_position": relative_y,
                    "is_top_of_page": relative_y < 0.2,
                    "is_bottom_of_page": relative_y > 0.8,
                    "aspect_ratio": width / height if height != 0 else 0,
                    "area": width * height,
                    "x": x0,
                    "y": y0,
                    "width": width,
                    "height": height,
                    "relative_x": x0 / page_width if page_width else 0,
                    "relative_y": relative_y,
                    "relative_width": width / page_width if page_width else 0,
                    "relative_height": height / page_height if page_height else 0,
                }
                lines.append(line_data)

    doc.close()
    lines = filter_meaningful_lines(lines)
    return lines


def extract_comprehensive_features(lines):
    """Extract comprehensive features for ML-based heading detection"""
    features = []
    
    for i, line in enumerate(lines):
        prev_line = lines[i-1] if i > 0 else None
        next_line = lines[i+1] if i < len(lines)-1 else None
        
        # Basic text features
        text = line["text"]
        char_count = len(text)
        word_count = len(text.split())
        avg_word_length = char_count / word_count if word_count > 0 else 0
        sentence_count = len(re.split(r'[.!?]+', text))
        digit_ratio = sum(c.isdigit() for c in text) / char_count if char_count > 0 else 0
        whitespace_ratio = sum(c.isspace() for c in text) / char_count if char_count > 0 else 0
        
        # Pattern matching
        has_section_number = bool(re.match(r'^\d+(\.\d+)*\.?\s', text))
        has_roman_numeral = bool(re.match(r'^[IVX]+\.?\s', text))
        has_bullet_point = bool(re.match(r'^[•\-\*]\s', text))
        
        # Font and formatting features
        font_size = line["font_size"]
        is_bold = line["is_bold"]
        is_italic = line["is_italic"]
        
        # Calculate relative font size features
        all_font_sizes = [l["font_size"] for l in lines]
        font_size_percentile = np.percentile(all_font_sizes, 75) if all_font_sizes else font_size
        is_large_font = font_size > np.percentile(all_font_sizes, 75) if all_font_sizes else False
        is_very_large_font = font_size > np.percentile(all_font_sizes, 90) if all_font_sizes else False
        font_size_normalized = (font_size - min(all_font_sizes)) / (max(all_font_sizes) - min(all_font_sizes) + 1e-6) if all_font_sizes else 0.5
        
        # Layout features
        is_left_aligned = line["relative_x"] < 0.1
        is_centered = 0.4 < line["relative_x"] < 0.6
        is_right_aligned = line["relative_x"] > 0.9
        is_indented = line["relative_x"] > 0.05
        
        # Position features
        is_first_line = i == 0
        is_last_line = i == len(lines) - 1
        line_position_ratio = i / len(lines) if len(lines) > 1 else 0.5
        
        # Contextual features
        prev_font_size_diff = (font_size - prev_line["font_size"]) if prev_line else 0
        prev_is_bold = prev_line["is_bold"] if prev_line else False
        prev_same_page = (line["page"] == prev_line["page"]) if prev_line else False
        prev_y_gap = (line["y"] - prev_line["y"]) if prev_line and prev_same_page else 0
        
        next_font_size_diff = (next_line["font_size"] - font_size) if next_line else 0
        next_is_bold = next_line["is_bold"] if next_line else False
        next_same_page = (line["page"] == next_line["page"]) if next_line else False
        next_y_gap = (next_line["y"] - line["y"]) if next_line and next_same_page else 0
        
        # Line height and spacing
        line_height = line["height"]
        
        feature_dict = {
            # Text features
            'char_count': char_count,
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'sentence_count': sentence_count,
            'digit_ratio': digit_ratio,
            'whitespace_ratio': whitespace_ratio,
            
            # Pattern features
            'has_section_number': int(has_section_number),
            'has_roman_numeral': int(has_roman_numeral),
            'has_bullet_point': int(has_bullet_point),
            
            # Font features
            'font_size': font_size,
            'font_size_normalized': font_size_normalized,
            'font_size_percentile': font_size_percentile,
            'is_bold': int(is_bold),
            'is_italic': int(is_italic),
            'is_large_font': int(is_large_font),
            'is_very_large_font': int(is_very_large_font),
            
            # Layout features
            'relative_x': line["relative_x"],
            'relative_y': line["relative_y"],
            'relative_width': line["relative_width"],
            'is_left_aligned': int(is_left_aligned),
            'is_centered': int(is_centered),
            'is_right_aligned': int(is_right_aligned),
            'is_indented': int(is_indented),
            
            # Position features
            'is_first_line': int(is_first_line),
            'is_last_line': int(is_last_line),
            'line_position_ratio': line_position_ratio,
            'line_height': line_height,
            
            # Contextual features
            'prev_font_size_diff': prev_font_size_diff,
            'prev_is_bold': int(prev_is_bold),
            'prev_same_page': int(prev_same_page),
            'prev_y_gap': prev_y_gap,
            'next_font_size_diff': next_font_size_diff,
            'next_is_bold': int(next_is_bold),
            'next_same_page': int(next_same_page),
            'next_y_gap': next_y_gap,
            
            # Additional features for model compatibility
            'x_coord': line["x"],
            'y_coord': line["y"],
            'indentation': line["relative_x"],
            'line_spacing': prev_y_gap,
            'has_numbers': int(bool(re.search(r'\d', text))),
            'starts_with_number': int(bool(re.match(r'^\d', text))),
            'is_small_font': int(font_size < np.percentile(all_font_sizes, 25)) if all_font_sizes else 0,
            'page_id': line["page"]
        }
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)


def prepare_temporal_features(df):
    features = []

    for i in range(len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1] if i > 0 else row
        next_ = df.iloc[i + 1] if i < len(df) - 1 else row

        font_jump = abs(row["font_size"] - prev["font_size"])
        y_gap = row.get("prev_y_gap", 0)
        line_height = row.get("line_height", 0)
        is_bold = row.get("is_bold", 0)
        indent = row.get("is_indented", 0)

        features.append([font_jump, y_gap, line_height, is_bold, indent])

    return np.array(features)


def calculate_composite_heading_scores(features_df, X_scaled):
    """Unsupervised + Supervised layout-aware heading scoring with full-width + flat-line suppression"""

    scores = np.zeros(len(X_scaled))

    # Define features
    formatting_features = [
        'font_size_normalized', 'is_bold', 'is_large_font', 'is_very_large_font', 'relative_x',
        'relative_y', 'relative_width', 'line_height', 'is_italic', 'font_size_percentile'
    ]
    pattern_features = [
        'has_section_number', 'has_roman_numeral', 'has_bullet_point'
    ]
    contextual_features = [
        'is_left_aligned', 'is_centered', 'is_right_aligned', 'is_indented', 'is_first_line',
        'is_last_line', 'line_position_ratio', 'prev_font_size_diff', 'prev_is_bold',
        'prev_same_page', 'prev_y_gap', 'next_font_size_diff', 'next_is_bold',
        'next_same_page', 'next_y_gap'
    ]
    semantic_text_features = [
        'char_count', 'word_count', 'avg_word_length', 'sentence_count',
        'digit_ratio', 'whitespace_ratio'
    ]
    all_features = formatting_features + pattern_features + contextual_features + semantic_text_features

    # Normalize features per page
    def normalize_per_page(df, col):
        return df.groupby('page_id')[col].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))

    norm_df = pd.DataFrame()
    for feature in all_features:
        if feature in features_df.columns:
            if features_df[feature].dtype in [np.float64, np.int64]:
                norm_df[feature] = normalize_per_page(features_df, feature) if 'page_id' in features_df.columns else \
                    (features_df[feature] - features_df[feature].min()) / (features_df[feature].max() - features_df[feature].min() + 1e-6)
            else:
                norm_df[feature] = features_df[feature]

    # PCA for dimensionality reduction
    valid_norm = norm_df.dropna(axis=1)
    X_norm = valid_norm.values
    pca = PCA(n_components=1)
    pca_scores = pca.fit_transform(X_norm)
    pca_scores_norm = (pca_scores - pca_scores.min()) / (pca_scores.max() - pca_scores.min() + 1e-6)
    pca_scores_flat = pca_scores_norm.ravel()

    # Clustering to find heading-like cluster
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(X_norm)
    cluster_sizes = np.bincount(cluster_labels)
    heading_cluster = np.argmin(cluster_sizes)
    cluster_score = (cluster_labels == heading_cluster).astype(float)

    # Combine PCA + Cluster-based boost
    scores = 0.7 * pca_scores_flat + 0.3 * cluster_score

    # Suppress full-width lines
    if 'relative_x' in features_df.columns and 'relative_width' in features_df.columns:
        full_width_mask = (features_df["relative_x"] < 0.02) & (features_df["relative_width"] > 0.95)
        scores[full_width_mask] = 0.0

    # Suppress "flat paragraph-like lines"
    for i in range(1, len(features_df) - 1):
        curr = features_df.iloc[i]
        prev = features_df.iloc[i - 1]
        next_ = features_df.iloc[i + 1]

        same_length = (
            abs(curr['char_count'] - prev['char_count']) <= 2 and
            abs(curr['char_count'] - next_['char_count']) <= 2
        )
        same_indent = (
            curr.get("is_indented", 0) == prev.get("is_indented", 0) ==
            next_.get("is_indented", 0)
        )
        not_bold = curr.get("is_bold", 0) == 0

        if same_length and same_indent and not_bold:
            scores[i] = 0.0

    # Initialize default values for missing components
    temporal_scores = np.zeros(len(features_df))
    heading_prob = np.zeros(len(features_df))

    # Try to apply supervised model if available
    try:
        if os.path.exists("model.pkl") and os.path.exists("label_encoder.pkl"):
            model = joblib.load("model.pkl")
            label_encoder = joblib.load("label_encoder.pkl")
            model_features = [
                'font_size', 'is_bold', 'is_italic',
                'x_coord', 'y_coord', 'indentation', 'line_spacing', 'char_count',
                'has_numbers', 'starts_with_number', 'is_large_font', 'is_small_font'
            ]
            
            if hasattr(model, "predict_proba") and all(f in features_df.columns for f in model_features):
                X_model = features_df[model_features].copy()
                X_model = X_model.replace([np.inf, -np.inf], np.nan).fillna(0)

                proba = model.predict_proba(X_model)
                if 'H1' in label_encoder.classes_:
                    h1_index = list(label_encoder.classes_).index('H1')
                    heading_prob = proba[:, h1_index]
    except Exception:
        pass

    # Try HMM for temporal patterns
    try:
        temporal_X = prepare_temporal_features(features_df)
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(temporal_X)
        hmm_states = model.predict(temporal_X)
        state_means = model.means_
        heading_like_state = np.argmax(state_means[:, 0] + state_means[:, 1])
        temporal_scores = (hmm_states == heading_like_state).astype(float)
    except Exception:
        pass

    # Normalize all score components
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    temporal_scores_norm = (temporal_scores - temporal_scores.min()) / (temporal_scores.max() - temporal_scores.min() + 1e-6)
    heading_prob_norm = (heading_prob - heading_prob.min()) / (heading_prob.max() - heading_prob.min() + 1e-6)

    # Weighted combination
    final_scores = (
        0.55 * scores_norm +           
        0.25 * temporal_scores_norm +  
        0.20 * heading_prob_norm       
    )

    # Bonus for top-of-page, bold, large-font lines
    font_size_boost = features_df["font_size_normalized"].clip(lower=0.8, upper=1.0)
    top_of_page_bonus = (1 - features_df["relative_y"]).clip(lower=0.7, upper=1.0)
    is_bold_bonus = features_df["is_bold"].astype(float)

    bonus = 0.1 * (font_size_boost * top_of_page_bonus * is_bold_bonus)
    final_scores += bonus

    # Final normalization to [0, 100] range
    final_scores = (final_scores - final_scores.min()) / (final_scores.max() - final_scores.min() + 1e-6) * 100

    return final_scores


def identify_headings_with_ml(features_df):
    """Use unsupervised ML to identify headings"""
    if features_df is None or len(features_df) == 0:
        return [], []

    # Select features for heading detection
    feature_cols = [col for col in features_df.columns]
    X = features_df[feature_cols].fillna(0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate composite heading scores
    heading_scores = calculate_composite_heading_scores(features_df, X_scaled)

    # Select headings based on composite score
    otsu_threshold = threshold_otsu(np.array(heading_scores))
    heading_indices = [i for i, score in enumerate(heading_scores) if score > otsu_threshold]

    return heading_indices, heading_scores


def assign_levels_by_font_size(lines, heading_indices):
    font_sizes = [lines[i]["font_size"] for i in heading_indices]
    unique_sizes = sorted(set(font_sizes), reverse=True)

    levels = []
    for idx in heading_indices:
        font_size = lines[idx]["font_size"]
        level = unique_sizes.index(font_size) + 1
        level = min(level, 3)  # Cap at level 3
        levels.append(level)
    return levels


def convert_clusters_to_levels(cluster_labels, heading_indices, lines):
    cluster_font_sizes = defaultdict(list)
    for cluster_id, heading_idx in zip(cluster_labels, heading_indices):
        font_size = lines[heading_idx].get("font_size", 12)
        cluster_font_sizes[cluster_id].append(font_size)

    cluster_avg_sizes = {
        cid: np.mean(sizes) for cid, sizes in cluster_font_sizes.items()
    }
    sorted_clusters = sorted(cluster_avg_sizes.items(), key=lambda x: x[1], reverse=True)
    cluster_to_level = {cid: i + 1 for i, (cid, _) in enumerate(sorted_clusters)}
    levels = [cluster_to_level[label] for label in cluster_labels]
    return levels


def assign_levels_by_clustering(features_df, heading_indices, lines):
    if len(heading_indices) <= 1:
        return assign_levels_by_font_size(lines, heading_indices)
        
    selected_features = list(features_df.columns)
    clustering_features = features_df.loc[heading_indices, selected_features].fillna(0)
    X_scaled = MinMaxScaler().fit_transform(clustering_features)

    best_labels = None
    best_score = -1
    for n_clusters in range(2, min(4, len(heading_indices))):
        try:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
        except:
            continue

    if best_labels is None:
        return assign_levels_by_font_size(lines, heading_indices)

    return convert_clusters_to_levels(best_labels, heading_indices, lines)


def assign_hierarchy_levels(lines, features_df, heading_indices, min_heading_confidence=0.6):
    """Assign smart hierarchy levels using clustering, ML model, and layout-aware rules"""

    if not heading_indices:
        return []

    # Clustering-based level assignment
    levels = assign_levels_by_clustering(features_df, heading_indices, lines)

    # Enforce rule: first heading after title is always H1
    sorted_with_levels = sorted(zip(heading_indices, levels), key=lambda x: (lines[x[0]]["page"], x[0]))
    if sorted_with_levels:
        first_idx, _ = sorted_with_levels[0]
        for i in range(len(sorted_with_levels)):
            if sorted_with_levels[i][0] == first_idx:
                sorted_with_levels[i] = (first_idx, 1)
                break

    # Final adjustment: Ensure levels are capped and sensible
    final_levels = [lvl if lvl <= 3 else 3 for _, lvl in sorted_with_levels]

    return final_levels


def extract_document_title(lines):
    """Extract document title from the first few lines"""
    if not lines:
        return "Untitled Document"
    
    # Look for the title in the first few lines
    for i in range(min(5, len(lines))):
        line = lines[i]
        # Title is usually large font, bold, or at the top
        if (line["font_size"] > 14 or 
            line["is_bold"] or 
            line["relative_y"] < 0.1):
            return line["text"]
    
    # Fallback to first line
    return lines[0]["text"]


def process_pdf(pdf_path, output_path):
    """Main function to process a single PDF and generate outline"""
    try:
        print(f"Processing: {pdf_path}")
        
        # Step 1: Extract text with metadata
        print("Step 1: Extracting text with metadata...")
        lines = extract_text(pdf_path)
        print(f"Extracted {len(lines)} meaningful lines")

        if not lines:
            print("No meaningful content found!")
            return False

        # Step 2: Extract comprehensive features
        print("Step 2: Extracting comprehensive features...")
        features_df = extract_comprehensive_features(lines)
        print(f"Extracted {len(features_df.columns)} features per line")

        # Step 3: Identify headings using ML
        print("Step 3: Identifying headings using ML...")
        heading_indices, heading_scores = identify_headings_with_ml(features_df)
        print(f"Identified {len(heading_indices)} potential headings")

        if not heading_indices:
            print("No headings detected!")
            # Create minimal outline with title only
            title = extract_document_title(lines)
            output_data = {
                "title": title,
                "outline": []
            }
        else:
            # Step 4: Assign hierarchy levels
            print("Step 4: Assigning hierarchy levels...")
            levels = assign_hierarchy_levels(lines, features_df, heading_indices)

            # Step 5: Create outline
            print("Step 5: Creating outline...")
            title = extract_document_title(lines)
            outline = []
            
            for i, (heading_idx, level) in enumerate(zip(heading_indices, levels)):
                line = lines[heading_idx]
                outline.append({
                    "level": f"H{level}",
                    "text": line["text"],
                    "page": line["page"] + 1
                })

            output_data = {
                "title": title,
                "outline": outline
            }

        # Step 6: Save output
        print(f"Step 6: Saving output to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully processed {pdf_path}")
        print(f"Generated outline with {len(output_data.get('outline', []))} headings")
        return True

    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return False


def main():
    """Main function to process all PDFs in the input directory"""
    input_dir = "sample_dataset/pdfs"
    output_dir = "sample_dataset/outputs"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    success_count = 0
    
    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_file = os.path.splitext(pdf_file)[0] + ".json"
        output_path = os.path.join(output_dir, output_file)
        
        if process_pdf(pdf_path, output_path):
            success_count += 1
        
        print("-" * 60)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(pdf_files)} files")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()