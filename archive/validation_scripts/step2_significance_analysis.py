#!/usr/bin/env python3
"""
Step 2: Identify Significant Features
Analyze statistical validation results to identify truly predictive indicators
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
from analysis.statistical_feature_validator import StatisticalFeatureValidator

def analyze_significant_features():
    """Analyze and identify statistically significant features"""
    print("🎯 STEP 2: IDENTIFY SIGNIFICANT FEATURES")
    print("=" * 50)
    
    # Load validation results
    try:
        with open('data/validation_results.json', 'r') as f:
            results = json.load(f)
        print("✅ Loaded validation results from Step 1")
    except FileNotFoundError:
        print("❌ Validation results not found - run Step 1 first")
        return None
    
    # Convert master results back to DataFrame
    master_df = pd.DataFrame(results['master_results'])
    category_analysis = results['category_analysis']
    overall = results['overall_summary']
    
    print(f"\n📊 OVERALL VALIDATION SUMMARY:")
    print(f"   Total Features: {overall['total_features']}")
    print(f"   Significant Features: {overall['significant_features']} ({overall['overall_significance_rate']:.1%})")
    print(f"   Validation Date: {overall['validation_date']}")
    
    # Analyze significant features
    significant_features = master_df[master_df['overall_significant'] == True]
    
    print(f"\n🔥 STATISTICALLY SIGNIFICANT FEATURES ({len(significant_features)}):")
    print("-" * 80)
    
    # Sort by validation score
    significant_features = significant_features.sort_values('validation_score', ascending=False)
    
    for i, (_, row) in enumerate(significant_features.iterrows()):
        score = row['validation_score']
        corr = row['avg_correlation']
        perm = row['importance_mean']
        cv_imp = row['improvement_pct']
        category = row['category']
        
        # Create significance indicators
        corr_sig = "✓" if row['corr_significant'] else "✗"
        perm_sig = "✓" if row['perm_significant'] else "✗"
        cv_sig = "✓" if row['cv_predictive'] else "✗"
        
        print(f"{i+1:2d}. {row['feature']:25s} | Score: {score:6.3f} | {category:8s}")
        print(f"     Corr: {corr:5.3f} ({corr_sig}) | Perm: {perm:5.3f} ({perm_sig}) | CV: {cv_imp:5.1f}% ({cv_sig})")
    
    # Category-wise analysis
    print(f"\n📈 CATEGORY PERFORMANCE ANALYSIS:")
    print("-" * 50)
    
    for category, analysis in category_analysis.items():
        total = analysis['total_features']
        significant = analysis['significant_features']
        rate = analysis['significance_rate']
        avg_score = analysis['avg_validation_score']
        top_feature = analysis['top_feature']
        top_score = analysis['top_score']
        
        print(f"🔸 {category.upper():12s}: {int(significant):2d}/{int(total):2d} significant ({rate:.1%})")
        print(f"   Avg Score: {avg_score:.3f} | Top: {top_feature} ({top_score:.3f})")
    
    # Identify non-significant features
    non_significant = master_df[master_df['overall_significant'] == False]
    
    if len(non_significant) > 0:
        print(f"\n⚠️ NON-SIGNIFICANT FEATURES ({len(non_significant)}):")
        for _, row in non_significant.iterrows():
            print(f"   • {row['feature']} (Score: {row['validation_score']:.3f})")
    
    # Feature importance tiers
    print(f"\n🏆 FEATURE IMPORTANCE TIERS:")
    print("-" * 30)
    
    # Top tier (top 10%)
    top_threshold = significant_features['validation_score'].quantile(0.9)
    top_tier = significant_features[significant_features['validation_score'] >= top_threshold]
    
    # High tier (top 25%)
    high_threshold = significant_features['validation_score'].quantile(0.75)
    high_tier = significant_features[significant_features['validation_score'] >= high_threshold]
    
    # Medium tier (top 50%)
    medium_threshold = significant_features['validation_score'].quantile(0.5)
    medium_tier = significant_features[significant_features['validation_score'] >= medium_threshold]
    
    print(f"🥇 TOP TIER (≥90th percentile): {len(top_tier)} features")
    for _, row in top_tier.iterrows():
        print(f"   • {row['feature']} (Score: {row['validation_score']:.3f})")
    
    print(f"\n🥈 HIGH TIER (≥75th percentile): {len(high_tier)} features")
    high_except_top = high_tier[~high_tier['feature'].isin(top_tier['feature'])]
    for _, row in high_except_top.iterrows():
        print(f"   • {row['feature']} (Score: {row['validation_score']:.3f})")
    
    print(f"\n🥉 MEDIUM TIER (≥50th percentile): {len(medium_tier)} features")
    medium_except_high = medium_tier[~medium_tier['feature'].isin(high_tier['feature'])]
    for _, row in medium_except_high.head(5).iterrows():  # Show first 5
        print(f"   • {row['feature']} (Score: {row['validation_score']:.3f})")
    if len(medium_except_high) > 5:
        print(f"   ... and {len(medium_except_high) - 5} more")
    
    # Recommendations for optimization
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 40)
    
    # Best performing category
    best_category = max(category_analysis.items(), key=lambda x: x[1]['avg_validation_score'])
    print(f"🎯 BEST PERFORMING CATEGORY: {best_category[0].upper()}")
    print(f"   Average Score: {best_category[1]['avg_validation_score']:.3f}")
    print(f"   Significance Rate: {best_category[1]['significance_rate']:.1%}")
    
    # Recommended feature sets
    print(f"\n📋 RECOMMENDED FEATURE SETS:")
    
    # Conservative set (only top tier)
    conservative_features = top_tier['feature'].tolist()
    print(f"🔒 CONSERVATIVE ({len(conservative_features)} features): Top-tier only")
    print(f"   Features: {', '.join(conservative_features[:3])}...")
    
    # Balanced set (high tier and above)
    balanced_features = high_tier['feature'].tolist()
    print(f"⚖️ BALANCED ({len(balanced_features)} features): High-tier and above")
    print(f"   Features: {', '.join(balanced_features[:5])}...")
    
    # Comprehensive set (all significant)
    comprehensive_features = significant_features['feature'].tolist()
    print(f"🌟 COMPREHENSIVE ({len(comprehensive_features)} features): All significant")
    print(f"   Features: {', '.join(comprehensive_features[:7])}...")
    
    # Save analysis results
    analysis_results = {
        'significant_features': significant_features.to_dict('records'),
        'non_significant_features': non_significant.to_dict('records'),
        'category_analysis': category_analysis,
        'feature_tiers': {
            'top_tier': top_tier['feature'].tolist(),
            'high_tier': high_tier['feature'].tolist(),
            'medium_tier': medium_tier['feature'].tolist()
        },
        'recommendations': {
            'conservative_set': conservative_features,
            'balanced_set': balanced_features,
            'comprehensive_set': comprehensive_features,
            'best_category': best_category[0]
        },
        'summary': {
            'total_features': len(master_df),
            'significant_count': len(significant_features),
            'significance_rate': len(significant_features) / len(master_df),
            'top_feature': significant_features.iloc[0]['feature'],
            'top_score': significant_features.iloc[0]['validation_score']
        }
    }
    
    with open('data/feature_significance_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to data/feature_significance_analysis.json")
    
    return analysis_results

if __name__ == "__main__":
    results = analyze_significant_features()
    if results:
        print(f"\n🎯 STEP 2 COMPLETE - Ready for Step 3: ML Model Optimization")
    else:
        print(f"\n❌ STEP 2 FAILED - Check data and retry")
