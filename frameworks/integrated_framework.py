"""
Complete Integration: Macro + Geopolitical Risk Framework
Sequential application: Base → Macro → Geopolitical → Final
"""

from frameworks.macro_framework import calculate_macro_framework
from frameworks.geopolitical_framework import calculate_geopolitical_framework
import json
from datetime import datetime

def calculate_complete_risk_assessment() -> dict:
    """
    Calculate both frameworks and return complete risk assessment
    """
    print("\n" + "="*80)
    print("[EMOJI] COMPLETE MACRO + GEOPOLITICAL RISK ASSESSMENT")
    print("="*80)
    
    # Calculate both frameworks
    macro = calculate_macro_framework()
    geo = calculate_geopolitical_framework()
    
    return {
        'timestamp': str(datetime.now()),
        'macro': macro,
        'geopolitical': geo
    }

def display_complete_analysis(result: dict):
    """Display comprehensive analysis"""
    
    macro = result['macro']
    geo = result['geopolitical']
    
    print("\n" + "="*80)
    print("[EMOJI] MACRO ECONOMIC ENVIRONMENT")
    print("="*80)
    print(f"Regime:           {macro['regime']}")
    print(f"Multiplier:       {macro['multiplier']:.4f} (0.75-1.25)")
    print(f"Composite Score:  {macro['composite_score']:.2f}/100")
    print("\nFactor Scores:")
    for factor, score in macro['factors'].items():
        print(f"  • {factor.replace('_', ' ').title():<30} {score:>6.2f}/100")
    
    print("\n" + "="*80)
    print("[EMOJI]  GEOPOLITICAL RISK ENVIRONMENT")
    print("="*80)
    print(f"Risk Level:       {geo['risk_level']}")
    print(f"Risk Score:       {geo['risk_score']:.2f}/100")
    print("\nPillar Scores:")
    for pillar, score in geo['pillars'].items():
        print(f"  • {pillar.replace('_', ' ').title():<30} {score:>6.2f}/100")
    
    print("\n" + "="*80)
    print("[EMOJI] INTEGRATED RECOMMENDATIONS")
    print("="*80)
    
    # Combined interpretation
    if macro['regime'] == 'CRISIS' and geo['risk_level'] in ['SEVERE', 'EXTREME']:
        print("""
[EMOJI] MAXIMUM DEFENSE MODE
• Both macro and geopolitical risks elevated
• Reduce all risk assets significantly
• Increase cash, treasuries, gold
• Consider defensive sectors only
• Monitor daily for regime changes
        """)
    elif macro['regime'] == 'CRISIS' or geo['risk_level'] in ['HIGH', 'SEVERE', 'EXTREME']:
        print("""
🟡 DEFENSIVE POSITIONING
• Elevated risk from macro OR geopolitical factors
• Reduce cyclical exposure
• Add defensive hedges
• Maintain higher cash levels
• Focus on quality assets
        """)
    elif macro['regime'] == 'GOLDILOCKS' and geo['risk_level'] == 'LOW':
        print("""
🟢 AGGRESSIVE POSITIONING
• Favorable macro environment + low geopolitical risk
• Maximize growth exposure
• Consider leverage if appropriate
• Reduce defensive positions
• Take advantage of risk-on environment
        """)
    else:
        print("""
🟡 BALANCED POSITIONING
• Mixed signals from macro and geopolitical factors
• Maintain diversified exposure
• Keep defensive hedge active
• Be prepared to adjust
• Monitor both frameworks closely
        """)
    
    print("\n" + "="*80)
    print("[EMOJI] APPLICATION GUIDE")
    print("="*80)
    print(f"""
To apply these frameworks to your forecasts:

1. MACRO ADJUSTMENT (to forecasts):
   adjusted_forecast = base_forecast × {macro['multiplier']:.4f}
   
2. GEOPOLITICAL ADJUSTMENT (to confidence/position sizing):
   # For risk assets (stocks, EM, etc.):
   risk_penalty = (geo_score / 100) × exposure × max_penalty
   adjusted_confidence = confidence × (1 - risk_penalty)
   
   # For safe havens (gold, treasuries, etc.):
   risk_boost = (geo_score / 100) × |exposure| × max_boost
   adjusted_confidence = confidence × (1 + risk_boost)

3. FINAL POSITION:
   final_allocation = base_allocation × macro_multiplier × geo_adjusted_confidence

Current Risk Parameters:
[EMOJI] Macro Multiplier:    {macro['multiplier']:.4f}
[EMOJI] Geo Risk Score:      {geo['risk_score']:.2f}/100
[EMOJI] Combined Signal:     {macro['regime']} + {geo['risk_level']}
[EMOJI] Suggested Action:    {"DEFENSIVE" if macro['regime'] == 'CRISIS' or geo['risk_level'] in ['HIGH', 'SEVERE', 'EXTREME'] else "BALANCED" if macro['regime'] == 'TRANSITIONAL' else "AGGRESSIVE"}
    """)
    
    print("="*80 + "\n")

def export_results(result: dict, filename: str = 'complete_risk_assessment.json'):
    """Export complete results to JSON"""
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[EMOJI] Complete analysis saved to: {filename}\n")

if __name__ == "__main__":
    # Calculate complete assessment
    result = calculate_complete_risk_assessment()
    
    # Display analysis
    display_complete_analysis(result)
    
    # Export to JSON
    export_results(result)

