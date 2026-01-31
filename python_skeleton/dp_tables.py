'''
ENHANCED DP BETTING TABLES - Version 3.0
Pre-computed GTO bet sizes, fold frequencies, and bluff frequencies
Optimized for tournament/bracket play with polarized river strategy
'''

# ==================== BET SIZE TABLE ====================
# Format: (hand_strength, pot_category, street) → bet_size_as_fraction_of_pot
BET_SIZE_TABLE = {
    # RIVER (street 5+) - POLARIZED SIZING
    ('VERY_STRONG', 'SMALL', 'RIVER'): 0.80,
    ('VERY_STRONG', 'MEDIUM', 'RIVER'): 0.85,
    ('VERY_STRONG', 'LARGE', 'RIVER'): 0.90,
    
    ('STRONG', 'SMALL', 'RIVER'): 0.65,
    ('STRONG', 'MEDIUM', 'RIVER'): 0.70,
    ('STRONG', 'LARGE', 'RIVER'): 0.75,
    
    ('MEDIUM', 'SMALL', 'RIVER'): 0.00,  # Check medium hands
    ('MEDIUM', 'MEDIUM', 'RIVER'): 0.00,
    ('MEDIUM', 'LARGE', 'RIVER'): 0.00,
    
    ('WEAK', 'SMALL', 'RIVER'): 0.00,  # Bluffs handled separately
    ('WEAK', 'MEDIUM', 'RIVER'): 0.00,
    ('WEAK', 'LARGE', 'RIVER'): 0.00,
    
    # TURN (street 4)
    ('VERY_STRONG', 'SMALL', 'TURN'): 0.68,
    ('VERY_STRONG', 'MEDIUM', 'TURN'): 0.72,
    ('VERY_STRONG', 'LARGE', 'TURN'): 0.78,
    
    ('STRONG', 'SMALL', 'TURN'): 0.55,
    ('STRONG', 'MEDIUM', 'TURN'): 0.60,
    ('STRONG', 'LARGE', 'TURN'): 0.65,
    
    ('MEDIUM', 'SMALL', 'TURN'): 0.42,
    ('MEDIUM', 'MEDIUM', 'TURN'): 0.45,
    ('MEDIUM', 'LARGE', 'TURN'): 0.00,  # Check in large pots
    
    ('WEAK', 'SMALL', 'TURN'): 0.00,
    ('WEAK', 'MEDIUM', 'TURN'): 0.00,
    ('WEAK', 'LARGE', 'TURN'): 0.00,
    
    # FLOP (street 1-3)
    ('VERY_STRONG', 'SMALL', 'FLOP'): 0.58,
    ('VERY_STRONG', 'MEDIUM', 'FLOP'): 0.62,
    ('VERY_STRONG', 'LARGE', 'FLOP'): 0.68,
    
    ('STRONG', 'SMALL', 'FLOP'): 0.48,
    ('STRONG', 'MEDIUM', 'FLOP'): 0.52,
    ('STRONG', 'LARGE', 'FLOP'): 0.58,
    
    ('MEDIUM', 'SMALL', 'FLOP'): 0.38,
    ('MEDIUM', 'MEDIUM', 'FLOP'): 0.42,
    ('MEDIUM', 'LARGE', 'FLOP'): 0.00,
    
    ('WEAK', 'SMALL', 'FLOP'): 0.00,
    ('WEAK', 'MEDIUM', 'FLOP'): 0.00,
    ('WEAK', 'LARGE', 'FLOP'): 0.00,
}

# ==================== FOLD FREQUENCY TABLE ====================
# Format: (hand_strength, bet_size_category, street) → fold_frequency
# v5.4: LOWERED TO MEET MDF (Minimum Defense Frequency)
# MDF = pot / (pot + bet) - we must call at least this often to prevent profitable bluffs
# For a pot-sized bet, MDF = 50%. For 1.5x pot, MDF = 40%.
FOLD_FREQ_TABLE = {
    # vs HUGE bets (>100% pot) - v5.4: LOWERED significantly
    ('VERY_STRONG', 'HUGE', 'RIVER'): 0.00,
    ('STRONG', 'HUGE', 'RIVER'): 0.10,
    ('MEDIUM', 'HUGE', 'RIVER'): 0.45,  # Was 0.65 - now meets MDF
    ('WEAK', 'HUGE', 'RIVER'): 0.70,    # Was 0.92 - CRITICAL FIX
    
    ('VERY_STRONG', 'HUGE', 'TURN'): 0.00,
    ('STRONG', 'HUGE', 'TURN'): 0.15,
    ('MEDIUM', 'HUGE', 'TURN'): 0.42,   # Was 0.55
    ('WEAK', 'HUGE', 'TURN'): 0.68,     # Was 0.85
    
    ('VERY_STRONG', 'HUGE', 'FLOP'): 0.00,
    ('STRONG', 'HUGE', 'FLOP'): 0.10,
    ('MEDIUM', 'HUGE', 'FLOP'): 0.35,   # Was 0.45
    ('WEAK', 'HUGE', 'FLOP'): 0.62,     # Was 0.78
    
    # vs LARGE bets (75-100% pot) - v5.4: LOWERED
    ('VERY_STRONG', 'LARGE', 'RIVER'): 0.00,
    ('STRONG', 'LARGE', 'RIVER'): 0.08,
    ('MEDIUM', 'LARGE', 'RIVER'): 0.38,  # Was 0.55
    ('WEAK', 'LARGE', 'RIVER'): 0.65,    # Was 0.88 - CRITICAL FIX
    
    ('VERY_STRONG', 'LARGE', 'TURN'): 0.00,
    ('STRONG', 'LARGE', 'TURN'): 0.12,
    ('MEDIUM', 'LARGE', 'TURN'): 0.38,   # Was 0.48
    ('WEAK', 'LARGE', 'TURN'): 0.62,     # Was 0.80
    
    ('VERY_STRONG', 'LARGE', 'FLOP'): 0.00,
    ('STRONG', 'LARGE', 'FLOP'): 0.08,
    ('MEDIUM', 'LARGE', 'FLOP'): 0.30,   # Was 0.38
    ('WEAK', 'LARGE', 'FLOP'): 0.55,     # Was 0.70
    
    # vs MEDIUM bets (50-75% pot) - v5.4: LOWERED
    ('VERY_STRONG', 'MEDIUM', 'RIVER'): 0.00,
    ('STRONG', 'MEDIUM', 'RIVER'): 0.05,
    ('MEDIUM', 'MEDIUM', 'RIVER'): 0.32,  # Was 0.42
    ('WEAK', 'MEDIUM', 'RIVER'): 0.58,    # Was 0.78
    
    ('VERY_STRONG', 'MEDIUM', 'TURN'): 0.00,
    ('STRONG', 'MEDIUM', 'TURN'): 0.08,
    ('MEDIUM', 'MEDIUM', 'TURN'): 0.28,   # Was 0.35
    ('WEAK', 'MEDIUM', 'TURN'): 0.52,     # Was 0.70
    
    ('VERY_STRONG', 'MEDIUM', 'FLOP'): 0.00,
    ('STRONG', 'MEDIUM', 'FLOP'): 0.05,
    ('MEDIUM', 'MEDIUM', 'FLOP'): 0.22,   # Was 0.28
    ('WEAK', 'MEDIUM', 'FLOP'): 0.45,     # Was 0.60
    
    # vs SMALL bets (<50% pot) - v5.4: LOWERED
    ('VERY_STRONG', 'SMALL', 'RIVER'): 0.00,
    ('STRONG', 'SMALL', 'RIVER'): 0.02,
    ('MEDIUM', 'SMALL', 'RIVER'): 0.20,   # Was 0.28
    ('WEAK', 'SMALL', 'RIVER'): 0.45,     # Was 0.65
    
    ('VERY_STRONG', 'SMALL', 'TURN'): 0.00,
    ('STRONG', 'SMALL', 'TURN'): 0.02,
    ('MEDIUM', 'SMALL', 'TURN'): 0.15,    # Was 0.22
    ('WEAK', 'SMALL', 'TURN'): 0.38,      # Was 0.55
    
    ('VERY_STRONG', 'SMALL', 'FLOP'): 0.00,
    ('STRONG', 'SMALL', 'FLOP'): 0.02,
    ('MEDIUM', 'SMALL', 'FLOP'): 0.12,    # Was 0.18
    ('WEAK', 'SMALL', 'FLOP'): 0.32,      # Was 0.45
}

# ==================== BLUFF FREQUENCY TABLE ====================
# When to bluff with weak hands (frequency)
BLUFF_FREQ_TABLE = {
    # River - polarized bluffing
    ('WEAK', 'SMALL', 'RIVER'): 0.18,
    ('WEAK', 'MEDIUM', 'RIVER'): 0.15,
    ('WEAK', 'LARGE', 'RIVER'): 0.12,
    
    # Turn - more semi-bluffs
    ('WEAK', 'SMALL', 'TURN'): 0.22,
    ('WEAK', 'MEDIUM', 'TURN'): 0.18,
    ('WEAK', 'LARGE', 'TURN'): 0.12,
    
    # Flop - continuation betting
    ('WEAK', 'SMALL', 'FLOP'): 0.28,
    ('WEAK', 'MEDIUM', 'FLOP'): 0.22,
    ('WEAK', 'LARGE', 'FLOP'): 0.00,  # Don't bluff large pots on flop
}

# ==================== BLUFF SIZE TABLE ====================
# How big to bet when bluffing (as fraction of pot)
BLUFF_SIZE_TABLE = {
    ('SMALL', 'RIVER'): 0.80,   # Big bluffs on river
    ('MEDIUM', 'RIVER'): 0.85,
    ('LARGE', 'RIVER'): 0.90,
    
    ('SMALL', 'TURN'): 0.60,
    ('MEDIUM', 'TURN'): 0.65,
    ('LARGE', 'TURN'): 0.70,
    
    ('SMALL', 'FLOP'): 0.50,
    ('MEDIUM', 'FLOP'): 0.55,
    ('LARGE', 'FLOP'): 0.00,
}

# ==================== VALUE SIZE TABLE ====================
# Size for value bets (as fraction of pot)
VALUE_SIZE_TABLE = {
    ('VERY_STRONG', 'SMALL', 'RIVER'): 0.85,
    ('VERY_STRONG', 'MEDIUM', 'RIVER'): 0.90,
    ('VERY_STRONG', 'LARGE', 'RIVER'): 0.95,
    
    ('STRONG', 'SMALL', 'RIVER'): 0.68,
    ('STRONG', 'MEDIUM', 'RIVER'): 0.72,
    ('STRONG', 'LARGE', 'RIVER'): 0.78,
}

# ==================== HELPER FUNCTIONS ====================

def get_bet_size(hand_strength, pot_category, street_name):
    """Get GTO bet size from DP table"""
    key = (hand_strength, pot_category, street_name)
    return BET_SIZE_TABLE.get(key, 0.50)

def get_fold_frequency(hand_strength, bet_category, street_name):
    """Get GTO fold frequency from DP table"""
    key = (hand_strength, bet_category, street_name)
    return FOLD_FREQ_TABLE.get(key, 0.50)

def get_bluff_frequency(hand_strength, pot_category, street_name):
    """Get GTO bluff frequency from DP table"""
    key = (hand_strength, pot_category, street_name)
    return BLUFF_FREQ_TABLE.get(key, 0.10)

def get_bluff_size(pot_category, street_name):
    """Get bluff bet size"""
    key = (pot_category, street_name)
    return BLUFF_SIZE_TABLE.get(key, 0.70)

def get_value_size(hand_strength, pot_category, street_name):
    """Get value bet size"""
    key = (hand_strength, pot_category, street_name)
    return VALUE_SIZE_TABLE.get(key, get_bet_size(hand_strength, pot_category, street_name))

def categorize_hand_strength(equity):
    """Convert equity to hand strength category"""
    if equity >= 0.68:
        return 'VERY_STRONG'
    elif equity >= 0.52:
        return 'STRONG'
    elif equity >= 0.35:
        return 'MEDIUM'
    else:
        return 'WEAK'

def categorize_pot(pot):
    """Convert pot size to category"""
    if pot <= 12:
        return 'SMALL'
    elif pot <= 50:
        return 'MEDIUM'
    else:
        return 'LARGE'

def categorize_bet(bet, pot):
    """Convert bet size to category"""
    if pot == 0:
        return 'SMALL'
    
    ratio = bet / pot
    if ratio > 1.0:
        return 'HUGE'
    elif ratio > 0.75:
        return 'LARGE'
    elif ratio > 0.50:
        return 'MEDIUM'
    else:
        return 'SMALL'

def categorize_street(street):
    """Convert street number to category"""
    if street >= 5:
        return 'RIVER'
    elif street >= 4:
        return 'TURN'
    else:
        return 'FLOP'

# ==================== OPPONENT-ADJUSTED TABLES ====================

def get_adjusted_bet_size(hand_strength, pot_category, street_name, opp_type):
    """Get bet size adjusted for opponent type"""
    base = get_bet_size(hand_strength, pot_category, street_name)
    
    if opp_type in ('CALLING_STATION', 'LOOSE_PASSIVE'):
        # Value bet bigger, don't bluff
        if hand_strength in ('VERY_STRONG', 'STRONG'):
            return base * 1.12
        else:
            return 0  # Don't bet without value
    
    elif opp_type in ('VERY_TIGHT', 'TIGHT'):
        # Bet smaller for value (they fold), bluff more
        if hand_strength in ('VERY_STRONG', 'STRONG'):
            return base * 0.88
        else:
            return base * 1.05  # Bluff a bit more
    
    elif opp_type == 'MANIAC':
        # Let them bet into us, smaller value bets
        return base * 0.90
    
    return base

def get_adjusted_fold_freq(hand_strength, bet_category, street_name, opp_type):
    """Get fold frequency adjusted for opponent type"""
    base = get_fold_frequency(hand_strength, bet_category, street_name)
    
    if opp_type in ('MANIAC', 'LOOSE_AGGRESSIVE'):
        # Call down wider vs maniacs
        return base * 0.75
    
    elif opp_type in ('VERY_TIGHT', 'TIGHT'):
        # Fold more vs tight players when they bet
        return min(0.95, base * 1.15)
    
    return base
