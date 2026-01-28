"""
IMPROVED GPU CFR FOR TOSS OR HOLD'EM
====================================
Full-featured training with:
1. All streets (preflop, flop, turn, river)
2. Toss decisions (which card to discard)
3. Position awareness (IP vs OOP)
4. Facing raise vs not
5. Pot size buckets (small/medium/large)
6. Better hand buckets (10 categories)
7. All actions (fold/check/call/raise + toss)

Game flow:
- Street 0: Preflop betting
- Street 2: P1 (OOP) tosses 1 card to board
- Street 3: P0 (IP) tosses 1 card to board
- Street 4: Flop betting (board +1 card)
- Street 5: Turn betting (board +1 card)
- Street 6: River betting (board +1 card)
- Showdown: Best 5 of 7 cards wins
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import random
import time
from collections import defaultdict
from itertools import combinations

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class ImprovedGPUCFR:
    def __init__(self):
        # Card setup
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.suits = ['h', 'd', 'c', 's']
        self.full_deck = [r + s for r in self.ranks for s in self.suits]
        
        self.rank_values = {r: i for i, r in enumerate(self.ranks)}  # 0-12
        self.suit_values = {s: i for i, s in enumerate(self.suits)}  # 0-3
        
        # Strategy storage
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        
        # Hand bucket definitions (10 buckets)
        self.PREFLOP_BUCKETS = [
            'PREMIUM',      # AA, KK, QQ, AKs
            'STRONG_PAIR',  # JJ, TT, 99
            'MED_PAIR',     # 88, 77, 66
            'LOW_PAIR',     # 55, 44, 33, 22
            'STRONG_HIGH',  # AK, AQ, AJs, KQs
            'MED_HIGH',     # AJ, AT, KQ, KJ, QJ
            'SUITED_CONN',  # Suited connectors
            'SUITED',       # Other suited
            'CONNECTED',    # Connected cards
            'TRASH'         # Everything else
        ]
        
        self.POSTFLOP_BUCKETS = [
            'MONSTER',      # Quads, full house, flush, straight
            'TRIPS',        # Three of a kind
            'TWO_PAIR',     # Two pair
            'OVERPAIR',     # Pair above board
            'TOP_PAIR',     # Top pair
            'MID_PAIR',     # Middle pair
            'LOW_PAIR',     # Bottom pair or pocket under board
            'FLUSH_DRAW',   # 4 to flush
            'STRAIGHT_DRAW', # Open-ended or gutshot
            'NOTHING'       # High card only
        ]
    
    def card_to_idx(self, card):
        """Convert card string to index 0-51"""
        rank = self.rank_values[card[0]]
        suit = self.suit_values[card[1]]
        return rank + suit * 13
    
    def evaluate_hand(self, cards):
        """Evaluate best 5-card hand from given cards"""
        if len(cards) < 5:
            return (0, [0])
        
        best = (0, [0])
        for five in combinations(cards, 5):
            score = self._score_5card(list(five))
            if score > best:
                best = score
        return best
    
    def _score_5card(self, cards):
        """Score a 5-card hand. Returns (hand_type, kickers)"""
        ranks = sorted([self.rank_values[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        
        # Count ranks
        rank_counts = defaultdict(int)
        for r in ranks:
            rank_counts[r] += 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        
        # Check flush
        is_flush = len(set(suits)) == 1
        
        # Check straight
        unique_ranks = sorted(set(ranks))
        is_straight = False
        straight_high = 0
        
        if len(unique_ranks) >= 5:
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i+4] - unique_ranks[i] == 4:
                    is_straight = True
                    straight_high = unique_ranks[i+4]
        
        # Check A-2-3-4-5 straight
        if set([0, 1, 2, 3, 12]).issubset(set(ranks)):
            is_straight = True
            straight_high = 3  # 5-high straight
        
        # Determine hand type
        if is_straight and is_flush:
            return (8, [straight_high])  # Straight flush
        if counts == [4, 1]:
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            return (7, [quad_rank])  # Four of a kind
        if counts == [3, 2]:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return (6, [trip_rank, pair_rank])  # Full house
        if is_flush:
            return (5, ranks)  # Flush
        if is_straight:
            return (4, [straight_high])  # Straight
        if counts == [3, 1, 1]:
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            return (3, [trip_rank] + sorted([r for r in ranks if rank_counts[r] == 1], reverse=True))  # Trips
        if counts == [2, 2, 1]:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return (2, pairs + [kicker])  # Two pair
        if counts == [2, 1, 1, 1]:
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return (1, [pair_rank] + kickers)  # One pair
        
        return (0, ranks)  # High card
    
    def get_preflop_bucket(self, cards):
        """Classify 3-card preflop hand into bucket"""
        ranks = sorted([self.rank_values[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        
        # Check for pairs
        rank_counts = defaultdict(int)
        for r in ranks:
            rank_counts[r] += 1
        
        has_pair = max(rank_counts.values()) >= 2
        pair_rank = max([r for r, c in rank_counts.items() if c >= 2], default=-1)
        
        # Check suited (at least 2)
        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
        max_suited = max(suit_counts.values())
        is_suited = max_suited >= 2
        
        # Check connected
        sorted_ranks = sorted(set(ranks))
        is_connected = False
        if len(sorted_ranks) >= 2:
            gaps = [sorted_ranks[i+1] - sorted_ranks[i] for i in range(len(sorted_ranks)-1)]
            is_connected = min(gaps) <= 2
        
        high = ranks[0]
        second = ranks[1] if len(ranks) > 1 else 0
        
        # Classify
        if has_pair:
            if pair_rank >= 10:  # QQ+
                return 'PREMIUM'
            elif pair_rank >= 8:  # 99-JJ
                return 'STRONG_PAIR'
            elif pair_rank >= 5:  # 66-88
                return 'MED_PAIR'
            else:
                return 'LOW_PAIR'
        
        # High cards
        if high == 12:  # Ace
            if second >= 11 and is_suited:  # AKs, AQs
                return 'PREMIUM'
            elif second >= 10:  # AK, AQ, AJ
                return 'STRONG_HIGH'
            elif second >= 8:  # AT, A9
                return 'MED_HIGH'
            elif is_suited:
                return 'SUITED'
            else:
                return 'TRASH'
        
        if high >= 10:  # Broadway
            if second >= 9:
                if is_suited:
                    return 'STRONG_HIGH'
                return 'MED_HIGH'
        
        if is_suited and is_connected:
            return 'SUITED_CONN'
        
        if is_suited:
            return 'SUITED'
        
        if is_connected and high >= 7:
            return 'CONNECTED'
        
        return 'TRASH'
    
    def get_postflop_bucket(self, hole_cards, board_cards):
        """Classify postflop hand into bucket"""
        all_cards = list(hole_cards) + list(board_cards)
        hand_eval = self.evaluate_hand(all_cards)
        hand_type = hand_eval[0]
        
        if hand_type >= 4:  # Straight or better
            return 'MONSTER'
        if hand_type == 3:  # Trips
            return 'TRIPS'
        if hand_type == 2:  # Two pair
            return 'TWO_PAIR'
        
        if hand_type == 1:  # One pair
            pair_rank = hand_eval[1][0]
            board_ranks = [self.rank_values[c[0]] for c in board_cards]
            hole_ranks = [self.rank_values[c[0]] for c in hole_cards]
            
            if pair_rank > max(board_ranks, default=-1):
                # Overpair (pocket pair above board)
                if pair_rank in hole_ranks and hole_ranks.count(pair_rank) >= 2:
                    return 'OVERPAIR'
                # Top pair
                if pair_rank == max(board_ranks, default=-1):
                    return 'TOP_PAIR'
            
            board_sorted = sorted(board_ranks, reverse=True)
            if len(board_sorted) >= 1 and pair_rank == board_sorted[0]:
                return 'TOP_PAIR'
            elif len(board_sorted) >= 2 and pair_rank >= board_sorted[1]:
                return 'MID_PAIR'
            else:
                return 'LOW_PAIR'
        
        # Check for draws
        suits = [c[1] for c in all_cards]
        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
        
        if max(suit_counts.values()) >= 4:
            return 'FLUSH_DRAW'
        
        # Check straight draw
        all_ranks = sorted(set([self.rank_values[c[0]] for c in all_cards]))
        for i in range(len(all_ranks) - 3):
            window = all_ranks[i:i+5] if i+5 <= len(all_ranks) else all_ranks[i:]
            if len(window) >= 4 and window[-1] - window[0] <= 4:
                return 'STRAIGHT_DRAW'
        
        return 'NOTHING'
    
    def get_toss_bucket(self, cards):
        """Get bucket for toss decision based on 3-card hand"""
        return self.get_preflop_bucket(cards)
    
    def get_info_set(self, state, player):
        """
        Build comprehensive info set string:
        Format: {street}|{hand_bucket}|{position}|{facing_raise}|{pot_size}
        """
        street = state['street']
        my_hand = state['p0_hand'] if player == 0 else state['p1_hand']
        board = state['board']
        history = state['history']
        pot = state['pot']
        
        # Street name
        street_names = {0: 'preflop', 2: 'toss_p1', 3: 'toss_p0', 4: 'flop', 5: 'turn', 6: 'river'}
        street_name = street_names.get(street, 'unknown')
        
        # Hand bucket
        if street in [0, 2, 3]:
            hand_bucket = self.get_preflop_bucket(my_hand)
        else:
            hand_bucket = self.get_postflop_bucket(my_hand, board)
        
        # Position (P0 is button/IP, P1 is OOP)
        position = 'IP' if player == 0 else 'OOP'
        
        # Facing raise
        facing_raise = 'R' if 'r' in history else 'NR'
        num_raises = history.count('r')
        if num_raises >= 2:
            facing_raise = 'RR'  # Re-raised
        
        # Pot size
        if pot <= 10:
            pot_bucket = 'S'
        elif pot <= 30:
            pot_bucket = 'M'
        else:
            pot_bucket = 'L'
        
        return f"{street_name}|{hand_bucket}|{position}|{facing_raise}|{pot_bucket}"
    
    def get_actions(self, state, player):
        """Get available actions for current state"""
        street = state['street']
        history = state['history']
        
        # Toss phase
        if street in [2, 3]:
            active_player = 1 if street == 2 else 0
            if player == active_player:
                hand = state['p0_hand'] if player == 0 else state['p1_hand']
                return [f't{i}' for i in range(len(hand))]
            else:
                return ['wait']  # Not our turn
        
        # Betting phase
        if len(history) == 0:
            return ['check', 'raise']
        elif history[-1] == 'c':
            return ['check', 'raise']
        elif history[-1] == 'r':
            if history.count('r') >= 3:
                return ['fold', 'call']  # Raise cap
            return ['fold', 'call', 'raise']
        
        return ['check', 'raise']
    
    def get_strategy(self, info_set, actions):
        """Get current strategy using regret matching"""
        regrets = self.regret_sum[info_set]
        
        strategy = {}
        total = 0
        for a in actions:
            strategy[a] = max(regrets[a], 0)
            total += strategy[a]
        
        if total > 0:
            for a in actions:
                strategy[a] /= total
        else:
            for a in actions:
                strategy[a] = 1.0 / len(actions)
        
        return strategy
    
    def sample_action(self, strategy):
        """Sample action from strategy distribution"""
        r = random.random()
        cumsum = 0
        for action, prob in strategy.items():
            cumsum += prob
            if r < cumsum:
                return action
        return list(strategy.keys())[-1]
    
    def showdown(self, state):
        """Determine winner at showdown. Returns 1 if P0 wins, -1 if P1 wins, 0 for tie."""
        p0_eval = self.evaluate_hand(state['p0_hand'] + state['board'])
        p1_eval = self.evaluate_hand(state['p1_hand'] + state['board'])
        
        if p0_eval > p1_eval:
            return 1
        elif p0_eval < p1_eval:
            return -1
        return 0
    
    def proceed_street(self, state):
        """Advance game to next street"""
        street = state['street']
        deck = state['deck']
        
        if street == 0:
            # Preflop -> Toss phase, reveal 2 board cards
            new_board = deck[6:8]
            return {
                'p0_hand': state['p0_hand'][:],
                'p1_hand': state['p1_hand'][:],
                'board': new_board,
                'deck': deck,
                'street': 2,
                'history': '',
                'pot': state['pot']
            }
        elif street == 2:
            # P1 toss done -> P0 toss
            return {
                'p0_hand': state['p0_hand'][:],
                'p1_hand': state['p1_hand'][:],
                'board': state['board'][:],
                'deck': deck,
                'street': 3,
                'history': '',
                'pot': state['pot']
            }
        elif street == 3:
            # P0 toss done -> Flop
            new_board = state['board'][:] + [deck[10]]
            return {
                'p0_hand': state['p0_hand'][:],
                'p1_hand': state['p1_hand'][:],
                'board': new_board,
                'deck': deck,
                'street': 4,
                'history': '',
                'pot': state['pot']
            }
        elif street == 4:
            # Flop -> Turn
            new_board = state['board'][:] + [deck[11]]
            return {
                'p0_hand': state['p0_hand'][:],
                'p1_hand': state['p1_hand'][:],
                'board': new_board,
                'deck': deck,
                'street': 5,
                'history': '',
                'pot': state['pot']
            }
        elif street == 5:
            # Turn -> River
            new_board = state['board'][:] + [deck[12]]
            return {
                'p0_hand': state['p0_hand'][:],
                'p1_hand': state['p1_hand'][:],
                'board': new_board,
                'deck': deck,
                'street': 6,
                'history': '',
                'pot': state['pot']
            }
        else:
            # River -> Showdown
            return None
    
    def cfr_iteration(self, state, player, traverser):
        """
        External sampling MCCFR iteration
        - Traverse all actions for traverser
        - Sample actions for opponent
        """
        street = state['street']
        history = state['history']
        pot = state['pot']
        
        # Terminal: fold
        if 'f' in history:
            folder = len(history) % 2
            if folder == traverser:
                return -pot / 2
            return pot / 2
        
        # Terminal: showdown after river betting
        if street == 6 and len(history) >= 2 and history[-1] == 'c':
            result = self.showdown(state)
            if traverser == 0:
                return (pot / 2) * result
            return (pot / 2) * -result
        
        # Toss phase
        if street in [2, 3]:
            active_player = 1 if street == 2 else 0
            
            if active_player != player:
                # Not this player's turn, proceed
                next_state = self.proceed_street(state)
                return self.cfr_iteration(next_state, 0, traverser)
            
            actions = self.get_actions(state, player)
            if actions == ['wait']:
                next_state = self.proceed_street(state)
                return self.cfr_iteration(next_state, 0, traverser)
            
            info_set = self.get_info_set(state, player)
            strategy = self.get_strategy(info_set, actions)
            
            if player != traverser:
                # Sample opponent action
                action = self.sample_action(strategy)
                toss_idx = int(action[1])
                
                next_state = {
                    'p0_hand': state['p0_hand'][:],
                    'p1_hand': state['p1_hand'][:],
                    'board': state['board'][:],
                    'deck': state['deck'],
                    'street': street,
                    'history': '',
                    'pot': pot
                }
                
                if active_player == 0:
                    tossed = next_state['p0_hand'].pop(toss_idx)
                else:
                    tossed = next_state['p1_hand'].pop(toss_idx)
                next_state['board'].append(tossed)
                
                next_state = self.proceed_street(next_state)
                return self.cfr_iteration(next_state, 0, traverser)
            
            else:
                # Traverse all toss actions
                action_values = {}
                for action in actions:
                    toss_idx = int(action[1])
                    
                    next_state = {
                        'p0_hand': state['p0_hand'][:],
                        'p1_hand': state['p1_hand'][:],
                        'board': state['board'][:],
                        'deck': state['deck'],
                        'street': street,
                        'history': '',
                        'pot': pot
                    }
                    
                    if active_player == 0:
                        tossed = next_state['p0_hand'].pop(toss_idx)
                    else:
                        tossed = next_state['p1_hand'].pop(toss_idx)
                    next_state['board'].append(tossed)
                    
                    next_state = self.proceed_street(next_state)
                    action_values[action] = self.cfr_iteration(next_state, 0, traverser)
                
                # Update regrets
                node_value = sum(strategy[a] * action_values[a] for a in actions)
                for action in actions:
                    regret = action_values[action] - node_value
                    self.regret_sum[info_set][action] += regret
                
                # Update strategy sum
                for a in actions:
                    self.strategy_sum[info_set][a] += strategy[a]
                
                return node_value
        
        # Betting phase
        acting_player = len(history) % 2
        actions = self.get_actions(state, acting_player)
        info_set = self.get_info_set(state, acting_player)
        strategy = self.get_strategy(info_set, actions)
        
        if acting_player != traverser:
            # Sample opponent action
            action = self.sample_action(strategy)
            
            if action == 'fold':
                if traverser == acting_player:
                    return -pot / 2
                return pot / 2
            
            new_history = history + ('c' if action in ['check', 'call'] else 'r')
            new_pot = pot + (2 if action == 'raise' else 0)
            
            next_state = {
                'p0_hand': state['p0_hand'][:],
                'p1_hand': state['p1_hand'][:],
                'board': state['board'][:],
                'deck': state['deck'],
                'street': street,
                'history': new_history,
                'pot': new_pot
            }
            
            # Check if betting round ends
            if len(new_history) >= 2 and new_history[-1] == 'c':
                next_state = self.proceed_street(next_state)
                if next_state is None:
                    result = self.showdown(state)
                    if traverser == 0:
                        return (new_pot / 2) * result
                    return (new_pot / 2) * -result
            
            return self.cfr_iteration(next_state, (acting_player + 1) % 2, traverser)
        
        else:
            # Traverse all actions
            action_values = {}
            
            for action in actions:
                if action == 'fold':
                    action_values[action] = -pot / 2
                    continue
                
                new_history = history + ('c' if action in ['check', 'call'] else 'r')
                new_pot = pot + (2 if action == 'raise' else 0)
                
                next_state = {
                    'p0_hand': state['p0_hand'][:],
                    'p1_hand': state['p1_hand'][:],
                    'board': state['board'][:],
                    'deck': state['deck'],
                    'street': street,
                    'history': new_history,
                    'pot': new_pot
                }
                
                if len(new_history) >= 2 and new_history[-1] == 'c':
                    next_state = self.proceed_street(next_state)
                    if next_state is None:
                        result = self.showdown(state)
                        if traverser == 0:
                            action_values[action] = (new_pot / 2) * result
                        else:
                            action_values[action] = (new_pot / 2) * -result
                        continue
                
                action_values[action] = self.cfr_iteration(next_state, (acting_player + 1) % 2, traverser)
            
            # Update regrets
            node_value = sum(strategy[a] * action_values[a] for a in actions)
            for action in actions:
                regret = action_values[action] - node_value
                self.regret_sum[info_set][action] += regret
            
            # Update strategy sum
            for a in actions:
                self.strategy_sum[info_set][a] += strategy[a]
            
            return node_value
    
    def run_iteration(self):
        """Run one full CFR iteration"""
        # Deal cards
        deck = self.full_deck[:]
        random.shuffle(deck)
        
        p0_hand = deck[0:3]
        p1_hand = deck[3:6]
        
        state = {
            'p0_hand': p0_hand,
            'p1_hand': p1_hand,
            'board': [],
            'deck': deck,
            'street': 0,
            'history': '',
            'pot': 3  # SB + BB
        }
        
        # Traverse for P0
        self.cfr_iteration(state, 0, 0)
        
        # Traverse for P1
        state = {
            'p0_hand': p0_hand[:],
            'p1_hand': p1_hand[:],
            'board': [],
            'deck': deck,
            'street': 0,
            'history': '',
            'pot': 3
        }
        self.cfr_iteration(state, 0, 1)
    
    def train(self, num_iterations, checkpoint_every=1000000):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("IMPROVED GPU CFR TRAINING")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations:,}")
        print(f"Checkpoint every: {checkpoint_every:,}")
        print(f"Features: streets, toss, position, raise-facing, pot size")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for i in range(1, num_iterations + 1):
            self.run_iteration()
            
            # Progress update
            if i % 10000 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (num_iterations - i) / rate / 60
                print(f"Iteration {i:,} | {rate:.0f} iter/s | "
                      f"Info sets: {len(self.regret_sum)} | "
                      f"ETA: {eta:.1f} min", flush=True)
            
            # Checkpoint
            if i % checkpoint_every == 0:
                self.save(f'improved_cfr_{i//1000000}M.pkl')
                print(f"✅ Checkpoint saved: improved_cfr_{i//1000000}M.pkl", flush=True)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Total info sets: {len(self.regret_sum)}")
        print(f"{'='*60}")
    
    def get_average_strategy(self):
        """Compute average strategy from strategy sums"""
        avg_strategy = {}
        for info_set, sums in self.strategy_sum.items():
            avg_strategy[info_set] = {}
            total = sum(sums.values())
            if total > 0:
                for a, v in sums.items():
                    avg_strategy[info_set][a] = v / total
            else:
                # Uniform if no data
                for a in sums.keys():
                    avg_strategy[info_set][a] = 1.0 / len(sums)
        return avg_strategy
    
    def save(self, filename):
        """Save strategy to file"""
        avg_strategy = self.get_average_strategy()
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'strategy': avg_strategy,
                'regret_sum': dict(self.regret_sum),
                'strategy_sum': dict(self.strategy_sum)
            }, f)
        print(f"💾 Saved to {filename}")
    
    def load(self, filename):
        """Load strategy from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.strategy_sum = defaultdict(lambda: defaultdict(float), data.get('strategy_sum', {}))
        self.regret_sum = defaultdict(lambda: defaultdict(float), data.get('regret_sum', {}))
        print(f"📂 Loaded from {filename}")


def main():
    print("🚀 Starting Improved GPU CFR Training")
    
    trainer = ImprovedGPUCFR()
    
    # Train 50M iterations (adjust based on time)
    # At ~100-500 iter/s, this takes 1-6 hours
    trainer.train(
        num_iterations=50_000_000,
        checkpoint_every=10_000_000
    )
    
    trainer.save('improved_cfr_final.pkl')
    print("\n✅ ✅ ✅ DONE! ✅ ✅ ✅")


if __name__ == '__main__':
    main()
