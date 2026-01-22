'''
HYBRID POKER BOT - TOP 5-8 EDITION
===================================
Combines:
1. GPU CFR (50M) for betting decisions
2. Smart toss heuristics (equity-based)
3. Opponent modeling & exploitation
4. Pot odds calculations
5. Position-aware adjustments
6. Time management for 60-second limit

Designed for Toss or Hold'em variant.
'''
import pickle
import random
import time
from collections import defaultdict
from itertools import combinations
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot


class Player(Bot):
    '''
    Hybrid poker bot for top 5-8 finish
    '''

    def __init__(self):
        # Load GPU CFR strategy
        self.cfr_strategy = {}
        try:
            with open('gpu_cfr_50M.pkl', 'rb') as f:
                data = pickle.load(f)
                self.cfr_strategy = data.get('strategy', {})
            print(f"✅ Loaded GPU CFR with {len(self.cfr_strategy)} info sets")
        except FileNotFoundError:
            print("⚠️ gpu_cfr_50M.pkl not found, using pure heuristics")

        # Try loading improved CFR if available
        self.improved_cfr = {}
        try:
            with open('improved_cfr_final.pkl', 'rb') as f:
                data = pickle.load(f)
                self.improved_cfr = data.get('strategy', {})
            print(f"✅ Loaded Improved CFR with {len(self.improved_cfr)} info sets")
        except FileNotFoundError:
            pass  # Not available yet

        # Card values
        self.rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        self.rank_names = {v: k for k, v in self.rank_values.items()}

        # Opponent modeling
        self.opp_stats = {
            'hands': 0,
            'vpip_hands': 0,      # Voluntarily put money in
            'pfr_hands': 0,       # Preflop raises
            'cbet_opportunities': 0,
            'cbets': 0,
            'folds_to_raise': 0,
            'raises_faced': 0,
            'showdowns': 0,
            'showdown_wins': 0,
            'total_bets': 0,
            'total_calls': 0,
            'total_folds': 0,
            'aggression_actions': 0,
            'passive_actions': 0
        }

        # Time management
        self.game_start_time = None
        self.total_time = 60.0
        self.decisions_made = 0
        self.round_num = 0

        # Track current round actions
        self.current_round_opp_actions = []

    # ==================== TIME MANAGEMENT ====================

    def get_remaining_time(self):
        """Get remaining time in seconds"""
        if self.game_start_time is None:
            return self.total_time
        return max(0.1, self.total_time - (time.time() - self.game_start_time))

    def get_time_per_decision(self):
        """Calculate time budget per decision"""
        remaining = self.get_remaining_time()
        # Estimate ~5-8 decisions per round, ~500 rounds remaining
        estimated_decisions = max(100, (NUM_ROUNDS - self.round_num) * 6)
        return remaining / estimated_decisions

    # ==================== OPPONENT MODELING ====================

    def get_opp_vpip(self):
        """Opponent's voluntarily put $ in pot %"""
        if self.opp_stats['hands'] < 5:
            return 0.5  # Assume average
        return self.opp_stats['vpip_hands'] / self.opp_stats['hands']

    def get_opp_pfr(self):
        """Opponent's preflop raise %"""
        if self.opp_stats['hands'] < 5:
            return 0.3
        return self.opp_stats['pfr_hands'] / self.opp_stats['hands']

    def get_opp_aggression(self):
        """Opponent's aggression factor"""
        passive = self.opp_stats['passive_actions']
        aggressive = self.opp_stats['aggression_actions']
        if passive == 0:
            return 2.0 if aggressive > 0 else 1.0
        return aggressive / passive

    def get_opp_fold_to_raise(self):
        """How often opponent folds to raise"""
        if self.opp_stats['raises_faced'] < 3:
            return 0.4  # Assume average
        return self.opp_stats['folds_to_raise'] / self.opp_stats['raises_faced']

    def is_opponent_tight(self):
        """Is opponent playing tight?"""
        return self.get_opp_vpip() < 0.35

    def is_opponent_loose(self):
        """Is opponent playing loose?"""
        return self.get_opp_vpip() > 0.55

    def is_opponent_passive(self):
        """Is opponent passive?"""
        return self.get_opp_aggression() < 1.0

    def is_opponent_aggressive(self):
        """Is opponent aggressive?"""
        return self.get_opp_aggression() > 2.0

    def update_opponent_stats(self, action, street, facing_raise):
        """Update opponent statistics based on their action"""
        if isinstance(action, str):
            action_type = action
        else:
            action_type = type(action).__name__

        if street == 0:  # Preflop
            if action_type in ['RaiseAction', 'raise']:
                self.opp_stats['vpip_hands'] += 0.5  # Partial credit
                self.opp_stats['pfr_hands'] += 0.5
                self.opp_stats['aggression_actions'] += 1
            elif action_type in ['CallAction', 'call']:
                self.opp_stats['vpip_hands'] += 0.5
                self.opp_stats['passive_actions'] += 1

        if action_type in ['RaiseAction', 'raise']:
            self.opp_stats['aggression_actions'] += 1
            self.opp_stats['total_bets'] += 1
        elif action_type in ['CallAction', 'call', 'CheckAction', 'check']:
            self.opp_stats['passive_actions'] += 1
            self.opp_stats['total_calls'] += 1
        elif action_type in ['FoldAction', 'fold']:
            self.opp_stats['total_folds'] += 1
            if facing_raise:
                self.opp_stats['folds_to_raise'] += 1

        if facing_raise:
            self.opp_stats['raises_faced'] += 1

    # ==================== HAND EVALUATION ====================

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
        """Score a 5-card hand"""
        ranks = sorted([self.rank_values[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]

        rank_counts = defaultdict(int)
        for r in ranks:
            rank_counts[r] += 1
        counts = sorted(rank_counts.values(), reverse=True)

        is_flush = len(set(suits)) == 1

        # Check straight
        unique_ranks = sorted(set(ranks))
        is_straight = False
        if len(unique_ranks) >= 5:
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i+4] - unique_ranks[i] == 4:
                    is_straight = True
        # A-2-3-4-5
        if set([14, 2, 3, 4, 5]).issubset(set(ranks)):
            is_straight = True

        if is_straight and is_flush:
            return (8, ranks)
        if counts == [4, 1]:
            return (7, ranks)
        if counts == [3, 2]:
            return (6, ranks)
        if is_flush:
            return (5, ranks)
        if is_straight:
            return (4, ranks)
        if counts == [3, 1, 1]:
            return (3, ranks)
        if counts == [2, 2, 1]:
            return (2, ranks)
        if counts == [2, 1, 1, 1]:
            return (1, ranks)
        return (0, ranks)

    # ==================== HAND ABSTRACTION ====================

    def get_hand_bucket(self, hole_cards, board_cards=None):
        """Get hand bucket for CFR lookup"""
        if not board_cards or len(board_cards) < 3:
            return self._get_preflop_bucket(hole_cards)
        return self._get_postflop_bucket(hole_cards, board_cards)

    def _get_preflop_bucket(self, cards):
        """Classify preflop hand (3 cards)"""
        ranks = sorted([self.rank_values[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]

        # Check pairs
        rank_counts = defaultdict(int)
        for r in ranks:
            rank_counts[r] += 1
        has_pair = max(rank_counts.values()) >= 2

        # Check suited (at least 2 cards)
        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
        max_suited = max(suit_counts.values())
        is_suited = max_suited >= 2
        is_tri_suited = max_suited >= 3

        # Check trips
        if max(rank_counts.values()) >= 3:
            return 'MONSTER'

        # Pairs
        if has_pair:
            pair_rank = max([r for r, c in rank_counts.items() if c >= 2])
            if pair_rank >= 10:  # TT+
                return 'STRONG'
            if pair_rank >= 6:  # 66-99
                return 'MEDIUM'
            return 'WEAK'  # 22-55

        # High card hands
        high = ranks[0]
        second = ranks[1]
        third = ranks[2] if len(ranks) > 2 else 0

        # Three suited with an ace - strong flush potential
        if is_tri_suited and high == 14:
            return 'STRONG'

        # Premium: A + K or A + Q + J type hands
        if high == 14:  # Ace high
            if second >= 11:  # AK, AQ, AJ
                return 'STRONG'
            if is_suited:  # Any suited ace
                return 'MEDIUM'
            if second >= 8:
                return 'MEDIUM'
            return 'WEAK'

        # Broadway heavy (KQJ, KQT, etc)
        if high >= 11 and second >= 10:
            if is_suited:
                return 'STRONG'
            return 'MEDIUM'

        # Suited connectors
        if is_suited:
            gap = ranks[0] - ranks[1]
            if gap <= 2 and high >= 7:
                return 'MEDIUM'
            if high >= 10:
                return 'MEDIUM'
            return 'WEAK'

        # Connected cards
        if ranks[0] - ranks[1] <= 2 and ranks[1] - ranks[2] <= 2:
            if high >= 8:
                return 'WEAK'

        # True trash - disconnected low cards
        if high <= 8 and (ranks[0] - ranks[2]) >= 4:
            return 'TRASH'

        return 'WEAK' if high >= 8 else 'TRASH'

    def _get_postflop_bucket(self, hole_cards, board_cards):
        """Classify postflop hand"""
        all_cards = list(hole_cards) + list(board_cards)
        hand_eval = self.evaluate_hand(all_cards)
        hand_type = hand_eval[0]

        # MONSTER: Straight or better
        if hand_type >= 4:
            return 'MONSTER'

        # STRONG: Trips or better
        if hand_type == 3:
            return 'STRONG'

        # Two pair
        if hand_type == 2:
            return 'STRONG'

        # One pair - need to check quality
        if hand_type == 1:
            pair_rank = hand_eval[1][0]  # The rank of the pair
            board_ranks = sorted([self.rank_values[c[0]] for c in board_cards], reverse=True)
            hole_ranks = [self.rank_values[c[0]] for c in hole_cards]

            # Check if it's an overpair (pocket pair above board)
            if hole_ranks[0] == hole_ranks[1] if len(hole_ranks) >= 2 else False:
                if min(hole_ranks) > max(board_ranks):
                    return 'STRONG'

            # Top pair with top kicker
            if pair_rank == board_ranks[0]:  # Top pair
                if max(hole_ranks) >= 12:  # A or K kicker
                    return 'STRONG'
                return 'MEDIUM'

            # Middle or bottom pair
            return 'WEAK'

        # No made hand - check draws
        suits = [c[1] for c in all_cards]
        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
        if max(suit_counts.values()) >= 4:
            return 'MEDIUM'  # Flush draw

        # Check straight draw
        all_ranks = sorted(set([self.rank_values[c[0]] for c in all_cards]))
        for i in range(len(all_ranks) - 3):
            window = all_ranks[i:i+4]
            if len(window) >= 4 and window[-1] - window[0] <= 4:
                return 'WEAK'  # Straight draw

        hole_ranks = [self.rank_values[c[0]] for c in hole_cards]
        if max(hole_ranks) >= 12:
            return 'WEAK'  # At least has overcards

        return 'TRASH'

    # ==================== TOSS DECISIONS ====================

    def get_toss_decision(self, cards, board):
        """
        Decide which card to toss (0, 1, or 2)
        Uses equity estimation for each possible toss
        """
        if len(cards) != 3:
            return 0

        # Quick heuristics for time pressure
        if self.get_time_per_decision() < 0.01:
            return self._quick_toss(cards)

        best_toss = 0
        best_score = -1

        for toss_idx in range(3):
            remaining = [cards[i] for i in range(3) if i != toss_idx]
            tossed = cards[toss_idx]

            score = self._evaluate_toss(remaining, tossed, board)
            if score > best_score:
                best_score = score
                best_toss = toss_idx

        return best_toss

    def _quick_toss(self, cards):
        """Fast toss decision under time pressure"""
        ranks = [self.rank_values[c[0]] for c in cards]
        suits = [c[1] for c in cards]

        # Keep pairs
        rank_counts = defaultdict(list)
        for i, r in enumerate(ranks):
            rank_counts[r].append(i)

        for r, indices in rank_counts.items():
            if len(indices) >= 2:
                # Have a pair, toss the non-pair card
                for i in range(3):
                    if i not in indices:
                        return i

        # Keep suited cards
        suit_counts = defaultdict(list)
        for i, s in enumerate(suits):
            suit_counts[s].append(i)

        for s, indices in suit_counts.items():
            if len(indices) >= 2:
                # Have suited, toss the off-suit
                for i in range(3):
                    if i not in indices:
                        return i

        # Toss lowest card
        min_rank = min(ranks)
        return ranks.index(min_rank)

    def _evaluate_toss(self, remaining, tossed, board):
        """
        Evaluate the quality of keeping 'remaining' and tossing 'tossed'
        Higher score = better toss decision
        """
        r1, r2 = [self.rank_values[c[0]] for c in remaining]
        s1, s2 = [c[1] for c in remaining]
        tossed_rank = self.rank_values[tossed[0]]

        score = 0

        # Base score: sum of kept ranks
        score += (r1 + r2) * 2

        # Bonus for pair
        if r1 == r2:
            score += 50 + r1 * 3

        # Bonus for suited
        if s1 == s2:
            score += 20

        # Bonus for connected
        gap = abs(r1 - r2)
        if gap == 1:
            score += 15
        elif gap == 2:
            score += 8

        # Bonus for high cards
        if r1 >= 12 or r2 >= 12:  # A or K
            score += 15
        if r1 >= 10 or r2 >= 10:  # T+
            score += 8

        # Penalty for tossing high card (unless keeping better)
        if tossed_rank >= 12 and r1 < 12 and r2 < 12:
            score -= 10

        # Consider board interaction if board exists
        if board:
            board_ranks = [self.rank_values[c[0]] for c in board]
            # Bonus for matching board
            if r1 in board_ranks:
                score += 25
            if r2 in board_ranks:
                score += 25

        return score

    # ==================== BETTING DECISIONS ====================

    def get_cfr_action(self, round_state, active):
        """Get action from CFR strategy"""
        hole_cards = round_state.hands[active]
        board = round_state.board
        street = round_state.street

        # Build info set
        if street >= 4:
            bucket = self._get_postflop_bucket(hole_cards, board)
        else:
            bucket = self._get_preflop_bucket(hole_cards)

        # Check improved CFR first
        info_set_improved = self._build_improved_info_set(round_state, active, bucket)
        if info_set_improved in self.improved_cfr:
            return self._sample_cfr_action(self.improved_cfr[info_set_improved])

        # Fall back to simple CFR
        info_set_simple = f"postflop|{bucket}"
        if info_set_simple in self.cfr_strategy:
            return self._sample_cfr_action(self.cfr_strategy[info_set_simple])

        return None  # No CFR guidance

    def _build_improved_info_set(self, round_state, active, bucket):
        """Build info set for improved CFR"""
        street = round_state.street

        street_names = {0: 'preflop', 2: 'toss_p1', 3: 'toss_p0',
                       4: 'flop', 5: 'turn', 6: 'river'}
        street_name = street_names.get(street, 'unknown')

        # Position
        position = 'IP' if active == 0 else 'OOP'

        # Facing raise (simplified)
        pot = round_state.pips[0] + round_state.pips[1]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        facing_raise = 'R' if opp_pip > my_pip else 'NR'

        # Pot size
        if pot <= 10:
            pot_bucket = 'S'
        elif pot <= 30:
            pot_bucket = 'M'
        else:
            pot_bucket = 'L'

        return f"{street_name}|{bucket}|{position}|{facing_raise}|{pot_bucket}"

    def _sample_cfr_action(self, strategy):
        """Sample action from CFR strategy distribution"""
        actions = list(strategy.keys())
        probs = [strategy[a] for a in actions]

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p/total for p in probs]
        else:
            probs = [1.0/len(probs)] * len(probs)

        return random.choices(actions, weights=probs)[0]

    def get_pot_odds(self, round_state, active):
        """Calculate pot odds for calling"""
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        pot = my_pip + opp_pip

        call_amount = opp_pip - my_pip
        if call_amount <= 0:
            return 1.0  # No cost to call

        return call_amount / (pot + call_amount)

    def get_hand_equity_estimate(self, hole_cards, board):
        """Estimate hand equity (0-1)"""
        if not board:
            bucket = self._get_preflop_bucket(hole_cards)
            equity_map = {'STRONG': 0.70, 'MEDIUM': 0.50, 'WEAK': 0.35, 'TRASH': 0.25}
            return equity_map.get(bucket, 0.3)

        bucket = self._get_postflop_bucket(hole_cards, board)
        equity_map = {'MONSTER': 0.90, 'STRONG': 0.75, 'MEDIUM': 0.55, 'WEAK': 0.35, 'TRASH': 0.20}
        return equity_map.get(bucket, 0.3)

    def should_bluff(self, round_state, active):
        """Determine if we should bluff"""
        # Bluff more against tight/passive opponents
        if self.is_opponent_tight() and self.get_opp_fold_to_raise() > 0.5:
            return random.random() < 0.25

        if self.is_opponent_passive():
            return random.random() < 0.15

        return random.random() < 0.08

    def get_bet_size(self, round_state, active, strength):
        """Determine bet sizing based on strength"""
        min_raise, max_raise = round_state.raise_bounds()
        pot = round_state.pips[0] + round_state.pips[1]

        if strength == 'MONSTER':
            # Value bet big
            target = min(max_raise, min_raise + pot)
        elif strength == 'STRONG':
            # Medium value bet
            target = min(max_raise, min_raise + pot // 2)
        elif strength == 'bluff':
            # Small bluff
            target = min_raise
        else:
            # Default min raise
            target = min_raise

        return max(min_raise, min(target, max_raise))

    # ==================== MAIN ACTION LOGIC ====================

    def handle_new_round(self, game_state, round_state, active):
        """Called at start of each round"""
        if self.game_start_time is None:
            self.game_start_time = time.time()

        self.round_num += 1
        self.opp_stats['hands'] += 1
        self.current_round_opp_actions = []

    def handle_round_over(self, game_state, terminal_state, active):
        """Called at end of each round"""
        pass

    def get_action(self, game_state, round_state, active):
        """Main decision function"""
        self.decisions_made += 1
        legal_actions = round_state.legal_actions()

        # ===== TOSS PHASE =====
        if round_state.street in [2, 3]:
            if DiscardAction in legal_actions:
                my_cards = round_state.hands[active]
                board = round_state.board
                toss_idx = self.get_toss_decision(my_cards, board)
                return DiscardAction(toss_idx)
            return CheckAction()

        # ===== BETTING PHASE =====
        hole_cards = round_state.hands[active]
        board = round_state.board
        street = round_state.street

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        pot = my_pip + opp_pip
        continue_cost = opp_pip - my_pip

        # Get hand strength
        if street >= 4 and board:
            bucket = self._get_postflop_bucket(hole_cards, board)
        else:
            bucket = self._get_preflop_bucket(hole_cards)

        equity = self.get_hand_equity_estimate(hole_cards, board)
        pot_odds = self.get_pot_odds(round_state, active)

        # Get CFR suggestion
        cfr_action = self.get_cfr_action(round_state, active)

        # ===== DECISION LOGIC =====

        # MONSTER hands - always bet/raise
        if bucket == 'MONSTER':
            if RaiseAction in legal_actions:
                amount = self.get_bet_size(round_state, active, 'MONSTER')
                return RaiseAction(amount)
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction()

        # STRONG hands - usually bet/raise, always call
        if bucket == 'STRONG':
            if cfr_action in ['bet', 'raise'] and RaiseAction in legal_actions:
                amount = self.get_bet_size(round_state, active, 'STRONG')
                return RaiseAction(amount)
            if continue_cost > 0:
                if CallAction in legal_actions:
                    return CallAction()
            if RaiseAction in legal_actions and random.random() < 0.6:
                amount = self.get_bet_size(round_state, active, 'STRONG')
                return RaiseAction(amount)
            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction()

        # MEDIUM hands - call if good odds, sometimes bet
        if bucket == 'MEDIUM':
            if continue_cost > 0:
                # Call if pot odds are good
                if equity > pot_odds * 1.2:
                    if CallAction in legal_actions:
                        return CallAction()
                # Fold if too expensive
                if continue_cost > pot * 0.5 and equity < 0.4:
                    if FoldAction in legal_actions:
                        return FoldAction()
                if CallAction in legal_actions:
                    return CallAction()
            else:
                # No cost - bet sometimes
                if cfr_action in ['bet', 'raise'] and RaiseAction in legal_actions:
                    if random.random() < 0.4:
                        return RaiseAction(round_state.raise_bounds()[0])
                return CheckAction()

        # WEAK hands - check/fold, occasional bluff
        if bucket == 'WEAK':
            if continue_cost > 0:
                # Bluff catch against aggressive opponent
                if self.is_opponent_aggressive() and continue_cost < pot * 0.3:
                    if random.random() < 0.2 and CallAction in legal_actions:
                        return CallAction()
                if FoldAction in legal_actions:
                    return FoldAction()
                return CallAction()  # Forced call
            else:
                # Bluff sometimes
                if self.should_bluff(round_state, active) and RaiseAction in legal_actions:
                    return RaiseAction(round_state.raise_bounds()[0])
                return CheckAction()

        # TRASH hands - fold or check
        if bucket == 'TRASH':
            if continue_cost > 0:
                # Only bluff if opponent is very tight
                if self.is_opponent_tight() and self.get_opp_fold_to_raise() > 0.6:
                    if random.random() < 0.1 and RaiseAction in legal_actions:
                        return RaiseAction(round_state.raise_bounds()[0])
                if FoldAction in legal_actions:
                    return FoldAction()
                return CallAction()
            else:
                # Check or rare bluff
                if self.should_bluff(round_state, active) and RaiseAction in legal_actions:
                    if random.random() < 0.15:
                        return RaiseAction(round_state.raise_bounds()[0])
                return CheckAction()

        # ===== FALLBACK =====
        if CheckAction in legal_actions:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
        if FoldAction in legal_actions:
            return FoldAction()

        return list(legal_actions)[0]


if __name__ == '__main__':
    from skeleton.runner import parse_args, run_bot
    run_bot(Player(), parse_args())
