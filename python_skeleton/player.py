'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
from itertools import combinations


class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.
        '''
        self.deck = self._create_deck()
        self.rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        self.rounds_played = 0

        # LIGHTWEIGHT opponent modeling (just counters - instant!)
        self.opp_vpip = 0  # Hands where opponent put in money
        self.opp_preflop_raises = 0  # Times they raised preflop

    def _create_deck(self):
        """Create a standard 52-card deck"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
        return [r + s for r in ranks for s in suits]

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''Track opponent stats - INSTANT (no simulations)'''
        self.rounds_played += 1

        # Track opponent VPIP (do they play lots of hands?)
        previous_state = terminal_state.previous_state
        if previous_state:
            opp_pip = previous_state.pips[1-active]
            # If they put in more than big blind, they played this hand
            if opp_pip > 2:
                self.opp_vpip += 1

    def _get_opponent_type(self):
        """Classify opponent - INSTANT (just division)"""
        if self.rounds_played < 50:
            return "UNKNOWN"  # Need data first

        vpip_rate = self.opp_vpip / self.rounds_played

        # LOOSE = weak bot (plays too many hands)
        if vpip_rate > 0.50:
            return "LOOSE"
        # TIGHT = strong bot (plays few hands)
        elif vpip_rate < 0.30:
            return "TIGHT"
        # NORMAL = balanced
        else:
            return "NORMAL"

    # ==================== HAND EVALUATION ====================
    def _evaluate_5card_hand(self, five_cards):
        """Evaluate a 5-card poker hand and return a score."""
        ranks = [self.rank_values[c[0]] for c in five_cards]
        suits = [c[1] for c in five_cards]

        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        counts = sorted(rank_counts.values(), reverse=True)

        is_flush = len(set(suits)) == 1
        is_straight = False
        straight_high = 0
        sorted_ranks = sorted(set(ranks))

        if len(sorted_ranks) >= 5:
            for i in range(len(sorted_ranks) - 4):
                if sorted_ranks[i+4] - sorted_ranks[i] == 4:
                    is_straight = True
                    straight_high = sorted_ranks[i+4]
                    break
            if set([14, 2, 3, 4, 5]).issubset(set(ranks)):
                is_straight = True
                straight_high = 5

        if is_straight and is_flush:
            if set([14, 13, 12, 11, 10]).issubset(set(ranks)):
                return (10, 14)
            return (9, straight_high)
        if counts == [4, 1]:
            quad_rank = [r for r in rank_counts if rank_counts[r] == 4][0]
            kicker = [r for r in rank_counts if rank_counts[r] == 1][0]
            return (8, quad_rank * 100 + kicker)
        if counts == [3, 2]:
            trips_rank = [r for r in rank_counts if rank_counts[r] == 3][0]
            pair_rank = [r for r in rank_counts if rank_counts[r] == 2][0]
            return (7, trips_rank * 100 + pair_rank)
        if is_flush:
            return (6, max(ranks))
        if is_straight:
            return (5, straight_high)
        if counts == [3, 1, 1]:
            trips_rank = [r for r in rank_counts if rank_counts[r] == 3][0]
            kickers = sorted([r for r in rank_counts if rank_counts[r] == 1], reverse=True)
            return (4, trips_rank * 10000 + kickers[0] * 100 + kickers[1])
        if counts == [2, 2, 1]:
            pairs = sorted([r for r in rank_counts if rank_counts[r] == 2], reverse=True)
            kicker = [r for r in rank_counts if rank_counts[r] == 1][0]
            return (3, pairs[0] * 10000 + pairs[1] * 100 + kicker)
        if counts == [2, 1, 1, 1]:
            pair_rank = [r for r in rank_counts if rank_counts[r] == 2][0]
            kickers = sorted([r for r in rank_counts if rank_counts[r] == 1], reverse=True)
            return (2, pair_rank * 1000000 + kickers[0] * 10000 + kickers[1] * 100 + kickers[2])

        sorted_ranks_desc = sorted(ranks, reverse=True)
        tiebreaker = (sorted_ranks_desc[0] * 100000000 +
                      sorted_ranks_desc[1] * 1000000 +
                      sorted_ranks_desc[2] * 10000 +
                      sorted_ranks_desc[3] * 100 +
                      sorted_ranks_desc[4])
        return (1, tiebreaker)

    def _best_hand(self, hole_cards, board):
        """Find the best 5-card hand from hole cards + board."""
        all_cards = hole_cards + board
        if len(all_cards) < 5:
            return (0, 0)

        best_score = (0, 0)
        for five_cards in combinations(all_cards, 5):
            score = self._evaluate_5card_hand(list(five_cards))
            if score > best_score:
                best_score = score
        return best_score

    # ==================== EQUITY CALCULATION ====================
    def _calculate_equity(self, my_cards, board, num_sims=15):
        """Calculate win probability via Monte Carlo."""
        if not my_cards:
            return 0.5

        wins = 0.0
        known_cards = set(my_cards + board)
        remaining_deck = [c for c in self.deck if c not in known_cards]
        cards_needed = 6 - len(board)

        for _ in range(num_sims):
            if len(remaining_deck) < 2 + cards_needed:
                continue

            sim_deck = remaining_deck.copy()
            random.shuffle(sim_deck)

            opp_cards = sim_deck[:2]
            future_board = sim_deck[2:2 + cards_needed] if cards_needed > 0 else []
            full_board = board + future_board

            my_hand = self._best_hand(my_cards, full_board)
            opp_hand = self._best_hand(opp_cards, full_board)

            if my_hand > opp_hand:
                wins += 1
            elif my_hand == opp_hand:
                wins += 0.5

        return wins / num_sims if num_sims > 0 else 0.5

    def _calculate_draw_equity(self, my_cards, board):
        """Calculate bonus equity from flush/straight draws - FAST!"""
        if len(board) >= 6:
            return 0.0

        all_cards = my_cards + board
        if len(all_cards) < 4:
            return 0.0

        suits = [c[1] for c in all_cards]
        ranks = [self.rank_values[c[0]] for c in all_cards]
        cards_to_come = 6 - len(board)
        draw_equity = 0.0

        # Flush draw (4 cards of same suit)
        for suit in set(suits):
            if suits.count(suit) == 4:
                if cards_to_come >= 2:
                    draw_equity += 0.35
                elif cards_to_come == 1:
                    draw_equity += 0.20

        # Straight draw (4 connected cards)
        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) >= 4:
            for i in range(len(unique_ranks) - 3):
                window = unique_ranks[i:i+4]
                if max(window) - min(window) == 3:
                    if cards_to_come >= 2:
                        draw_equity += 0.32
                    elif cards_to_come == 1:
                        draw_equity += 0.17
                    break

        return min(draw_equity, 0.40)

    def _is_dangerous_board(self, board_cards):
        """Check if board is dangerous (pairs, flush/straight draws)"""
        if len(board_cards) < 2:
            return False

        # Check for paired board (trips possible)
        board_ranks = [self.rank_values[c[0]] for c in board_cards]
        if len(board_ranks) != len(set(board_ranks)):
            return True

        # Check for 3+ cards on board (straight/flush possible)
        if len(board_cards) >= 3:
            # Check flush draw
            suits = [c[1] for c in board_cards]
            for suit in set(suits):
                if suits.count(suit) >= 3:
                    return True

            # Check straight draw
            sorted_ranks = sorted(set(board_ranks))
            if len(sorted_ranks) >= 3:
                for i in range(len(sorted_ranks) - 2):
                    if sorted_ranks[i+2] - sorted_ranks[i] <= 4:
                        return True

        return False

    # ==================== TOSS DECISION ====================
    def _smart_toss(self, my_cards):
        """Fast heuristic toss - keeps best cards."""
        ranks = [self.rank_values[c[0]] for c in my_cards]
        suits = [c[1] for c in my_cards]

        # ALWAYS keep pocket pairs
        for r in set(ranks):
            if ranks.count(r) == 2:
                for i in range(3):
                    if self.rank_values[my_cards[i][0]] != r:
                        return i

        # Keep suited cards
        for s in set(suits):
            if suits.count(s) == 2:
                for i in range(3):
                    if my_cards[i][1] != s:
                        return i

        # Keep connected cards (e.g., 8-9-7 → keep 8-9, toss 7)
        sorted_ranks = sorted(ranks)
        if sorted_ranks[2] - sorted_ranks[0] == 2:
            for i in range(3):
                if self.rank_values[my_cards[i][0]] == sorted_ranks[0]:
                    return i

        # Toss lowest card
        card_values = [(self.rank_values[c[0]], i) for i, c in enumerate(my_cards)]
        return min(card_values)[1]

    # ==================== BETTING HELPERS ====================
    def _get_bet_size(self, equity, pot, street):
        """GTO-inspired bet sizing based on equity and street."""

        # River - polarized (strong or bluff)
        if street >= 5:
            if equity >= 0.70:
                return int(pot * random.uniform(0.65, 0.85))
            elif equity >= 0.55:
                if random.random() < 0.35:
                    return int(pot * random.uniform(0.40, 0.55))
                return 0
            elif equity >= 0.30:
                if random.random() < 0.20:
                    return int(pot * random.uniform(0.55, 0.75))
                return 0
            else:
                if random.random() < 0.10:
                    return int(pot * 0.50)
                return 0

        # Turn
        elif street >= 4:
            if equity >= 0.65:
                return int(pot * random.uniform(0.55, 0.70))
            elif equity >= 0.50:
                return int(pot * random.uniform(0.40, 0.55))
            elif equity >= 0.35:
                if random.random() < 0.25:
                    return int(pot * 0.45)
                return 0
            else:
                if random.random() < 0.15:
                    return int(pot * 0.40)
                return 0

        # Flop and earlier
        else:
            if equity >= 0.60:
                return int(pot * random.uniform(0.50, 0.65))
            elif equity >= 0.45:
                return int(pot * random.uniform(0.35, 0.50))
            else:
                if random.random() < 0.20:
                    return int(pot * 0.40)
                return 0

    def _should_call(self, equity, pot_odds, pot, my_stack, opp_stack, street):
        """Decide whether to call based on equity and pot odds."""

        # Direct pot odds
        if equity >= pot_odds:
            return True

        # Implied odds (earlier streets with deep stacks)
        if street < 5:
            effective_stack = min(my_stack, opp_stack)
            if equity >= pot_odds * 0.85 and effective_stack > pot * 2:
                return True

        # Way behind
        if equity < pot_odds * 0.65:
            return False

        # Marginal - add randomness
        threshold = equity / pot_odds if pot_odds > 0 else 0
        return random.random() < threshold

    # ==================== PREFLOP HAND STRENGTH ====================
    def _evaluate_preflop_hand(self, my_cards):
        """Evaluate preflop hand strength (0-10 scale)"""
        ranks = sorted([self.rank_values[c[0]] for c in my_cards], reverse=True)
        suits = [c[1] for c in my_cards]

        # Check for pairs
        has_pair = len(ranks) != len(set(ranks))

        # Pocket pairs
        if has_pair:
            pair_rank = max([r for r in set(ranks) if ranks.count(r) >= 2])
            if pair_rank >= 13:  # AA, KK
                return 10
            elif pair_rank >= 10:  # QQ, JJ, TT
                return 8
            elif pair_rank >= 7:  # 99-77
                return 6
            else:  # 66-22
                return 4

        # High cards
        if ranks[0] == 14:  # Ace
            if ranks[1] >= 12:  # AK, AQ
                return 7 if suits[0] == suits[1] else 6
            elif ranks[1] >= 10:  # AJ, AT
                return 5 if suits[0] == suits[1] else 4
            else:  # A9-A2
                return 3

        if ranks[0] == 13:  # King
            if ranks[1] >= 11:  # KQ, KJ
                return 5 if suits[0] == suits[1] else 4
            elif ranks[1] >= 9:  # KT, K9
                return 3

        if ranks[0] == 12 and ranks[1] >= 10:  # QJ, QT
            return 4 if suits[0] == suits[1] else 3

        # Connected suited
        if suits[0] == suits[1] and abs(ranks[0] - ranks[1]) <= 2:
            return 3

        # Garbage
        return 1

    # ==================== MAIN ACTION ====================
    def get_action(self, game_state, round_state, active):
        '''Main decision function.'''
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.board

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]

        continue_cost = opp_pip - my_pip
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        pot = my_contribution + opp_contribution

        time_remaining = game_state.game_clock

        # ========================================
        # TOSS DECISION
        # ========================================
        if DiscardAction in legal_actions:
            return DiscardAction(self._smart_toss(my_cards))

        # ========================================
        # ADAPTIVE PREFLOP (Based on Opponent Type)
        # ========================================
        if street == 0:
            hand_strength = self._evaluate_preflop_hand(my_cards)
            opp_type = self._get_opponent_type()  # INSTANT - just division

            # FACING A RAISE
            if continue_cost > 2:
                # Adjust fold threshold based on opponent type
                if opp_type == "TIGHT":
                    # vs strong bots - fold more (they have strong hands)
                    fold_threshold = 6
                elif opp_type == "LOOSE":
                    # vs weak bots - call more (they raise garbage)
                    fold_threshold = 4
                else:
                    # vs unknown/normal - balanced
                    fold_threshold = 5

                # Apply threshold
                if hand_strength < fold_threshold:
                    return FoldAction() if FoldAction in legal_actions else CheckAction()
                elif hand_strength < 7 and continue_cost > pot * 0.5:
                    # Fold medium hands vs huge raises
                    return FoldAction() if FoldAction in legal_actions else CheckAction()

            # NOT FACING A RAISE
            else:
                # Only raise with decent hands (5+)
                if hand_strength >= 5 and RaiseAction in legal_actions:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(min_raise + random.randint(0, 2))
                # Call/check with weaker hands
                elif hand_strength >= 3:
                    if CheckAction in legal_actions:
                        return CheckAction()
                    return CallAction()
                # Fold garbage
                else:
                    if CheckAction in legal_actions and continue_cost == 0:
                        return CheckAction()
                    return FoldAction() if FoldAction in legal_actions else CheckAction()

        # ========================================
        # POSTFLOP BETTING
        # ========================================

        # Adaptive sim count
        if time_remaining < 15:
            num_sims = 12
        elif time_remaining < 30:
            num_sims = 20
        else:
            num_sims = 35 if street >= 5 else 25

        # Calculate equity
        base_equity = self._calculate_equity(my_cards, board_cards, num_sims=num_sims)

        # Add draw equity bonus
        if len(board_cards) < 6:
            draw_bonus = self._calculate_draw_equity(my_cards, board_cards)
            equity = min(1.0, base_equity + draw_bonus * 0.35)
        else:
            equity = base_equity

        # Reduce equity on dangerous boards
        if self._is_dangerous_board(board_cards):
            equity = equity * 0.90

        pot_odds = continue_cost / (pot + continue_cost) if pot + continue_cost > 0 else 0
        in_position = (active == 0)

        # Get opponent type for adaptive strategy
        opp_type = self._get_opponent_type()

        # ========================================
        # FACING A BET
        # ========================================
        if continue_cost > 0:
            bet_to_pot_ratio = continue_cost / pot if pot > 0 else 1

            # Very strong (70%+)
            if equity >= 0.70:
                if RaiseAction in legal_actions and random.random() < 0.55:
                    min_raise, max_raise = round_state.raise_bounds()
                    raise_size = min(max_raise, opp_pip + int(pot * random.uniform(0.65, 0.95)))
                    return RaiseAction(max(min_raise, raise_size))
                return CallAction() if CallAction in legal_actions else CheckAction()

            # Strong (55-70%)
            elif equity >= 0.55:
                if RaiseAction in legal_actions and random.random() < 0.30:
                    min_raise, max_raise = round_state.raise_bounds()
                    raise_size = min(max_raise, opp_pip + int(pot * 0.55))
                    return RaiseAction(max(min_raise, raise_size))
                return CallAction() if CallAction in legal_actions else CheckAction()

            # ADAPTIVE Medium (35-55%) - adjust based on opponent
            elif equity >= 0.35:
                # vs TIGHT bots - fold more to big bets (they have it)
                if opp_type == "TIGHT" and bet_to_pot_ratio > 0.6:
                    if equity < 0.50:
                        return FoldAction() if FoldAction in legal_actions else CheckAction()

                # vs LOOSE bots - call more (they bluff)
                elif opp_type == "LOOSE" and bet_to_pot_ratio > 0.6:
                    if equity < 0.42:
                        return FoldAction() if FoldAction in legal_actions else CheckAction()

                # vs NORMAL/UNKNOWN - balanced
                else:
                    if bet_to_pot_ratio > 0.6 and equity < 0.47:
                        return FoldAction() if FoldAction in legal_actions else CheckAction()

                # Use pot odds for smaller bets
                if self._should_call(equity, pot_odds, pot, my_stack, opp_stack, street):
                    return CallAction() if CallAction in legal_actions else CheckAction()
                return FoldAction() if FoldAction in legal_actions else CheckAction()

            # Weak (<35%)
            else:
                # Bluff more vs TIGHT bots (they fold)
                if opp_type == "TIGHT" and in_position and RaiseAction in legal_actions and random.random() < 0.12:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(min_raise)
                # Bluff less vs LOOSE bots (they call)
                elif in_position and RaiseAction in legal_actions and random.random() < 0.05:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(min_raise)
                return FoldAction() if FoldAction in legal_actions else CheckAction()

        # ========================================
        # NOT FACING A BET
        # ========================================
        else:
            bet_size = self._get_bet_size(equity, pot, street)

            if bet_size > 0 and RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                actual_bet = my_pip + bet_size
                actual_bet = max(min_raise, min(max_raise, actual_bet))
                return RaiseAction(actual_bet)

            return CheckAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())
