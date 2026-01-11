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

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        # Create full deck for Monte Carlo simulations
        self.deck = self._create_deck()
        # map card ranks to values
        self.rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        # Opponent modeling
        self.opp_model = {
            'hands_played': 0,
            'showdowns': [],  # List of (equity, won) tuples
            'tosses': [],     # Cards they tossed
            'aggression': 0,
            'vpip': 0,
        }

    def _create_deck(self):
        """Create a standard 52-card deck"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
        # combine like 2h
        return [r + s for r in ranks for s in suits]

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        # opponent's cards or [] if not revealed
        opp_cards = previous_state.hands[1-active]

        # Track opponent data
        self.opp_model['hands_played'] += 1

        # if they showed cards
        if opp_cards:
            board = previous_state.board
            # safety check
            if len(board) >= 2:
                # Calculate what equity they had at flop
                opp_equity = self._calculate_equity(opp_cards, board[:2], num_sims=50)
                won = my_delta < 0
                self.opp_model['showdowns'].append((opp_equity, won))

    # ==================== HAND EVALUATION HELPERS ====================
    def _evaluate_5card_hand(self, five_cards):
        """
        Evaluate a 5-card poker hand and return a score.
        Higher score = better hand.

        Returns: (hand_type, high_card) tuple for comparison
        """
        # get mapping to value
        ranks = [self.rank_values[c[0]] for c in five_cards]
        suits = [c[1] for c in five_cards]

        # Count rank frequencies
        # mapping from rank value to how many cards of that rank
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        # sort descending with highest rank in front
        # should be all uniqique values since set
        counts = sorted(rank_counts.values(), reverse=True)

        # Check for flush (all of the same suit)
        is_flush = len(set(suits)) == 1

        # Check for straight (ranks in sequence eg: 2,3,4,5,6)
        is_straight = False
        # straight_high is highest number in a straight (10-straight beats 9-straight)
        straight_high = 0
        sorted_ranks = sorted(set(ranks))
        if len(sorted_ranks) >= 5:
            # Check regular straight
            for i in range(len(sorted_ranks) - 4):
                # check if number at i and i+4 are 4 apart (if so then straight)
                if sorted_ranks[i+4] - sorted_ranks[i] == 4:
                    is_straight = True
                    straight_high = sorted_ranks[i+4]
                    break
            # Check wheel (A-2-3-4-5)
            if set([14, 2, 3, 4, 5]).issubset(set(ranks)):
                is_straight = True
                straight_high = 5

        # NOTE: we use these big numbers to multiply to differentiate different types of hands

        # Score hands (higher = better)
        if is_straight and is_flush:
            # Check for Royal Flush
            if set([14, 13, 12, 11, 10]).issubset(set(ranks)):
                return (10, 14)
            else:
                # Straight flush
                return (9, straight_high)
        if counts == [4, 1]:
            # Four of a kind
            # get the rank that has 4 and 1 of them
            quad_rank = [r for r in rank_counts if rank_counts[r] == 4][0]
            kicker = [r for r in rank_counts if rank_counts[r] == 1][0]
            return (8, quad_rank * 100 + kicker)

        if counts == [3, 2]:
            # Full house
            trips_rank = [r for r in rank_counts if rank_counts[r] == 3][0]
            pair_rank = [r for r in rank_counts if rank_counts[r] == 2][0]
            return (7, trips_rank * 100 + pair_rank)

        if is_flush:
            # Flush
            return (6, max(ranks))

        if is_straight:
            # Straight
            return (5, straight_high)

        if counts == [3, 1, 1]:
            # Three of a kind
            trips_rank = [r for r in rank_counts if rank_counts[r] == 3][0]
            kickers = sorted([r for r in rank_counts if rank_counts[r] == 1], reverse=True)
            return (4, trips_rank * 10000 + kickers[0] * 100 + kickers[1])

        if counts == [2, 2, 1]:
            # Two pair
            pairs = sorted([r for r in rank_counts if rank_counts[r] == 2], reverse=True)
            kicker = [r for r in rank_counts if rank_counts[r] == 1][0]
            return (3, pairs[0] * 10000 + pairs[1] * 100 + kicker)

        if counts == [2, 1, 1, 1]:
            # One pair
            pair_rank = [r for r in rank_counts if rank_counts[r] == 2][0]
            kickers = sorted([r for r in rank_counts if rank_counts[r] == 1], reverse=True)
            return (2, pair_rank * 1000000 + kickers[0] * 10000 + kickers[1] * 100 + kickers[2])

        # High card
        sorted_ranks_desc = sorted(ranks, reverse=True)
        tiebreaker = (sorted_ranks_desc[0] * 100000000 +
                      sorted_ranks_desc[1] * 1000000 +
                      sorted_ranks_desc[2] * 10000 +
                      sorted_ranks_desc[3] * 100 +
                      sorted_ranks_desc[4])
        return (1, tiebreaker)

    def _best_hand(self, hole_cards, board):
        """
        Find the best 5-card hand from hole cards + board.
        hole_cards: 2 cards (after toss) or 3 cards (before toss)
        board: up to 6 cards
        """
        all_cards = hole_cards + board

        if len(all_cards) < 5:
            # Not enough cards yet, return weak hand
            return (0, 0)

        best_score = (0, 0)

        # Try all 5-card combinations
        for five_cards in combinations(all_cards, 5):
            score = self._evaluate_5card_hand(list(five_cards))
            if score > best_score:
                best_score = score

        return best_score

    def _calculate_equity(self, my_cards, board, num_sims=100):
        """
        TRUE win probability via Monte Carlo.
        This replaces all hand strength calculations.
        """
        if not my_cards or len(my_cards) == 0:
            return 0.0

        wins = 0.0
        ties = 0.0
        known_cards = set(my_cards + board)
        remaining_deck = [c for c in self.deck if c not in known_cards]

        cards_needed = 6 - len(board)

        for _ in range(num_sims):
            if len(remaining_deck) < 2 + cards_needed:
                continue

            # Shuffle and deal
            sim_deck = remaining_deck.copy()
            random.shuffle(sim_deck)

            opp_cards = sim_deck[:2]
            future_board = sim_deck[2:2 + cards_needed] if cards_needed > 0 else []
            full_board = board + future_board

            # Evaluate both hands
            my_hand = self._best_hand(my_cards, full_board)
            opp_hand = self._best_hand(opp_cards, full_board)

            if my_hand > opp_hand:
                wins += 1
            elif my_hand == opp_hand:
                ties += 0.5

        return (wins + ties) / num_sims if num_sims > 0 else 0.0

    def _calculate_draw_equity(self, my_cards, board):
        """Estimate equity from draws (flush, straight)."""
        if len(board) >= 6:  # River - no draws matter
            return 0.0

        all_cards = my_cards + board
        if len(all_cards) < 4:
            return 0.0

        suits = [c[1] for c in all_cards]
        ranks = [self.rank_values[c[0]] for c in all_cards]

        draw_equity = 0.0
        cards_to_come = 6 - len(board)

        # Flush draw (4 to a flush)
        for suit in set(suits):
            if suits.count(suit) == 4:
                if cards_to_come >= 2:
                    draw_equity += 0.35  # ~35% to hit flush with 2 cards
                elif cards_to_come == 1:
                    draw_equity += 0.20  # ~20% with 1 card

        # Open-ended straight draw
        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) >= 4:
            for i in range(len(unique_ranks) - 3):
                window = unique_ranks[i:i+4]
                if max(window) - min(window) == 3:  # 4 consecutive
                    if cards_to_come >= 2:
                        draw_equity += 0.32
                    elif cards_to_come == 1:
                        draw_equity += 0.17
                    break

        return min(draw_equity, 0.40)  # Cap at 40%

    def _calculate_equity_with_draws(self, my_cards, board, num_sims=100):
        """Equity + draw potential."""
        base_equity = self._calculate_equity(my_cards, board, num_sims)

        if len(board) < 6:
            draw_bonus = self._calculate_draw_equity(my_cards, board)
            # Weight draws at 40% (they might not hit)
            return min(1.0, base_equity + draw_bonus * 0.4)

        return base_equity

    # quick toss function for if time is running out
    def _quick_toss_heuristic(self, my_cards):
        """
        Fast heuristic for toss decisions when time is tight.
        No simulations - just use simple rules.

        Priority:
        1. If we have a pair, toss the non-pair card
        2. If we have suited cards, toss the unsuited one
        3. Otherwise toss lowest card
        """
        ranks = [self.rank_values[c[0]] for c in my_cards]
        suits = [c[1] for c in my_cards]

        # If we have a pair, toss the non-pair card
        for r in set(ranks):
            if ranks.count(r) == 2:
                # Found a pair - toss the card that's not part of the pair
                for i in range(3):
                    if self.rank_values[my_cards[i][0]] != r:
                        return i

        # If we have two suited cards, toss the unsuited one
        for s in set(suits):
            if suits.count(s) == 2:
                # Found two suited cards - toss the unsuited one
                for i in range(3):
                    if my_cards[i][1] != s:
                        return i

        # Otherwise toss lowest card
        card_values = [(self.rank_values[c[0]], i) for i, c in enumerate(my_cards)]
        return min(card_values)[1]

    def _quick_prefilter_toss(self, my_cards):
        """Check for obvious toss decisions before Monte Carlo."""
        if len(my_cards) != 3:
            return None

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

        # Keep connected cards (like 8-9-7, keep 8-9)
        sorted_ranks = sorted(ranks)
        if sorted_ranks[2] - sorted_ranks[0] == 2:  # Three connected
            # Keep highest two
            for i in range(3):
                if self.rank_values[my_cards[i][0]] == sorted_ranks[0]:
                    return i

        return None  # No obvious choice

    # ==================== MONTE CARLO TOSS ====================
    def _monte_carlo_toss(self, my_cards, board, active, num_simulations=500):
        """
        Use Monte Carlo simulation to choose best card to toss.

        Args:
            my_cards: List of 3 cards
            board: Current board (should have 2 cards before toss)
            active: Your player index (0 or 1)
            num_simulations: Number of random scenarios to test

        Returns:
            Index of card to toss (0, 1, or 2)
        """
        if len(my_cards) != 3:
            return 0

        # NEW: Quick pre-filter
        quick_toss = self._quick_prefilter_toss(my_cards)
        if quick_toss is not None:
            return quick_toss

        # Determine who tosses first
        # Big blind (player 1) tosses first
        i_toss_first = (active == 1)  # True if I'm BB, False if I'm SB

        known_cards = set(my_cards + board)
        remaining_deck = [c for c in self.deck if c not in known_cards]

        best_toss_idx = 0
        best_win_rate = -1

        # Test each toss option
        for toss_idx in range(3):
            wins = 0

            for _ in range(num_simulations):
                my_final_hand = [c for i, c in enumerate(my_cards) if i != toss_idx]
                my_tossed_card = my_cards[toss_idx]

                # Simulate opponent's 3 starting cards
                sim_remaining = [c for c in remaining_deck if c != my_tossed_card]

                if len(sim_remaining) < 3:
                    continue

                opp_3_cards = random.sample(sim_remaining, 3)

                # MODEL TOSS ORDER!
                if i_toss_first:
                    # I toss first, opponent sees my toss and reacts
                    # Opponent will toss their worst card GIVEN they see my toss
                    opp_toss_idx = self._opponent_reacts_to_my_toss(
                        opp_3_cards, board, my_tossed_card
                    )
                else:
                    # Opponent tosses first (blind), I'll react later
                    # For now, assume opponent tosses reasonably (their worst card)
                    opp_toss_idx = self._opponent_tosses_blind(opp_3_cards, board)

                opp_tossed_card = opp_3_cards[opp_toss_idx]
                opp_final_hand = [c for i, c in enumerate(opp_3_cards) if i != opp_toss_idx]

                # Build board
                board_after_tosses = board + [my_tossed_card, opp_tossed_card]

                # Deal remaining community cards
                used_cards = set(my_final_hand + opp_final_hand + board_after_tosses)
                future_deck = [c for c in self.deck if c not in used_cards]

                if len(future_deck) < 2:
                    continue

                future_board = random.sample(future_deck, 2)
                full_board = board_after_tosses + future_board

                # Evaluate
                my_hand_score = self._best_hand(my_final_hand, full_board)
                opp_hand_score = self._best_hand(opp_final_hand, full_board)

                if my_hand_score > opp_hand_score:
                    wins += 1
                elif my_hand_score == opp_hand_score:
                    wins += 0.5

            win_rate = wins / num_simulations if num_simulations > 0 else 0

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_toss_idx = toss_idx

        return best_toss_idx

    def _opponent_tosses_blind(self, opp_3_cards, board):
        """Model opponent choosing optimal toss based on equity."""
        best_toss_idx = 0
        best_equity = -1

        for toss_idx in range(3):
            remaining = [c for i, c in enumerate(opp_3_cards) if i != toss_idx]
            # Simulate what equity they'd have after this toss
            equity = self._calculate_equity(remaining, board, num_sims=30)

            if equity > best_equity:
                best_equity = equity
                best_toss_idx = toss_idx

        return best_toss_idx

    def _opponent_reacts_to_my_toss(self, opp_3_cards, board, my_tossed_card):
        """Opponent sees my toss and optimizes."""
        board_with_my_toss = board + [my_tossed_card]
        return self._opponent_tosses_blind(opp_3_cards, board_with_my_toss)


    def _get_gto_bet_size(self, equity, pot, my_pip, street, in_position):
        """
        GTO-inspired bet sizing with randomization.
        Returns the actual bet amount to make (not just multiplier).
        """

        # River - polarized strategy
        if street >= 5:
            if equity >= 0.70:
                # Strong value bet
                return int(pot * random.uniform(0.70, 0.90))
            elif equity >= 0.55:
                # Medium - mostly check, sometimes small bet
                if random.random() < 0.35:
                    return int(pot * random.uniform(0.40, 0.60))
                return 0  # Check
            elif equity >= 0.30:
                # Bluff occasionally
                if random.random() < 0.25:
                    return int(pot * random.uniform(0.60, 0.80))
                return 0
            else:
                # Weak - rarely bluff
                if random.random() < 0.10:
                    return int(pot * 0.50)
                return 0

        # Turn
        elif street >= 4:
            if equity >= 0.65:
                return int(pot * random.uniform(0.60, 0.75))
            elif equity >= 0.50:
                return int(pot * random.uniform(0.45, 0.60))
            elif equity >= 0.35:
                if random.random() < 0.20:
                    return int(pot * random.uniform(0.35, 0.50))
                return 0
            else:
                if random.random() < 0.15:
                    return int(pot * 0.40)
                return 0

        # Flop and earlier
        else:
            if equity >= 0.60:
                return int(pot * random.uniform(0.55, 0.70))
            elif equity >= 0.45:
                return int(pot * random.uniform(0.40, 0.55))
            else:
                if random.random() < 0.20:
                    return int(pot * 0.40)
                return 0

    def _should_call(self, equity, pot_odds, pot, my_stack, opp_stack, street):
        """
        Decide whether to call based on equity, pot odds, and implied odds.
        """

        # Direct pot odds - if equity > pot odds, it's profitable
        if equity >= pot_odds:
            return True

        # Implied odds - if we have draws and deep stacks
        if street < 5:  # Not river
            effective_stack = min(my_stack, opp_stack)

            # Close to correct odds + deep stacks = call for implied odds
            if equity >= pot_odds * 0.85 and effective_stack > pot * 2:
                return True

        # Way behind - just fold
        if equity < pot_odds * 0.65:
            return False

        # Marginal spot - add some randomness (GTO balance)
        threshold = equity / pot_odds
        return random.random() < threshold


    def _get_adaptive_sim_count(self, time_remaining, street):
        """Allocate simulations based on time and street importance."""

        rounds_played = self.opp_model['hands_played']

        if time_remaining < 5:
            return {'equity': 10, 'toss': 15}
        elif time_remaining < 15:
            return {'equity': 20, 'toss': 30}
        elif time_remaining < 30:
            if street >= 5:
                return {'equity': 40, 'toss': 50}
            else:
                return {'equity': 25, 'toss': 30}
        else:
            if street >= 5:
                return {'equity': 60, 'toss': 80}
            elif street >= 4:
                return {'equity': 50, 'toss': 60}
            else:
                return {'equity': 30, 'toss': 40}


    def get_action(self, game_state, round_state, active):
        '''
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()
        street = round_state.street  # 0, 2, 3, 4, 5, or 6
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
            # Adaptive strategy based on time remaining
            if time_remaining < 10:
                best_toss_idx = self._quick_toss_heuristic(my_cards)
            elif time_remaining < 20:
                best_toss_idx = self._monte_carlo_toss(
                    my_cards, board_cards, active, num_simulations=30
                )
            elif time_remaining < 35:
                best_toss_idx = self._monte_carlo_toss(
                    my_cards, board_cards, active, num_simulations=50
                )
            else:
                best_toss_idx = self._monte_carlo_toss(
                    my_cards, board_cards, active, num_simulations=80
                )

            return DiscardAction(best_toss_idx)

        # ========================================
        # BETTING DECISION
        # ========================================

        # Get adaptive simulation count
        sim_counts = self._get_adaptive_sim_count(time_remaining, street)

        # Calculate equity with draws
        equity = self._calculate_equity_with_draws(
            my_cards, board_cards, num_sims=sim_counts['equity']
        )

        # Calculate pot odds
        pot_odds = continue_cost / (pot + continue_cost) if pot + continue_cost > 0 else 0

        # Position (SB has advantage - tosses second, acts second on most streets)
        in_position = (active == 0)

        # Stack depth
        effective_stack = min(my_stack, opp_stack)

        # ========================================
        # FACING A BET (continue_cost > 0)
        # ========================================
        if continue_cost > 0:
            # Very strong hands (70%+ equity) - raise frequently
            if equity >= 0.70:
                if RaiseAction in legal_actions and random.random() < 0.60:
                    min_raise, max_raise = round_state.raise_bounds()
                    # Raise 2-3x the pot for value
                    raise_size = min(max_raise, opp_pip + int((pot + continue_cost) * random.uniform(0.7, 1.0)))
                    return RaiseAction(max(min_raise, raise_size))

                # Always call if we can't raise
                return CallAction() if CallAction in legal_actions else CheckAction()

            # Strong hands (55-70%) - mostly call, sometimes raise
            elif equity >= 0.55:
                if RaiseAction in legal_actions and random.random() < 0.30:
                    min_raise, max_raise = round_state.raise_bounds()
                    raise_size = min(max_raise, opp_pip + int(pot * 0.6))
                    return RaiseAction(max(min_raise, raise_size))

                return CallAction() if CallAction in legal_actions else CheckAction()

            # Medium hands (35-55%) - use pot odds + implied odds
            elif equity >= 0.35:
                should_call = self._should_call(
                    equity, pot_odds, pot, my_stack, opp_stack, street
                )

                if should_call and CallAction in legal_actions:
                    return CallAction()

                return FoldAction() if FoldAction in legal_actions else CheckAction()

            # Weak hands (<35%) - mostly fold, occasionally bluff
            else:
                # Bluff raise when in position (positional advantage)
                if in_position and RaiseAction in legal_actions and random.random() < 0.12:
                    min_raise, max_raise = round_state.raise_bounds()
                    return RaiseAction(min_raise)

                return FoldAction() if FoldAction in legal_actions else CheckAction()

        # ========================================
        # NOT FACING A BET (can check or bet)
        # ========================================
        else:
            # Get GTO-inspired bet size
            bet_size = self._get_gto_bet_size(equity, pot, my_pip, street, in_position)

            if bet_size > 0 and RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                actual_bet = my_pip + bet_size
                actual_bet = max(min_raise, min(max_raise, actual_bet))
                return RaiseAction(actual_bet)

            # Otherwise check
            return CheckAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())
