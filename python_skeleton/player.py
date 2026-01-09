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
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # the total number of seconds your bot has left to play this game
        game_clock = game_state.game_clock
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
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
        street = previous_state.street  # 0,2,3,4,5,6 representing when this round ended
        my_cards = previous_state.hands[active]  # your cards
        # opponent's cards or [] if not revealed
        opp_cards = previous_state.hands[1-active]
        pass

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
            # unsure? is this true?
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

    def _estimate_hand_strength(self, hole_cards, board):
        """
        Estimate hand strength on a 0-1 scale.
        """
        if not hole_cards or len(hole_cards) == 0:
            return 0.0

        # get first element of tuple returned by best hand function
        hand_score = self._best_hand(hole_cards, board)
        hand_type = hand_score[0]

        # mapping hand types to strength
        strength_map = {
            10: 1.00,  # Royal Flush
            9: 0.99,   # Straight flush
            8: 0.95,   # Four of a kind
            7: 0.85,   # Full house
            6: 0.75,   # Flush
            5: 0.70,   # Straight
            4: 0.60,   # Three of a kind
            3: 0.45,   # Two pair
            2: 0.30,   # One pair
            1: 0.15,   # High card
            0: 0.05,   # Nothing yet
        }

        return strength_map.get(hand_type, 0.1)

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
        """
        Model opponent tossing without seeing our toss.
        Simple heuristic: toss lowest card.
        """
        opp_card_values = [(self.rank_values.get(c[0], 0), i) for i, c in enumerate(opp_3_cards)]
        return min(opp_card_values)[1]

    def _opponent_reacts_to_my_toss(self, opp_3_cards, board, my_tossed_card):
        """
        Model opponent reacting to seeing my toss.
        They can now make a smarter decision.
        """
        # For now, still toss their worst card
        # In Week 2+, you could add logic like:
        # - If my toss completes a straight, opponent tosses to block it
        # - If my toss is high, opponent keeps high cards
        opp_card_values = [(self.rank_values.get(c[0], 0), i) for i, c in enumerate(opp_3_cards)]
        return min(opp_card_values)[1]

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
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 2, 3, 4, 5, or 6
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.board  # the board cards

        my_pip = round_state.pips[active]  # chips you contributed this round of betting
        opp_pip = round_state.pips[1-active]  # chips opponent contributed this round of betting
        my_stack = round_state.stacks[active]  # chips you have remaining
        opp_stack = round_state.stacks[1-active]  # chips opponent has remaining

        continue_cost = opp_pip - my_pip  # chips needed to stay in the pot
        my_contribution = STARTING_STACK - my_stack  # total chips you've put in the pot
        opp_contribution = STARTING_STACK - opp_stack  # total chips opponent put in
        pot = my_contribution + opp_contribution

        # ========================================
        # TOSS DECISION - Use Monte Carlo!
        # ========================================
        if DiscardAction in legal_actions:
            time_remaining = game_state.game_clock

            # Adaptive strategy based on time remaining
            if time_remaining < 10:
                # Emergency mode - use fast heuristic only
                best_toss_idx = self._quick_toss_heuristic(my_cards)
            elif time_remaining < 20:
                # Low on time - reduced simulations
                best_toss_idx = self._monte_carlo_toss(
                    my_cards,
                    board_cards,
                    active,
                    num_simulations=50
                )
            elif time_remaining < 35:
                # Medium time - moderate simulations
                best_toss_idx = self._monte_carlo_toss(
                    my_cards,
                    board_cards,
                    active,
                    num_simulations=100
                )
            else:
                # Plenty of time - full simulations
                best_toss_idx = self._monte_carlo_toss(
                    my_cards,
                    board_cards,
                    active,
                    num_simulations=200
                )

            return DiscardAction(best_toss_idx)

        # ========================================
        # BETTING DECISION - Rule-Based
        # ========================================

        # Estimate hand strength
        strength = self._estimate_hand_strength(my_cards, board_cards)

        # Calculate pot odds for decision making
        # continue_cost is chips needed to stay in the pot
        # this calculates proportion of what we beed to input as cost vs. total in pot
        pot_odds = continue_cost / (pot + continue_cost) if pot + continue_cost > 0 else 0

        # Position: SB (player 0) tosses second and acts second - has advantage
        in_position = (active == 0)

        # ========================================
        # VERY STRONG HAND (75%+)
        # ========================================
        if strength >= 0.75:
            if RaiseAction in legal_actions:
                # if strength is high, raise bounds
                min_raise, max_raise = round_state.raise_bounds()
                # Bet bigger on later streets
                bet_multiplier = 0.75 if street >= 4 else 0.6
                bet_size = min(max_raise, my_pip + int(pot * bet_multiplier))
                return RaiseAction(max(min_raise, bet_size))
            if CallAction in legal_actions:
                return CallAction()
            return CheckAction()

        # ========================================
        # STRONG HAND (55-75%)
        # ========================================
        if strength >= 0.55:
            # Free to check
            if continue_cost == 0:
                # Value bet more often in position
                bet_frequency = 0.7 if in_position else 0.5
                if RaiseAction in legal_actions and random.random() < bet_frequency:
                    min_raise, max_raise = round_state.raise_bounds()
                    bet_size = min(max_raise, my_pip + int(pot * 0.5))
                    return RaiseAction(max(min_raise, bet_size))
                return CheckAction()

            # Facing a bet - use pot odds and hand strength
            if CallAction in legal_actions:
                # Call if we're strong or getting good odds
                if strength > 0.65 or pot_odds < 0.5:
                    return CallAction()

            # Sometimes raise as semi-bluff
            if RaiseAction in legal_actions and random.random() < 0.25:
                min_raise, max_raise = round_state.raise_bounds()
                return RaiseAction(min_raise)

            if CallAction in legal_actions:
                return CallAction()

            return CheckAction() if CheckAction in legal_actions else FoldAction()

        # ========================================
        # MEDIUM HAND (35-55%)
        # ========================================
        if strength >= 0.35:
            # Free to check - take it
            if continue_cost == 0:
                return CheckAction()

            # Use pot odds - only call if getting good price
            if CallAction in legal_actions:
                # Call if our strength beats pot odds with margin
                if strength > pot_odds * 1.2:  # Need 20% margin for safety
                    return CallAction()

            # Otherwise fold
            return FoldAction() if FoldAction in legal_actions else CheckAction()

        # ========================================
        # WEAK HAND (<35%)
        # ========================================
        if continue_cost == 0:
            # Bluff more often in position
            bluff_frequency = 0.15 if in_position else 0.08
            if RaiseAction in legal_actions and random.random() < bluff_frequency:
                min_raise, max_raise = round_state.raise_bounds()
                # Small bluff size
                return RaiseAction(min_raise)
            return CheckAction()

        # Facing a bet with weak hand - just fold
        if CheckAction in legal_actions:
            return CheckAction()

        return FoldAction() if FoldAction in legal_actions else CallAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())
