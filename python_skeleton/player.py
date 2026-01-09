'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random


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
        pass

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

def get_action(self, game_state, round_state, active):
    import random
    from itertools import combinations

    # ---------- Core state ----------
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

    # ---------- Helpers ----------
    rank_order = {
        '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
        '8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14
    }

    def hand_bucket(cards, board):
        """Very cheap hand strength bucket."""
        ranks = [c[0] for c in cards + board]
        counts = {r: ranks.count(r) for r in ranks}

        if max(counts.values()) >= 3:
            return "monster"
        if max(counts.values()) == 2:
            return "pair"
        return "weak"

    # ---------- DISCARD LOGIC ----------
    # ---------- MONTE CARLO DISCARD ----------
    if DiscardAction in legal_actions:
        from itertools import combinations

        SIMS = 25  # 20–30 is the sweet spot

        rank_order = {
        '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
        '8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14
        }

        def evaluate_hand(cards, board):
            """Very crude hand strength score."""
            ranks = [c[0] for c in cards + board]
            counts = sorted([ranks.count(r) for r in set(ranks)], reverse=True)

            if counts[0] >= 4:
                return 7  # quads+
            if counts[0] == 3:
                return 6  # trips / full house
            if counts[0] == 2:
                return 3  # pair / two pair
            return 1      # high card

        def mc_score(discard_idx):
            wins = 0

            for _ in range(SIMS):
                # Copy state
                my_remaining = my_cards[:]
                discarded = my_remaining.pop(discard_idx)

                # Random opponent discard
                opp_cards = round_state.hands[1-active][:]
                opp_discard = random.choice(range(len(opp_cards)))
                opp_remaining = opp_cards[:]
                opp_remaining.pop(opp_discard)

                # Board after discards
                sim_board = board_cards[:] + [discarded, opp_cards[opp_discard]]

                # Randomly deal remaining board cards
                deck = []
                for r in rank_order:
                    for s in "shdc":
                        card = r + s
                        if card not in sim_board and card not in my_remaining and card not in opp_remaining:
                            deck.append(card)

                random.shuffle(deck)

                # Fill board to 6 cards
                while len(sim_board) < 6:
                    sim_board.append(deck.pop())

                my_score = evaluate_hand(my_remaining, sim_board)
                opp_score = evaluate_hand(opp_remaining, sim_board)

                if my_score > opp_score:
                    wins += 1
                elif my_score == opp_score:
                    wins += 0.5

            return wins / SIMS

        # Pick discard with highest MC win rate
        best_idx = max(range(len(my_cards)), key=lambda i: mc_score(i))
        return DiscardAction(best_idx)


    # ---------- BETTING LOGIC ----------
    bucket = hand_bucket(my_cards, board_cards)

    # If we can check, usually do it
    if CheckAction in legal_actions:
        if bucket == "monster" and RaiseAction in legal_actions and street >= 4:
            min_raise, _ = round_state.raise_bounds()
            return RaiseAction(min_raise)
        return CheckAction()

    # If facing a bet
    if continue_cost > 0:
        # Monster hands: call or raise
        if bucket == "monster":
            if RaiseAction in legal_actions and street >= 4:
                min_raise, _ = round_state.raise_bounds()
                return RaiseAction(min_raise)
            return CallAction()

        # Pair hands: call small, fold big
        if bucket == "pair":
            if pot > 0 and continue_cost / pot < 0.4:
                return CallAction()
            return FoldAction()

        # Weak hands: fold to pressure
        return FoldAction()

    # No pressure, optional small value bet
    if RaiseAction in legal_actions and bucket == "monster":
        min_raise, _ = round_state.raise_bounds()
        return RaiseAction(min_raise)

    return CallAction()




if __name__ == '__main__':
    run_bot(Player(), parse_args())
