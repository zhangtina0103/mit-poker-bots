from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
from itertools import combinations


TOURNAMENT_MODE = False

# Try to import fast evaluator
try:
    import pkrbot
    HAVE_PKRBOT = True
except Exception:
    HAVE_PKRBOT = False

# Try to import C++ fast equity calculator
try:
    from cpp_equity import calculate_equity_fast, is_available as cpp_available
    HAVE_CPP_EQUITY = cpp_available()
    if HAVE_CPP_EQUITY:
        print("[v7.4] C++ equity engine loaded successfully")
    else:
        print("[v7.4] C++ equity engine NOT available - using Python fallback")
except Exception as e:
    HAVE_CPP_EQUITY = False
    print(f"[v7.4] C++ equity engine failed to load: {e}")

# Import DP tables
try:
    from dp_tables import (
        get_bet_size, get_fold_frequency, get_bluff_frequency,
        categorize_hand_strength, categorize_pot, categorize_bet, categorize_street,
        get_bluff_size, get_value_size
    )
    USE_DP_TABLES = True
except ImportError:
    USE_DP_TABLES = False

# constants
TARGET_BOARD_SIZE = 6
BETTING_STREETS = {0, 4, 5, 6}
DISCARD_STREETS = {2, 3}

# Hand type constants
HAND_HIGH_CARD = 1
HAND_PAIR = 2
HAND_TWO_PAIR = 3
HAND_TRIPS = 4
HAND_STRAIGHT = 5
HAND_FLUSH = 6
HAND_FULL_HOUSE = 7
HAND_QUADS = 8
HAND_STRAIGHT_FLUSH = 9
HAND_ROYAL_FLUSH = 10

# Raises of 150+ chips preflop are "big raises"
BIG_RAISE_THRESHOLD = 150
# 40% of stack or more counts as a shove for tracking
SHOVE_STACK_RATIO = 0.40

# preflop table
PREFLOP_TIERS = {
    # Very strong
    'AA': 1, 'KK': 1, 'QQ': 1, 'JJ': 1, 'TT': 1,
    'AKs': 1, 'AQs': 1,
    # Strong
    '99': 2, '88': 2, '77': 2,
    'AKo': 1, 'AQo': 2, 'AJs': 2, 'ATs': 2,
    'KQs': 2, 'KJs': 2, 'KTs': 2,
    'QJs': 2, 'JTs': 2,
    # Playable
    '66': 3, '55': 3, '44': 3, '33': 3, '22': 3,
    'AJo': 2, 'ATo': 3, 'A9s': 3, 'A8s': 3, 'A7s': 3,
    'A6s': 3, 'A5s': 3, 'A4s': 3, 'A3s': 3, 'A2s': 3,
    'KQo': 2, 'KJo': 3, 'KTo': 3, 'K9s': 3,
    'QTs': 3, 'Q9s': 3, 'QJo': 3, 'QTo': 3,
    'J9s': 3, 'JTo': 3,
    'T9s': 3, '98s': 3, '87s': 3, '76s': 3, '65s': 3, '54s': 3,
    # Defense range
    'A9o': 4, 'A8o': 4, 'A7o': 4, 'A6o': 4, 'A5o': 4,
    'K9o': 4, 'K8o': 4, 'Q9o': 4, 'J9o': 4, 'T9o': 4,
    '98o': 4, '87o': 4, '76o': 4, '65o': 4,
    '43s': 4, '32s': 4,
}


class Player(Bot):
    def __init__(self):
        self.deck = self._create_deck()
        self.rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        self.rank_chars = {v: k for k, v in self.rank_values.items()}
        self.rounds_played = 0

        self.equity_cache = {}

        # Opponent modeling
        self.opp_vpip = 0
        self.opp_preflop_raises = 0
        self.opp_postflop_aggression = 0
        self.opp_postflop_actions = 0
        self.opp_fold_to_raise = 0
        self.opp_raises_faced = 0
        self.opp_preflop_shoves = 0
        self.opp_big_raises = 0

        # Check-Raise tracking for postflop defense
        self.opp_check_raises = 0
        self.opp_flop_raises_faced = 0

        # Track opponent blind folds
        self.opp_blind_folds = 0
        self.opp_blind_opportunities = 0

        # Per-round tracking
        self.recent_results = []
        self.my_raises_this_round = 0

        # Per-street tracking
        self.raise_count_this_street = 0
        self._last_street = None
        # Track if we bet (for check-raise detection)
        self.we_bet_this_street = False

        # Track consecutive losses
        self.consecutive_losses = 0

        # Track preflop aggressor
        self.was_preflop_aggressor = False

        # Track opponent's multi-street pressure (for defense)
        # Streets where opp bet >45% pot
        self.opp_streets_pressured = set()

        # Track recent opponent aggression for counter-exploit
        self.opp_recent_bets = 0  # Bets in last 20 rounds
        self.opp_recent_checks = 0  # Checks in last 20 rounds

    def _create_deck(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
        return [r + s for r in ranks for s in suits]

    def handle_new_round(self, game_state, round_state, active):
        self.equity_cache = {}
        self.my_raises_this_round = 0
        self.raise_count_this_street = 0
        self._last_street = None
        self.was_preflop_aggressor = False
        self.we_bet_this_street = False
        # Reset multi-street pressure tracking
        self.opp_streets_pressured = set()

    def handle_round_over(self, game_state, terminal_state, active):
        self.rounds_played += 1

        delta = terminal_state.deltas[active]
        self.recent_results.append(delta)
        if len(self.recent_results) > 20:
            self.recent_results.pop(0)

        if delta < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Track fold-to-raise
        prev_state = terminal_state.previous_state
        if prev_state and hasattr(prev_state, 'street'):
            hand_ended_by_fold = (prev_state.street < 5)
            if hand_ended_by_fold and delta > 0 and self.my_raises_this_round > 0:
                self.opp_fold_to_raise += 1

        if self.my_raises_this_round > 0:
            self.opp_raises_faced += 1

        # Track blind folds for Nit Cracker
        # If we were BB and won without showdown, opponent likely folded
        if delta > 0 and prev_state and hasattr(prev_state, 'street'):
            if prev_state.street == 0:  # Ended preflop
                self.opp_blind_folds += 1
            self.opp_blind_opportunities += 1

        # VPIP counting and shove tracking
        state = terminal_state.previous_state
        counted_vpip = False
        counted_agg = False

        while state is not None and hasattr(state, 'previous_state') and state.previous_state is not None:
            prev_state = state.previous_state
            if not hasattr(prev_state, 'street'):
                break

            street = prev_state.street

            if street == 0 and not counted_vpip:
                opp_pip = prev_state.pips[1-active]
                forced_blind = SMALL_BLIND if (1-active) == 0 else BIG_BLIND
                if opp_pip > forced_blind:
                    self.opp_vpip += 1

                # Track both shoves AND big raises separately
                if opp_pip >= STARTING_STACK * SHOVE_STACK_RATIO:
                    self.opp_preflop_shoves += 1
                elif opp_pip >= BIG_RAISE_THRESHOLD:
                    self.opp_big_raises += 1

                counted_vpip = True

            if street in BETTING_STREETS and street > 0 and not counted_agg:
                opp_pip_now = state.pips[1-active] if hasattr(state, 'pips') else 0
                opp_pip_prev = prev_state.pips[1-active]
                if opp_pip_now > opp_pip_prev:
                    self.opp_postflop_aggression += 1
                self.opp_postflop_actions += 1
                counted_agg = True

            state = prev_state

    def _get_opponent_type(self):
        # Detect more specific playstyles including trappers
        if self.rounds_played < 3:
            return "UNKNOWN"

        # Combined shove + big raise rate for faster detection
        aggro_raise_rate = (self.opp_preflop_shoves + self.opp_big_raises) / self.rounds_played
        if aggro_raise_rate > 0.20:
            return "SHOVE_BOT"

        shove_rate = self.opp_preflop_shoves / self.rounds_played
        if shove_rate > 0.12:
            return "SHOVE_BOT"

        vpip_rate = self.opp_vpip / self.rounds_played
        agg_rate = self.opp_postflop_aggression / max(1, self.opp_postflop_actions)

        # detect low aggression but wins big pots
        # They check a lot, then suddenly bet big when they have it
        if vpip_rate > 0.35 and agg_rate < 0.30:
            # Playing lots of hands but not aggressive = trapper or calling station
            if self.opp_check_raises >= 3:
                return "TRAPPER"
            return "CALLING_STATION"

        # Detect lots of tiny bets through bet sizing patterns

        if vpip_rate > 0.60:
            return "MANIAC" if agg_rate >= 0.45 else "CALLING_STATION"
        elif vpip_rate > 0.45:
            return "LOOSE_AGGRESSIVE" if agg_rate >= 0.35 else "LOOSE_PASSIVE"
        elif vpip_rate < 0.25:
            return "VERY_TIGHT"
        elif vpip_rate < 0.35:
            return "TIGHT"
        return "NORMAL"

    def _is_opponent_trapper(self):
        """v7.3: Detect if opponent tends to trap (check-raise, slowplay)"""
        if self.rounds_played < 10:
            return False
        # High check-raise frequency = trapper
        if self.opp_flop_raises_faced > 0:
            check_raise_rate = self.opp_check_raises / self.opp_flop_raises_faced
            if check_raise_rate > 0.30:
                return True
        return False

    def _is_elite_opponent(self):
        if self.rounds_played < 5:
            return False

        vpip_rate = self.opp_vpip / self.rounds_played
        agg_rate = self.opp_postflop_aggression / max(1, self.opp_postflop_actions)

        is_balanced = 0.35 < agg_rate < 0.65
        is_active = vpip_rate > 0.40

        return is_balanced and is_active

    # Check if opponent is a NIT (folds blinds too much)
    def _is_nit_opponent(self):
        if self.opp_blind_opportunities < 8:
            return False
        fold_rate = self.opp_blind_folds / self.opp_blind_opportunities
        return fold_rate > 0.70

    # Check if opponent is a "fish" (VPIP > 60%) for 3-Bet Bullseye
    def _is_fish_opponent(self):
        if self.rounds_played < 5:
            return False
        vpip_rate = self.opp_vpip / self.rounds_played
        return vpip_rate > 0.60

    def _get_shove_rate(self):
        if self.rounds_played < 3:
            return 0.0
        return self.opp_preflop_shoves / self.rounds_played

    # get combined aggro raise rate (shoves + big raises)
    def _get_aggro_raise_rate(self):
        if self.rounds_played < 3:
            return 0.0
        return (self.opp_preflop_shoves + self.opp_big_raises) / self.rounds_played

    def _get_fold_to_raise_rate(self):
        if self.opp_raises_faced < 5:
            return 0.5
        return self.opp_fold_to_raise / self.opp_raises_faced

    def _get_gear(self, game_state):
        round_num = game_state.round_num
        bankroll = game_state.bankroll

        #adjust playing style based on state of game
        if self.consecutive_losses >= 8:
            return "CONSERVATIVE"

        if self.consecutive_losses >= 5:
            return "AGGRESSIVE"

        if round_num > 500:
            if bankroll < -150:
                return "DESPERATE"
            elif bankroll < -50:
                return "AGGRESSIVE"
            elif bankroll > 150:
                return "CONSERVATIVE"
            elif bankroll > 50:
                return "SLIGHTLY_TIGHT"

        if len(self.recent_results) >= 10:
            recent_sum = sum(self.recent_results[-10:])
            if recent_sum < -80:
                return "AGGRESSIVE"
            elif recent_sum > 80:
                return "SLIGHTLY_TIGHT"

        return "NORMAL"

    def _get_in_position(self, active, street):
        if street == 0:
            return active == 1
        else:
            return active == 0

    def _get_opponent_hole_count(self, street):
        if street == 0:
            return 3
        else:
            return 2

    def _pot_size(self, round_state):
        return 2 * STARTING_STACK - round_state.stacks[0] - round_state.stacks[1]

    # evaluate hand
    def _evaluate_5card_hand(self, five_cards):
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
                return (HAND_ROYAL_FLUSH, 14)
            return (HAND_STRAIGHT_FLUSH, straight_high)
        if counts == [4, 1]:
            quad_rank = [r for r in rank_counts if rank_counts[r] == 4][0]
            kicker = [r for r in rank_counts if rank_counts[r] == 1][0]
            return (HAND_QUADS, quad_rank * 100 + kicker)
        if counts == [3, 2]:
            trips = [r for r in rank_counts if rank_counts[r] == 3][0]
            pair = [r for r in rank_counts if rank_counts[r] == 2][0]
            return (HAND_FULL_HOUSE, trips * 100 + pair)
        if is_flush:
            return (HAND_FLUSH, max(ranks))
        if is_straight:
            return (HAND_STRAIGHT, straight_high)
        if counts == [3, 1, 1]:
            trips = [r for r in rank_counts if rank_counts[r] == 3][0]
            kickers = sorted([r for r in rank_counts if rank_counts[r] == 1], reverse=True)
            return (HAND_TRIPS, trips * 10000 + kickers[0] * 100 + kickers[1])
        if counts == [2, 2, 1]:
            pairs = sorted([r for r in rank_counts if rank_counts[r] == 2], reverse=True)
            kicker = [r for r in rank_counts if rank_counts[r] == 1][0]
            return (HAND_TWO_PAIR, pairs[0] * 10000 + pairs[1] * 100 + kicker)
        if counts == [2, 1, 1, 1]:
            pair = [r for r in rank_counts if rank_counts[r] == 2][0]
            kickers = sorted([r for r in rank_counts if rank_counts[r] == 1], reverse=True)
            return (HAND_PAIR, pair * 1000000 + kickers[0] * 10000 + kickers[1] * 100 + kickers[2])

        sorted_desc = sorted(ranks, reverse=True)
        tiebreaker = sum(r * (100 ** (4-i)) for i, r in enumerate(sorted_desc[:5]))
        return (HAND_HIGH_CARD, tiebreaker)

    def _best_hand(self, hole_cards, board):
        all_cards = list(hole_cards) + list(board)
        if len(all_cards) < 5:
            return (0, 0)

        best = (0, 0)
        for five in combinations(all_cards, 5):
            score = self._evaluate_5card_hand(list(five))
            if score > best:
                best = score
        return best

    def _fast_score(self, hole_cards, board_cards):
        if HAVE_PKRBOT:
            try:
                all_cards = list(hole_cards) + list(board_cards)
                return pkrbot.evaluate([pkrbot.Card(c) for c in all_cards])
            except:
                pass
        return self._best_hand(hole_cards, board_cards)

    # calculate equity while being time-aware
    def _calculate_equity(self, my_cards, board, street, num_sims=500):
        if not my_cards:
            return 0.5

        if HAVE_CPP_EQUITY and num_sims > 0:
            try:
                equity = calculate_equity_fast(list(my_cards), list(board), street, num_sims)
                if equity is not None:
                    return equity
            except Exception:
                pass

        # if C++ no work, fall back to Python capped at 50 simulations
        actual_sims = min(num_sims, 50)
        wins = 0.0
        known = set(my_cards) | set(board)
        remaining = [c for c in self.deck if c not in known]

        cards_needed = max(0, TARGET_BOARD_SIZE - len(board))
        opp_holes = self._get_opponent_hole_count(street)

        if len(remaining) < opp_holes + cards_needed:
            return 0.5

        for _ in range(actual_sims):
            sample = random.sample(remaining, opp_holes + cards_needed)
            opp_cards = sample[:opp_holes]
            future_board = sample[opp_holes:] if cards_needed > 0 else []
            full_board = list(board) + future_board

            my_hand = self._fast_score(my_cards, full_board)
            opp_hand = self._fast_score(opp_cards, full_board)

            if my_hand > opp_hand:
                wins += 1
            elif my_hand == opp_hand:
                wins += 0.5

        return wins / actual_sims if actual_sims > 0 else 0.5

    def _quick_hand_strength(self, my_cards, board_cards):
        if not my_cards:
            return 0.5

        all_cards = list(my_cards) + list(board_cards)
        if len(all_cards) < 5:
            ranks = sorted([self.rank_values[c[0]] for c in my_cards], reverse=True)
            if len(ranks) >= 2 and ranks[0] == ranks[1]:
                return 0.55 + (ranks[0] / 14) * 0.15
            return 0.35 + (ranks[0] / 14) * 0.20

        hand_score = self._best_hand(my_cards, board_cards)
        hand_type = hand_score[0]

        equity_map = {
            HAND_ROYAL_FLUSH: 0.99, HAND_STRAIGHT_FLUSH: 0.98,
            HAND_QUADS: 0.95, HAND_FULL_HOUSE: 0.90, HAND_FLUSH: 0.82,
            HAND_STRAIGHT: 0.75, HAND_TRIPS: 0.65, HAND_TWO_PAIR: 0.55,
            HAND_PAIR: 0.45, HAND_HIGH_CARD: 0.30, 0: 0.25,
        }
        return min(0.95, equity_map.get(hand_type, 0.40))

    def _calculate_draw_equity(self, my_cards, board, street):
        if len(board) >= TARGET_BOARD_SIZE:
            return 0.0

        all_cards = list(my_cards) + list(board)
        if len(all_cards) < 4:
            return 0.0

        suits = [c[1] for c in all_cards]
        ranks = [self.rank_values[c[0]] for c in all_cards]
        cards_to_come = TARGET_BOARD_SIZE - len(board)
        draw_equity = 0.0

        for suit in set(suits):
            if suits.count(suit) == 4:
                draw_equity += 0.35 if cards_to_come >= 2 else 0.18
            elif suits.count(suit) == 3 and cards_to_come >= 2:
                draw_equity += 0.10

        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) >= 4:
            for i in range(len(unique_ranks) - 3):
                gap = unique_ranks[i+3] - unique_ranks[i]
                if gap == 3:
                    draw_equity += 0.30 if cards_to_come >= 2 else 0.15
                    break
                elif gap == 4:
                    draw_equity += 0.15 if cards_to_come >= 2 else 0.08
                    break

        return min(draw_equity, 0.40)

    # analyze board
    def _count_flush_on_board(self, board):
        if len(board) < 3:
            return 0
        suits = [c[1] for c in board]
        return max(suits.count(s) for s in set(suits))

    def _has_straight_threat(self, board):
        if len(board) < 3:
            return False
        ranks = sorted(set(self.rank_values[c[0]] for c in board))

        # check 4 cards within range of 4
        for i in range(len(ranks) - 3):
            if ranks[i+3] - ranks[i] <= 4:
                return True

        # Check for 3 cards that could make a straight with 2 hole cards e.g., 5-7-9 can become 5-6-7-8-9 with 6-8
        for i in range(len(ranks) - 2):
            # if 3 board cards span 4 ranks (like 5-7-9 spans 5 to 9 = 4)
            if ranks[i+2] - ranks[i] <= 4:
                return True

        return False

    def _is_paired_board(self, board):
        if len(board) < 2:
            return False
        ranks = [c[0] for c in board]
        return len(ranks) != len(set(ranks))

    def _board_texture(self, board):
        if len(board) < 3:
            return "DRY"

        flush_count = self._count_flush_on_board(board)
        wet = 0
        if flush_count >= 4:
            wet += 4
        elif flush_count >= 3:
            wet += 2
        if self._has_straight_threat(board):
            wet += 2
        if self._is_paired_board(board):
            wet -= 1

        if wet >= 2:
            return "WET"
        elif wet <= 0:
            return "DRY"
        return "NEUTRAL"

    def _have_flush_blocker(self, my_cards, board):
        if len(board) < 3:
            return False
        board_suits = [c[1] for c in board]
        for suit in set(board_suits):
            if board_suits.count(suit) >= 3:
                for card in my_cards:
                    if card[1] == suit and self.rank_values[card[0]] >= 10:
                        return True
        return False

    def _have_straight_blocker(self, my_cards, board):
        if len(board) < 4:
            return False
        board_ranks = set(self.rank_values[c[0]] for c in board)
        my_ranks = set(self.rank_values[c[0]] for c in my_cards)
        sorted_board = sorted(board_ranks)
        for i in range(len(sorted_board) - 2):
            window = sorted_board[i:i+3]
            if max(window) - min(window) <= 4:
                needed = set(range(min(window)-1, max(window)+2)) - board_ranks
                if needed & my_ranks:
                    return True
        return False

    def _is_scary_board(self, board):
        if len(board) < 3:
            return False

        # check for flush threat (3+ of same suit)
        suits = [c[1] for c in board]
        for s in ['h', 'd', 'c', 's']:
            if suits.count(s) >= 3:
                return True

        # Use improved straight threat detection
        if self._has_straight_threat(board):
            return True

        return False

    def _we_have_scary_hand(self, my_cards, board):
        hand = self._best_hand(my_cards, board)
        return hand[0] >= HAND_STRAIGHT

    # detect nut blocker
    def _has_nut_blocker(self, my_cards, board):
        """Check if we hold A or K of a suit that makes a flush possible on board."""
        if len(board) < 3:
            return False

        board_suits = [c[1] for c in board]

        # Find suits with 3+ cards on board (flush possible)
        for suit in ['h', 'd', 'c', 's']:
            if board_suits.count(suit) >= 3:
                # Check if we hold A or K of this suit
                for card in my_cards:
                    if card[1] == suit and self.rank_values[card[0]] >= 13:  # K or A
                        return True
        return False

    # Check if opponent is over-aggressive (for counter-exploit)
    def _is_opponent_over_aggressive(self):
        """Returns True if opponent is betting >70% of the time (likely over-bluffing)."""
        total_actions = self.opp_recent_bets + self.opp_recent_checks
        if total_actions < 10:
            return False
        return (self.opp_recent_bets / total_actions) > 0.70

    # check if have top pair or not
    def _has_top_pair_or_better(self, my_cards, board):
        """Check if we have top pair or better for check-raise defense."""
        if len(board) < 3:
            return False

        hand = self._best_hand(my_cards, board)
        hand_type = hand[0]

        # Two pair or better is always good
        if hand_type >= HAND_TWO_PAIR:
            return True

        # Check if we have top pair or overpair
        if hand_type == HAND_PAIR:
            board_ranks = sorted([self.rank_values[c[0]] for c in board], reverse=True)
            my_ranks = [self.rank_values[c[0]] for c in my_cards]

            # Check for pocket pair (overpair)
            if len(my_ranks) >= 2 and my_ranks[0] == my_ranks[1]:
                # We have a pocket pair
                pair_rank = my_ranks[0]
                if pair_rank > board_ranks[0]:
                    return True

            # Find which rank makes our pair with the board
            for r in my_ranks:
                if r in board_ranks:
                    # We have a pair with the board
                    # top pair
                    if r >= board_ranks[0]:
                        return True

        return False

    # discard algorithm
    def _board_impact_discard(self, my_cards, board, street):
        """Fast heuristic discard - NO simulations"""
        if len(my_cards) != 3:
            return 0

        best_idx = 0
        best_score = float('-inf')

        for i in range(3):
            discard = my_cards[i]
            keep = [my_cards[j] for j in range(3) if j != i]

            keep_score = self._keep_score(keep, board)
            new_board = list(board) + [discard]
            danger = self._board_danger_score(discard, keep, new_board)
            realization = self._realization_score(keep, new_board)

            total = keep_score - danger + realization
            if total > best_score:
                best_score = total
                best_idx = i

        return best_idx

    def _keep_score(self, keep, board):
        if len(keep) != 2:
            return 0

        r1, r2 = self.rank_values[keep[0][0]], self.rank_values[keep[1][0]]
        s1, s2 = keep[0][1], keep[1][1]
        score = 0.0

        if r1 == r2:
            score += 6.0 + (r1 / 14.0) * 3.0

        if s1 == s2:
            score += 3.0
            if board:
                board_suits = [c[1] for c in board]
                if s1 in board_suits:
                    score += 4.0 if board_suits.count(s1) >= 2 else 2.0
            if r1 >= 10 or r2 >= 10:
                score += 1.0

        gap = abs(r1 - r2)
        if gap == 1:
            score += 1.8
        elif gap == 2:
            score += 1.0
        elif gap == 3:
            score += 0.4

        score += (r1 + r2) / 16.0

        if board:
            board_ranks = [self.rank_values[c[0]] for c in board]
            if r1 in board_ranks:
                score += 2.5 + (r1 / 14.0)
            if r2 in board_ranks:
                score += 2.5 + (r2 / 14.0)

        return score

    def _board_danger_score(self, discard, keep, new_board):
        danger = 0.0
        card_rank = self.rank_values[discard[0]]
        card_suit = discard[1]
        keep_ranks = [self.rank_values[c[0]] for c in keep]
        keep_suits = [c[1] for c in keep]

        board_suits = [c[1] for c in new_board]
        suit_counts = {}
        for s in board_suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1

        if suit_counts.get(card_suit, 0) == 3:
            if card_suit not in keep_suits:
                danger += 5.0
                if card_rank >= 10:
                    danger += 2.0
            else:
                high = any(c[1] == card_suit and self.rank_values[c[0]] >= 11 for c in keep)
                danger += 0.5 if high else 2.5

        if suit_counts.get(card_suit, 0) >= 4:
            danger += 6.0 if card_suit not in keep_suits else -1.0

        board_ranks = sorted([self.rank_values[c[0]] for c in new_board])
        for i in range(len(board_ranks) - 2):
            if i + 2 < len(board_ranks) and board_ranks[i+2] - board_ranks[i] <= 4:
                needed_ranks = set(range(board_ranks[i]-1, board_ranks[i+2]+2))
                if card_rank in needed_ranks:
                    if card_rank not in keep_ranks:
                        danger += 3.0
                blocking = any(r in keep_ranks for r in needed_ranks)
                if not blocking:
                    danger += 1.5
                break

        rank_counts = {}
        for c in new_board:
            r = self.rank_values[c[0]]
            rank_counts[r] = rank_counts.get(r, 0) + 1

        if rank_counts.get(card_rank, 0) >= 2:
            danger += 1.8 if card_rank not in keep_ranks else -2.5

        if card_rank >= 12:
            danger += 0.4

        return danger

    def _realization_score(self, keep, new_board):
        bonus = 0.0
        keep_suits = [c[1] for c in keep]
        keep_ranks = [self.rank_values[c[0]] for c in keep]
        board_suits = [c[1] for c in new_board]

        for suit in set(board_suits):
            if board_suits.count(suit) >= 3 and suit in keep_suits:
                bonus += 1.5
                for c in keep:
                    if c[1] == suit and self.rank_values[c[0]] >= 11:
                        bonus += 0.8

        board_ranks = [self.rank_values[c[0]] for c in new_board]
        for r in set(board_ranks):
            if board_ranks.count(r) >= 2 and r in keep_ranks:
                bonus += 2.0

        return bonus

    # time-aware discarding
    def _smart_toss(self, my_cards, board, street, time_remaining):
        if len(my_cards) != 3:
            return 0

        my_ranks = [c[0] for c in my_cards]
        unique_ranks = set(my_ranks)
        board_ranks = [c[0] for c in board] if board else []

        # Handle Three-of-a-Kind in hand
        if len(unique_ranks) == 1:
            return 0

        # check board matches first
        # If a card matches the board, DON'T discard it!
        # Trips > Pair - even if we have pocket pair, keep the board match
        cards_matching_board = []
        for i, card in enumerate(my_cards):
            if card[0] in board_ranks:
                cards_matching_board.append(i)

        # If we have a card that matches the board, keep it and discard something else
        if cards_matching_board:
            # Find ANY card that doesn't match the board to discard
            for i, card in enumerate(my_cards):
                if i not in cards_matching_board:
                    return i  # Discard this card (even if it's part of a pair)
            # All cards match board? Just keep first one
            return 0

        # Find pairs in hand
        pairs = [r for r in unique_ranks if my_ranks.count(r) > 1]

        # If we have a pair, find the card NOT in the pair
        if pairs:
            for i, card in enumerate(my_cards):
                if card[0] not in pairs:
                    return i
            return 0

        # Simulation scaling
        if time_remaining <= 5:
            return self._board_impact_discard(my_cards, board, street)
        elif time_remaining <= 10:
            mc_sims = 200
        elif time_remaining <= 20:
            mc_sims = 400
        else:
            mc_sims = 800

        best_idx = 0
        best_score = float('-inf')

        for i in range(len(my_cards)):
            keep = [my_cards[j] for j in range(len(my_cards)) if j != i]
            discard = my_cards[i]
            new_board = list(board) + [discard]

            equity = self._calculate_equity(keep, new_board, street, num_sims=mc_sims)
            keep_score = self._keep_score(keep, board)
            danger = self._board_danger_score(discard, keep, new_board)
            realization = self._realization_score(keep, new_board)

            combined = equity * 10 + keep_score - danger * 0.5 + realization * 0.5
            if combined > best_score:
                best_score = combined
                best_idx = i

        return best_idx

    # logic for preflop
    def _get_preflop_tier(self, cards):
        if len(cards) < 2:
            return 4

        if len(cards) == 2:
            return self._eval_2card_tier(cards[0], cards[1])

        best = 4
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                tier = self._eval_2card_tier(cards[i], cards[j])
                if tier < best:
                    best = tier
        return best

    def _eval_2card_tier(self, c1, c2):
        r1, r2 = self.rank_values[c1[0]], self.rank_values[c2[0]]
        s1, s2 = c1[1], c2[1]

        if r1 < r2:
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        r1c = self.rank_chars.get(r1, '?')
        r2c = self.rank_chars.get(r2, '?')

        if r1 == r2:
            key = r1c + r1c
        else:
            key = r1c + r2c + ('s' if s1 == s2 else 'o')

        return PREFLOP_TIERS.get(key, 4)

    def _evaluate_preflop_hand(self, my_cards):
        tier = self._get_preflop_tier(my_cards)
        base = {1: 9, 2: 7, 3: 5, 4: 2}.get(tier, 2)

        if len(my_cards) == 3:
            ranks = sorted([self.rank_values[c[0]] for c in my_cards], reverse=True)
            suits = [c[1] for c in my_cards]

            if ranks[0] == ranks[1] == ranks[2]:
                return 10
            if ranks[0] == ranks[1] or ranks[1] == ranks[2]:
                base += 1
            if suits[0] == suits[1] == suits[2]:
                base += 1
            if ranks[0] - ranks[2] == 2:
                base += 0.5

        return min(10, base)

    # all-in ranges (dynamic)
    def _is_premium_for_allin(self, my_cards, continue_cost):
        """
        TIGHTER ALL-IN CALLING
        Key insight: Bots that shove 400 chips usually have real hands.
        Even if they raise small frequently, their ALL-INS are selective.
        """
        aggro_rate = self._get_aggro_raise_rate()
        shove_rate = self._get_shove_rate()

        ranks = [self.rank_values[c[0]] for c in my_cards]
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        has_pair = max(rank_counts.values()) >= 2
        pair_rank = max([r for r, c in rank_counts.items() if c >= 2], default=0)
        has_ace = 14 in ranks
        has_king = 13 in ranks
        high_card = max(ranks)
        second_highest = sorted(ranks, reverse=True)[1] if len(ranks) > 1 else 0

        # For TRUE ALL-INS (350+ chips), be MUCH tighter regardless of aggro rate
        if continue_cost >= 350:
            # Only call all-ins with premium hands
            if has_pair and pair_rank >= 10:  # TT+
                return True
            if has_ace and second_highest >= 12:  # AQ+
                return True
            return False

        # Very tight opponent or early in match (shove rate < 5%)
        # Only call with premium: QQ+, AK
        if aggro_rate < 0.05 or self.rounds_played < 10:
            if has_pair and pair_rank >= 12:  # QQ+
                return True
            if has_ace and second_highest >= 13:  # AK
                return True
            return False

        # Moderate aggression (5-15% aggro rate)
        # Call with: 99+, AQ+, KQs
        elif aggro_rate < 0.15:
            if has_pair and pair_rank >= 9:  # 99+
                return True
            if has_ace and second_highest >= 12:  # AQ+
                return True
            if has_king and second_highest >= 12:  # KQ
                return True
            return False

        # High aggression (15-25% aggro rate)
        # Call with: 88+, AJ+, KQ+ (tightened from 77+, AT+)
        elif aggro_rate < 0.25:
            if has_pair and pair_rank >= 8:  # 88+
                return True
            if has_ace and second_highest >= 11:  # AJ+
                return True
            if has_king and second_highest >= 12:  # KQ+
                return True
            return False

        # MANIAC MODE (>25% aggro rate) - They're spamming raises
        # Call with: 77+, AT+, KJ+ (tightened from 55+, A8+)
        else:
            if has_pair and pair_rank >= 7:  # 77+
                return True
            if has_ace and second_highest >= 10:  # AT+
                return True
            if has_king and second_highest >= 11:  # KJ+
                return True
            return False

    # helpers for betting
    def _is_big_pressure(self, continue_cost, pot, my_stack, street):
        if pot <= 0:
            return continue_cost > 8

        pot_ratio = continue_cost / pot
        stack_ratio = continue_cost / my_stack if my_stack > 0 else 1

        if street >= 5:
            return pot_ratio > 0.45 or stack_ratio > 0.10
        return pot_ratio > 0.55 or stack_ratio > 0.15

    def _is_allin_or_near(self, continue_cost, my_stack):
        return continue_cost >= my_stack * 0.8

    def _should_bluff_river(self, board, opp_type, in_position, my_cards):
        texture = self._board_texture(board)
        if texture == "DRY":
            return False
        if opp_type in ("CALLING_STATION", "LOOSE_PASSIVE"):
            return False

        fold_rate = self._get_fold_to_raise_rate()
        if opp_type in ("VERY_TIGHT", "TIGHT") and fold_rate > 0.4:
            return True

        if texture == "WET":
            has_blocker = self._have_flush_blocker(my_cards, board) or self._have_straight_blocker(my_cards, board)
            if has_blocker:
                return True
            if in_position and fold_rate > 0.35:
                return True
        return False

    def _get_bet_size(self, equity, pot, street, opp_type, is_bluff=False):
        if USE_DP_TABLES:
            hs = categorize_hand_strength(equity)
            pc = categorize_pot(pot)
            sn = categorize_street(street)
            frac = get_bluff_size(pc, sn) if is_bluff else get_bet_size(hs, pc, sn)

            if opp_type in ("CALLING_STATION", "LOOSE_PASSIVE"):
                frac = frac * 1.15 if not is_bluff else 0
            elif opp_type in ("VERY_TIGHT", "TIGHT"):
                frac = frac * 1.1 if is_bluff else frac * 0.85
            elif opp_type == "MANIAC":
                frac *= 0.9

            return int(pot * frac)

        if is_bluff:
            if opp_type in ("CALLING_STATION", "LOOSE_PASSIVE"):
                return 0
            return int(pot * 0.75)
        if equity >= 0.70:
            return int(pot * random.uniform(0.70, 0.85))
        elif equity >= 0.55:
            return int(pot * random.uniform(0.50, 0.65))
        elif equity >= 0.40 and random.random() < 0.35:
            return int(pot * 0.45)
        return 0

    def _adjust_equity_for_texture(self, equity, board, my_cards):
        adj = equity
        flush_count = self._count_flush_on_board(board)

        if flush_count >= 4:
            adj *= 0.70 if self._have_flush_blocker(my_cards, board) else 0.45
        elif flush_count >= 3:
            adj *= 0.85 if self._have_flush_blocker(my_cards, board) else 0.72

        if self._has_straight_threat(board):
            adj *= 0.88 if self._have_straight_blocker(my_cards, board) else 0.78

        if self._is_paired_board(board):
            hand = self._best_hand(my_cards, board)
            if hand[0] <= HAND_PAIR:
                adj *= 0.90

        return adj

    def _get_mdf(self, bet, pot):
        if pot + bet <= 0:
            return 0.5
        return pot / (pot + bet)

    def _do_raise(self, amount):
        self.my_raises_this_round += 1
        self.raise_count_this_street += 1
        self.we_bet_this_street = True
        return RaiseAction(amount)

    # action
    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board = round_state.board

        if self._last_street != street:
            self.raise_count_this_street = 0
            self.we_bet_this_street = False
            self._last_street = street

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1-active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]

        continue_cost = opp_pip - my_pip
        pot = self._pot_size(round_state)

        time_remaining = game_state.game_clock
        in_position = self._get_in_position(active, street)
        opp_type = self._get_opponent_type()
        gear = self._get_gear(game_state)

        is_elite = self._is_elite_opponent()

        # lock in for lead
        my_bankroll = game_state.bankroll
        rounds_left = NUM_ROUNDS - game_state.round_num
        lead_lock_threshold = 250 if is_elite else 400
        is_lead_lock = (my_bankroll > lead_lock_threshold and rounds_left < 150)

        # discard action
        if DiscardAction in legal:
            return DiscardAction(self._smart_toss(my_cards, board, street, time_remaining))

        # simulations for time-aware betting
        if time_remaining <= 5:
            sims = 100
        elif time_remaining <= 8:
            sims = 300
        elif time_remaining <= 15:
            sims = 800
        else:
            sims = 2000

        # preflop
        if street == 0:
            tier = self._get_preflop_tier(my_cards)
            strength = self._evaluate_preflop_hand(my_cards)
            preflop_equity = strength / 10.0

            is_nit = self._is_nit_opponent()
            is_fish = self._is_fish_opponent()

            # Pocket pair detection
            is_pair = (my_cards[0][0] == my_cards[1][0] or
                       my_cards[0][0] == my_cards[2][0] or
                       my_cards[1][0] == my_cards[2][0])

            high_card = max(self.rank_values[c[0]] for c in my_cards)

            # Suited detection
            suits = [c[1] for c in my_cards]
            is_suited = suits[0] == suits[1] or suits[0] == suits[2] or suits[1] == suits[2]

            # defense
            # Don't let opponents "farm" our blinds with small raises
            # Defend with any playable hand
            is_big_blind = (active == 1)  # Player B is big blind
            if is_big_blind and continue_cost > 0 and continue_cost <= 10:
                # Defend BB against small raises
                if is_pair:  # Any pair
                    return CallAction() if CallAction in legal else CheckAction()
                if is_suited:  # Any suited
                    return CallAction() if CallAction in legal else CheckAction()
                if high_card >= 11:  # J+ high card
                    return CallAction() if CallAction in legal else CheckAction()

            # Tightened to stop value-donating to adapted pools
            # Only call with hands that have real equity vs value hands
            # If opponent is raising constantly, WIDEN our calling range
            if continue_cost >= 60 and continue_cost <= 300:
                high_card = max(self.rank_values[c[0]] for c in my_cards)
                second_card = sorted([self.rank_values[c[0]] for c in my_cards], reverse=True)[1]

                # Get pair rank if we have one
                my_ranks = [c[0] for c in my_cards]
                pair_rank = 0
                for r in set(my_ranks):
                    if my_ranks.count(r) >= 2:
                        pair_rank = max(pair_rank, self.rank_values[r])

                should_call = False

                # Detect constant aggression - if they raise >25% of hands, they're a bully
                is_constant_bully = False
                if self.rounds_played >= 5:
                    aggro_rate = (self.opp_preflop_shoves + self.opp_big_raises) / self.rounds_played
                    if aggro_rate > 0.25:
                        is_constant_bully = True

                if is_constant_bully or opp_type in ("MANIAC", "SHOVE_BOT", "LOOSE_AGGRESSIVE"):
                    # They're raising constantly - call with top 40% to punish
                    if is_pair: should_call = True                              # Any pair 22+
                    elif high_card >= 14: should_call = True                    # Any Ace (A2+)
                    elif high_card >= 13 and second_card >= 5: should_call = True    # K5+
                    elif high_card >= 12 and second_card >= 9: should_call = True    # Q9+
                    elif high_card >= 11 and second_card >= 9: should_call = True    # J9+
                elif opp_type == "VERY_TIGHT":
                    if is_pair and pair_rank >= 9: should_call = True           # 99+
                    elif high_card >= 14 and second_card >= 11: should_call = True  # AJ+
                    elif high_card >= 13 and second_card >= 12: should_call = True  # KQ+
                else:
                    if is_pair and pair_rank >= 5: should_call = True           # 55+
                    elif high_card >= 14 and second_card >= 7: should_call = True    # A7+
                    elif high_card >= 13 and second_card >= 10: should_call = True   # KT+
                    elif high_card >= 12 and second_card >= 10: should_call = True   # QT+

                if should_call:
                    return CallAction() if CallAction in legal else CheckAction()
                else:
                    return FoldAction() if FoldAction in legal else CheckAction()

            # big raise defensee
            if continue_cost >= BIG_RAISE_THRESHOLD:
                # They're making a huge raise (150+ chips)
                # Use dynamic calling range based on how often they do this
                if self._is_premium_for_allin(my_cards, continue_cost):
                    return CallAction() if CallAction in legal else CheckAction()
                else:
                    return FoldAction() if FoldAction in legal else CheckAction()


            # Re-raise passive/fishy bots to defend rank
            if is_fish and continue_cost > 0 and continue_cost <= 20 and tier <= 2:
                if RaiseAction in legal:
                    min_r, max_r = round_state.raise_bounds()
                    # 3-bet to punish their wide opens
                    threeb_size = min(max_r, opp_pip + int(pot * 0.8))
                    self.was_preflop_aggressor = True
                    return self._do_raise(max(min_r, threeb_size))

            # adaptive small blind defense
            is_small_blind = (active == 0 and my_pip == SMALL_BLIND)
            if is_small_blind and continue_cost <= 6:
                pot_odds = continue_cost / (pot + continue_cost) if (pot + continue_cost) > 0 else 0.5

                if is_elite:
                    if pot_odds < 0.35:
                        if tier <= 4:
                            return CallAction() if CallAction in legal else CheckAction()
                        elif tier == 5 or strength >= 3:
                            if random.random() < 0.60:
                                return CallAction() if CallAction in legal else CheckAction()
                else:
                    if pot_odds < 0.20:
                        if tier <= 3:
                            return CallAction() if CallAction in legal else CheckAction()
                    if gear == "DESPERATE" and pot_odds < 0.25 and tier == 4:
                        return CallAction() if CallAction in legal else CheckAction()

            # Never fold pairs vs Maniacs
            if opp_type in ("MANIAC", "SHOVE_BOT") and is_pair and continue_cost <= 40:
                return CallAction() if CallAction in legal else CheckAction()

            if continue_cost >= 20 and continue_cost < 50:
                if is_elite:
                    if high_card >= 14 or is_pair:
                        if my_bankroll > -200:
                            return CallAction() if CallAction in legal else CheckAction()
                else:
                    if opp_type in ("MANIAC", "UNKNOWN", "LOOSE_AGGRESSIVE", "SHOVE_BOT"):
                        if high_card >= 13 or is_pair:
                            if my_bankroll > -200:
                                return CallAction() if CallAction in legal else CheckAction()

            if is_pair and continue_cost <= 20:
                if continue_cost == 0:
                    if RaiseAction in legal and tier <= 2:
                        min_r, max_r = round_state.raise_bounds()
                        self.was_preflop_aggressor = True
                        return self._do_raise(min_r + random.randint(0, min(6, max_r - min_r)))
                    return CheckAction() if CheckAction in legal else CallAction()
                else:
                    return CallAction() if CallAction in legal else CheckAction()

            # NIT CRACKER
            is_button = (active == 0)
            if is_button and is_nit and RaiseAction in legal:
                min_r, max_r = round_state.raise_bounds()
                self.was_preflop_aggressor = True
                return self._do_raise(min_r + 2)

            # Button aggression
            if is_button and my_pip == SMALL_BLIND and continue_cost <= BIG_BLIND:
                button_freq = 0.25
                if gear in ("AGGRESSIVE", "DESPERATE"):
                    button_freq = 0.40

                if 0.50 < preflop_equity < 0.75:
                    if random.random() < button_freq:
                        if RaiseAction in legal:
                            min_r, max_r = round_state.raise_bounds()
                            raise_amt = max(min_r, min(max_r, my_pip + 3 * BIG_BLIND))
                            self.was_preflop_aggressor = True
                            return self._do_raise(raise_amt)

            # Facing standard raise
            if continue_cost > BIG_BLIND:
                # adaptive
                # - TIGHT vs elite opponents (they have real hands)
                # - LOOSE vs fish/nit opponents (exploit their mistakes)
                is_elite = self._is_elite_opponent()
                is_fish = self._is_fish_opponent()
                is_nit = self._is_nit_opponent()

                if is_elite:
                    #They only play premium, so should we
                    fold_thresh = 7
                elif is_fish:
                    # They play trash, call wider to exploit
                    fold_thresh = 4
                elif is_nit:
                    # They fold a lot - we can call wider, they're tight
                    fold_thresh = 5
                else:
                    # Unknown/normal - default moderate
                    fold_thresh = 5

                if is_lead_lock:
                    fold_thresh += 1

                if strength < fold_thresh:
                    return FoldAction() if FoldAction in legal else CheckAction()
                if continue_cost > pot * 0.4 and strength < fold_thresh + 1:
                    return FoldAction() if FoldAction in legal else CheckAction()

                if tier <= 2 and RaiseAction in legal and random.random() < 0.4:
                    min_r, max_r = round_state.raise_bounds()
                    self.was_preflop_aggressor = True
                    return self._do_raise(min_r + random.randint(0, min(8, max_r - min_r)))
                return CallAction() if CallAction in legal else CheckAction()

            # Unopened pot - also adaptive
            is_elite = self._is_elite_opponent()
            is_fish = self._is_fish_opponent()
            is_nit = self._is_nit_opponent()

            if tier <= 2:
                # Always raise premium
                if RaiseAction in legal:
                    min_r, max_r = round_state.raise_bounds()
                    self.was_preflop_aggressor = True
                    return self._do_raise(min_r + random.randint(0, min(4, max_r - min_r)))
            if tier == 3:
                if is_fish or is_nit:
                    # Raise vs weak opponents
                    if RaiseAction in legal and random.random() < 0.6:
                        min_r, max_r = round_state.raise_bounds()
                        self.was_preflop_aggressor = True
                        return self._do_raise(min_r + random.randint(0, min(2, max_r - min_r)))
                # Just call vs elite/unknown
                return CallAction() if CallAction in legal else CheckAction()
            if tier >= 4:
                if continue_cost == 0:
                    if (is_fish or is_nit) and random.random() < 0.3:
                        # Steal attempt vs weak players
                        if RaiseAction in legal:
                            min_r, max_r = round_state.raise_bounds()
                            return self._do_raise(min_r)
                    return CheckAction()
                return FoldAction() if FoldAction in legal else CheckAction()

        # postflop
        equity = self._calculate_equity(my_cards, board, street, num_sims=sims)
        draw_eq = self._calculate_draw_equity(my_cards, board, street)
        equity = min(0.98, equity + draw_eq * 0.5)

        hand = self._best_hand(my_cards, board)
        hand_type = hand[0]

        bet_ratio = continue_cost / pot if pot > 0 else 0
        pot_odds = continue_cost / (pot + continue_cost) if (pot + continue_cost) > 0 else 0.5

        # In position (acting last) = can be more aggressive
        # Out of position (acting first) = play tighter
        # In this game, active=0 means Player A (small blind), active=1 means Player B (big blind)
        # Player B acts last postflop = in position
        in_position = (active == 1)
        position_buffer = -0.04 if in_position else 0.02  # Looser in position, tighter OOP

        big_pressure = self._is_big_pressure(continue_cost, pot, my_stack, street)
        is_allin = self._is_allin_or_near(continue_cost, my_stack)
        is_scary_board = self._is_scary_board(board)
        we_have_nuts = self._we_have_scary_hand(my_cards, board)
        is_strong_hand = (equity >= 0.60 and hand_type >= HAND_PAIR)

        # facing bet now
        if continue_cost > 0:
            adj_equity = self._adjust_equity_for_texture(equity, board, my_cards)

            if hand_type < HAND_PAIR:
                # We have HIGH CARD only (no pair, no draw completing)
                # Only call if:
                # 1. It's a tiny bet (<10 chips) AND we have overcards
                # 2. It's a tiny bet AND pot odds are insane (<20%)
                postflop_high = max(self.rank_values[c[0]] for c in my_cards)

                if continue_cost >= 20:
                    # Significant bet with no pair = ALWAYS FOLD
                    return FoldAction() if FoldAction in legal else CheckAction()
                elif continue_cost >= 10:
                    # Medium bet - only call with Ace high
                    if postflop_high < 14:  # Not Ace high
                        return FoldAction() if FoldAction in legal else CheckAction()
                else:
                    # Tiny bet (<10) - only call with good high card and pot odds
                    pot_odds_here = continue_cost / (pot + continue_cost) if (pot + continue_cost) > 0 else 1
                    if pot_odds_here > 0.20 or postflop_high < 11:  # Not J+ high
                        return FoldAction() if FoldAction in legal else CheckAction()

            if hand_type < HAND_PAIR and RaiseAction in legal:
                # Can't raise with nothing - at best we can call small bets
                pass  # Don't raise, let it fall through to fold/call logic

            # On river with big bets, need REAL hands
            if street == 6 and continue_cost > pot * 0.5:
                # Big river bet - need better than just a pair on dangerous boards
                board_suits = [c[1] for c in board]
                has_4_flush = any(board_suits.count(s) >= 4 for s in ['h','d','c','s'])

                if has_4_flush and hand_type < HAND_FLUSH:
                    # 4-flush board and we don't have a flush = FOLD
                    return FoldAction() if FoldAction in legal else CheckAction()

                if self._is_paired_board(board) and hand_type < HAND_TRIPS:
                    # Paired board + big bet = they likely have trips/full house
                    if continue_cost > pot * 0.75:
                        return FoldAction() if FoldAction in legal else CheckAction()

                if hand_type == HAND_PAIR and continue_cost > pot * 0.75:
                    # Just a pair facing big river bet - fold unless we have overpair
                    my_pair_rank = 0
                    for i, c1 in enumerate(my_cards):
                        for c2 in my_cards[i+1:]:
                            if c1[0] == c2[0]:
                                my_pair_rank = self.rank_values[c1[0]]
                    board_high = max(self.rank_values[c[0]] for c in board)
                    if my_pair_rank <= board_high:
                        # Our pair is not an overpair - fold to big bet
                        if adj_equity < 0.45:
                            return FoldAction() if FoldAction in legal else CheckAction()

            is_true_allin = continue_cost >= my_stack * 0.80
            if is_true_allin:
                # Check for premium/medium pairs
                my_ranks = sorted([self.rank_values[c[0]] for c in my_cards], reverse=True)
                has_premium_pair = (my_ranks[0] == my_ranks[1] if len(my_ranks) >= 2 else False) and my_ranks[0] >= 12  # QQ+
                has_medium_pair = (my_ranks[0] == my_ranks[1] if len(my_ranks) >= 2 else False) and my_ranks[0] >= 10  # TT+

                # On paired boards with less than trips, need high equity
                if self._is_paired_board(board) and hand_type < HAND_TRIPS:
                    threshold = 0.52 if has_premium_pair else (0.55 if has_medium_pair else 0.62)
                    if adj_equity < threshold:
                        return FoldAction() if FoldAction in legal else CheckAction()

                # On scary boards (straight/flush threats), need higher equity
                # EXCEPTION: Low boards (all cards < 9) are less scary for overpairs
                board_high = max(self.rank_values[c[0]] for c in board)
                is_low_board = board_high <= 9

                if self._is_scary_board(board) and hand_type < HAND_TRIPS:
                    if is_low_board and (has_premium_pair or has_medium_pair):
                        # Low board with overpair - use looser threshold
                        threshold = 0.45 if has_premium_pair else 0.48
                    else:
                        threshold = 0.52 if has_premium_pair else (0.55 if has_medium_pair else 0.62)
                    if adj_equity < threshold:
                        return FoldAction() if FoldAction in legal else CheckAction()

                # For any all-in without trips, need minimum equity
                if hand_type < HAND_TRIPS:
                    threshold = 0.45 if has_premium_pair else (0.48 if has_medium_pair else 0.52)
                    if adj_equity < threshold:
                        return FoldAction() if FoldAction in legal else CheckAction()

            # If we have monster on flop/turn and they bet into us,
            # RAISE instead of just calling to trap them
            # Only trigger if WE didn't bet first (true check-raise)
            if street <= 5 and equity > 0.85 and not self.we_bet_this_street:  # Flop or Turn with monster
                if RaiseAction in legal:
                    if random.random() < 0.60:  # Check-raise 60% of time with monsters
                        min_r, max_r = round_state.raise_bounds()
                        # Triple their bet to trap them
                        trap_size = my_pip + (continue_cost * 3)
                        return self._do_raise(max(min_r, min(max_r, trap_size)))

            # If we have top pair+ and opponent is aggressive, reduce buffer to near-zero
            has_top_pair = self._has_top_pair_or_better(my_cards, board)
            opp_aggro = self._get_aggro_raise_rate()
            if has_top_pair and opp_aggro > 0.20:
                # They're aggressive - call with top pair, minimal buffer
                if adj_equity >= pot_odds + 0.01:
                    return CallAction() if CallAction in legal else CheckAction()

            # Never fold >40% equity on river in large pots unless obvious danger
            if street == 6 and pot > 200 and adj_equity >= 0.40:
                # Check for 4-flush or 4-straight (obvious danger)
                board_suits = [c[1] for c in board]
                has_4_flush = any(board_suits.count(s) >= 4 for s in ['h','d','c','s'])
                board_ranks = sorted([self.rank_values[c[0]] for c in board])
                has_4_straight = False
                if len(board_ranks) >= 4:
                    for i in range(len(board_ranks) - 3):
                        if board_ranks[i+3] - board_ranks[i] <= 4:
                            has_4_straight = True
                            break

                if not has_4_flush and not has_4_straight:
                    # No obvious danger - call with 40%+ equity
                    return CallAction() if CallAction in legal else CheckAction()

            # Never fold JJ+ or monster equity to tiny probes
            # This stops the "death by thousand cuts" from small-ballers
            ranks = sorted([self.rank_values[c[0]] for c in my_cards], reverse=True)
            is_premium_pair = False
            for i, c1 in enumerate(my_cards):
                for c2 in my_cards[i+1:]:
                    if c1[0] == c2[0] and self.rank_values[c1[0]] >= 11:  # JJ+
                        is_premium_pair = True
                        break

            is_premium = is_premium_pair or equity > 0.85 or hand_type >= HAND_TWO_PAIR

            # Never fold premiums to probes < 15 chips
            if is_premium and continue_cost < 15:
                return CallAction() if CallAction in legal else CheckAction()

            # Track multi-street pressure
            if bet_ratio > 0.45:
                self.opp_streets_pressured.add(street)
                self.opp_recent_bets += 1

            # Never fold Top Pair to a single raise on the flop
            # If opponent check-raises frequently, they may be bluffing
            if street == 4 and self.we_bet_this_street and continue_cost > 0:
                # We bet, they raised
                self.opp_check_raises += 1  # Track this
                self.opp_flop_raises_faced += 1

                if self._has_top_pair_or_better(my_cards, board):
                    # We have Top Pair or better - DON'T fold to a single raise
                    if adj_equity > 0.40:
                        return CallAction() if CallAction in legal else CheckAction()

                # Even without top pair, if they check-raise a LOT, consider calling wider
                if self.opp_check_raises >= 3 and self.opp_flop_raises_faced >= 5:
                    check_raise_rate = self.opp_check_raises / self.opp_flop_raises_faced
                    if check_raise_rate > 0.50 and hand_type >= HAND_PAIR and adj_equity > 0.35:
                        # They check-raise >50% of the time - likely over-bluffing
                        return CallAction() if CallAction in legal else CheckAction()

            # Adjust equity based on opponent type
            if opp_type in ("TIGHT", "VERY_TIGHT"):
                adj_equity -= 0.07
            elif opp_type in ("MANIAC", "SHOVE_BOT"):
                adj_equity += 0.10
            elif opp_type == "LOOSE_AGGRESSIVE":
                adj_equity += 0.05

            adj_equity = max(0.0, min(1.0, adj_equity))

            if is_elite and len(self.opp_streets_pressured) >= 2:
                if hand_type < HAND_TRIPS and adj_equity < 0.78:
                    return FoldAction() if FoldAction in legal else CheckAction()

            aggression_adjustment = 0
            if is_elite and self._is_opponent_over_aggressive():
                aggression_adjustment = 0.08

            # Scary board protection
            if is_scary_board and not is_elite and not we_have_nuts:
                if adj_equity < 0.85:
                    max_call_allowed = int(pot * 0.40)
                    if continue_cost > max_call_allowed:
                        return FoldAction() if FoldAction in legal else CheckAction()

            # Lead lock protection
            if is_lead_lock and adj_equity < 0.60:
                return FoldAction() if FoldAction in legal else CheckAction()

            # defend against tiny bets
            # When opponent bets tiny amounts (2-10 chips), call very wide
            # Pot odds are so good we should call with almost anything
            if continue_cost <= 10 and continue_cost > 0:
                # Tiny bet - call with any pair or any draw potential
                if hand_type >= HAND_PAIR:
                    return CallAction() if CallAction in legal else CheckAction()
                if adj_equity >= 0.25:  # Very low threshold for tiny bets
                    return CallAction() if CallAction in legal else CheckAction()
                # Even with nothing, call if pot odds are ridiculous
                if pot > 0 and continue_cost / (pot + continue_cost) < 0.25:
                    postflop_high = max(self.rank_values[c[0]] for c in my_cards)
                    if adj_equity >= 0.18 or postflop_high >= 11:  # J+ high
                        return CallAction() if CallAction in legal else CheckAction()

            # Small bets - defend wider
            if bet_ratio < 0.35 and street >= 4:
                if adj_equity > (0.32 - aggression_adjustment) or hand_type >= HAND_PAIR:
                    return CallAction() if CallAction in legal else CheckAction()

            # If bet is <15% pot on turn/river, use 0% buffer
            # Elite bots use tiny probes to push you off best hands
            if bet_ratio < 0.15 and street >= 5:
                if hand_type >= HAND_PAIR or adj_equity >= pot_odds:
                    return CallAction() if CallAction in legal else CheckAction()

            # On flop/turn, call at least 60% with any pair
            # Elite bots exploit folders - don't fold pairs to standard bets
            if street >= 4 and street <= 5 and hand_type >= HAND_PAIR:
                if bet_ratio <= 0.67:  # Standard bet or smaller
                    if adj_equity >= 0.35:  # Reasonable equity
                        return CallAction() if CallAction in legal else CheckAction()

            # Big pressure handling
            if big_pressure or is_allin:
                # If opponent is a trapper and suddenly bets big, they likely have it
                is_trapper = opp_type == "TRAPPER" or self._is_opponent_trapper()
                if is_trapper and hand_type < HAND_TWO_PAIR:
                    # Trappers bet big with monsters - fold weak hands
                    return FoldAction() if FoldAction in legal else CheckAction()

                # Don't call all-in with just a pair on a 3+ flush board
                if is_scary_board and hand_type < HAND_FLUSH:
                    # Check if board has 3+ of same suit
                    board_suits = [c[1] for c in board]
                    for suit in ['h', 'd', 'c', 's']:
                        if board_suits.count(suit) >= 3:
                            # We don't have a flush, they might - fold unless we have trips+
                            if hand_type < HAND_TRIPS:
                                return FoldAction() if FoldAction in legal else CheckAction()

                # On paired boards, our "two pair" might just be board pair + one card
                # Require trips+ or very high equity to call all-ins
                if self._is_paired_board(board) and hand_type < HAND_TRIPS:
                    if adj_equity < 0.65:  # Need high equity on paired boards
                        return FoldAction() if FoldAction in legal else CheckAction()

                # For true all-ins (80%+ stack), require trips+ or 60%+ equity
                if is_allin:
                    if hand_type < HAND_TRIPS and adj_equity < 0.60:
                        return FoldAction() if FoldAction in legal else CheckAction()

                if street >= 5:
                    if hand_type < HAND_TWO_PAIR:
                        if opp_type not in ("MANIAC", "SHOVE_BOT"):
                            return FoldAction() if FoldAction in legal else CheckAction()
                        if adj_equity < 0.45:
                            return FoldAction() if FoldAction in legal else CheckAction()
                    if hand_type == HAND_TWO_PAIR and adj_equity < 0.55:
                        return FoldAction() if FoldAction in legal else CheckAction()
                    if adj_equity < 0.50:
                        return FoldAction() if FoldAction in legal else CheckAction()
                else:
                    if hand_type < HAND_TWO_PAIR and adj_equity < 0.55:
                        return FoldAction() if FoldAction in legal else CheckAction()
                    if adj_equity < 0.45:
                        return FoldAction() if FoldAction in legal else CheckAction()

            # Strong hand - but DON'T raise into their bets unless we have trips+
            if adj_equity >= 0.68:
                if hand_type >= HAND_TRIPS:
                    # We have trips or better - NOW we can raise for value
                    if RaiseAction in legal and random.random() < 0.70:
                        min_r, max_r = round_state.raise_bounds()
                        size = min(max_r, opp_pip + int(pot * random.uniform(0.8, 1.2)))
                        return self._do_raise(max(min_r, size))
                # With less than trips, just CALL (don't raise into traps)
                return CallAction() if CallAction in legal else CheckAction()

            elif adj_equity >= 0.52:
                # Medium strength - just call, don't raise
                return CallAction() if CallAction in legal else CheckAction()

            elif adj_equity >= 0.35:
                if street >= 5 and opp_type in ("VERY_TIGHT", "TIGHT", "NORMAL"):
                    if bet_ratio > 0.5 and adj_equity < 0.55:
                        return FoldAction() if FoldAction in legal else CheckAction()
                    if bet_ratio > 0.75 and adj_equity < 0.60:
                        return FoldAction() if FoldAction in legal else CheckAction()

                if opp_type in ("VERY_TIGHT", "TIGHT") and bet_ratio > 0.5 and adj_equity < 0.48:
                    return FoldAction() if FoldAction in legal else CheckAction()

                equity_buffer = 0.08 if bet_ratio > 0.75 else 0.03
                equity_buffer += position_buffer  # Tighter OOP, looser IP
                if adj_equity >= pot_odds + equity_buffer:
                    return CallAction() if CallAction in legal else CheckAction()
                if street < 5 and adj_equity >= pot_odds * 0.85 and min(my_stack, opp_stack) > pot * 2:
                    return CallAction() if CallAction in legal else CheckAction()
                return FoldAction() if FoldAction in legal else CheckAction()

            else:
                if opp_type in ("VERY_TIGHT", "TIGHT") and in_position and RaiseAction in legal:
                    if random.random() < 0.08 and gear in ("DESPERATE", "AGGRESSIVE"):
                        min_r, _ = round_state.raise_bounds()
                        return self._do_raise(min_r)
                return FoldAction() if FoldAction in legal else CheckAction()

        # not facing bet
        else:
            if opp_type == "MANIAC" and is_strong_hand:
                return CheckAction()

            # With MONSTER hands (trips+, 85%+ equity), sometimes CHECK
            # to let aggressive opponents bet into us
            # Then we can check-raise or call down for max value
            is_monster = hand_type >= HAND_TRIPS or equity >= 0.85
            opp_is_aggressive = opp_type in ("MANIAC", "LOOSE_AGGRESSIVE", "SHOVE_BOT") or self._get_aggro_raise_rate() > 0.20

            if is_monster and opp_is_aggressive:
                # Check with monsters ~50% of time vs aggressive opponents
                if random.random() < 0.50:
                    return CheckAction()

            # Also trap on the RIVER with nuts - let them bluff
            if street == 6 and we_have_nuts and opp_is_aggressive:
                if random.random() < 0.40:
                    return CheckAction()

            # With NUTS on river, bet TINY to induce a raise, then we can re-raise
            if street == 6 and equity > 0.95 and not self.we_bet_this_street:
                if random.random() < 0.30:  # 30% of time, bet tiny to induce
                    if RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        induce_size = int(pot * 0.15)  # Tiny bet looks weak
                        self.we_bet_this_street = True
                        return self._do_raise(max(min_r, min(max_r, my_pip + induce_size)))

            # When board has 3-flush or 4-straight and we have nothing,
            # bluff 12% of the time to balance our range
            if is_scary_board and equity < 0.30 and not self.we_bet_this_street:
                if random.random() < 0.12:  # Bluff 12% of time
                    if RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        bluff_size = int(pot * 0.70)  # Big size to look like value
                        self.we_bet_this_street = True
                        return self._do_raise(max(min_r, min(max_r, my_pip + bluff_size)))

            # When opponent checks with weak equity, probe to steal
            if not self.we_bet_this_street and equity < 0.50 and equity > 0.25:
                # They checked, we have marginal hand - probe to steal
                if random.random() < 0.35:  # Probe 35% of time
                    if RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        probe_size = int(pot * 0.30)  # Small probe
                        self.we_bet_this_street = True
                        return self._do_raise(max(min_r, min(max_r, my_pip + probe_size)))

            # Check this FIRST since 80% > 70%
            if equity > 0.80 and hand_type >= HAND_TWO_PAIR and not self.we_bet_this_street and RaiseAction in legal:
                min_r, max_r = round_state.raise_bounds()
                # vary bet size to be unpredictable
                roll = random.random()
                if roll < 0.15:
                    # 15% - Overbet to look like bluff
                    protect_size = int(pot * 1.10)
                elif roll < 0.30:
                    # 15% - Small bet to induce raise
                    protect_size = int(pot * 0.40)
                else:
                    # 70% - Standard value bet
                    protect_size = int(pot * 0.75)
                self.we_bet_this_street = True
                return self._do_raise(max(min_r, min(max_r, my_pip + protect_size)))


            # Mix up actions to be less predictable
            if equity > 0.70 and not self.we_bet_this_street and RaiseAction in legal:
                roll = random.random()
                if roll < 0.15:
                    # 15% - Check to trap (protect checking range)
                    return CheckAction()
                elif roll < 0.25:
                    # 10% - Overbet to look like bluff
                    min_r, max_r = round_state.raise_bounds()
                    overbet_size = int(pot * 1.05)
                    self.we_bet_this_street = True
                    return self._do_raise(max(min_r, min(max_r, my_pip + overbet_size)))
                else:
                    # 75% - Standard protection bet
                    min_r, max_r = round_state.raise_bounds()
                    protection_size = int(pot * 0.55)
                    self.we_bet_this_street = True
                    return self._do_raise(max(min_r, min(max_r, my_pip + protection_size)))

            # Bet with strong hands even without preflop aggression
            if street >= 4 and equity >= 0.70 and hand_type >= HAND_PAIR and not self.we_bet_this_street:
                if RaiseAction in legal:
                    min_r, max_r = round_state.raise_bounds()
                    value_size = int(pot * 0.65)
                    self.we_bet_this_street = True
                    return self._do_raise(max(min_r, min(max_r, my_pip + value_size)))

            # If opponent checks turn (gave up?), probe with 33% pot
            if street == 5 and not self.we_bet_this_street and not is_elite:
                if equity >= 0.35 and RaiseAction in legal:
                    if random.random() < 0.55:  # 55% probe frequency
                        min_r, max_r = round_state.raise_bounds()
                        probe_size = int(pot * 0.33)
                        self.we_bet_this_street = True
                        return self._do_raise(max(min_r, min(max_r, my_pip + probe_size)))

            if street == 4 and self.was_preflop_aggressor and not is_elite:
                cbet_freq = 0.75
                if random.random() < cbet_freq:
                    if RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        cbet_size = int(pot * 0.50)  # Increased from 0.33
                        self.we_bet_this_street = True
                        return self._do_raise(max(min_r, min(max_r, my_pip + cbet_size)))

            if is_lead_lock and equity < 0.60:
                return CheckAction()

            if street >= 4 and 0.55 <= equity < 0.85:
                if is_elite:
                    if random.random() < 0.15:
                        if RaiseAction in legal:
                            min_r, max_r = round_state.raise_bounds()
                            lead_size = int(pot * random.uniform(0.45, 0.65))
                            return self._do_raise(max(min_r, min(max_r, my_pip + lead_size)))

            if street >= 4 and equity > 0.85:
                if random.random() < 0.40:
                    return CheckAction()

            if street >= 5:
                # When activated, unleash massive overbets
                if TOURNAMENT_MODE:
                    # If we have near-certain winner, shove for max value
                    if equity >= 0.95 and RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        # 200% pot overbet or all-in
                        overbet = min(max_r, my_pip + int(pot * 2.0))
                        return self._do_raise(max(min_r, overbet))

                    # if we have nut blocker on scary board, overbet bluff
                    if equity < 0.25 and self._has_nut_blocker(my_cards, board) and is_scary_board:
                        if RaiseAction in legal and random.random() < 0.35:
                            min_r, max_r = round_state.raise_bounds()
                            # All-in bluff
                            return self._do_raise(max_r)

                # MONSTER HANDS (90%+ equity) - GO BIG
                if equity >= 0.90:
                    # Bet 150-200% pot - extract maximum value
                    bet_size = int(pot * random.uniform(1.5, 2.0))
                    if bet_size > 0 and RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        return self._do_raise(max(min_r, min(max_r, my_pip + bet_size)))

                # STRONG HANDS (75-90% equity) - Bet for value
                elif equity >= 0.75:
                    # Bet 100-130% pot
                    bet_size = int(pot * random.uniform(1.0, 1.3))
                    if bet_size > 0 and RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        return self._do_raise(max(min_r, min(max_r, my_pip + bet_size)))

                # GOOD HANDS (65-75% equity) - Standard value bet
                elif equity >= 0.65:
                    if opp_type in ("CALLING_STATION", "LOOSE_PASSIVE"):
                        bet_size = int(pot * 1.2)  # They call too much, bet bigger
                    else:
                        bet_size = int(pot * 0.75)
                    if bet_size > 0 and RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        return self._do_raise(max(min_r, min(max_r, my_pip + bet_size)))

                # MEDIUM HANDS (50-65% equity) - Thin value
                elif equity >= 0.50 and hand_type >= HAND_PAIR:
                    thin_value_size = int(pot * 0.5)  # Increased from 0.25
                    if thin_value_size > 0 and RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        return self._do_raise(max(min_r, min(max_r, my_pip + thin_value_size)))

                # BLUFF CANDIDATES (<30% equity)
                elif equity < 0.30:
                    if is_elite:
                        if self._has_nut_blocker(my_cards, board) and self._is_scary_board(board):
                            if random.random() < 0.12:
                                bluff_size = int(pot * 0.75)
                                if bluff_size > 0 and RaiseAction in legal:
                                    min_r, max_r = round_state.raise_bounds()
                                    return self._do_raise(max(min_r, min(max_r, my_pip + bluff_size)))
                    else:
                        if opp_type not in ("CALLING_STATION", "LOOSE_PASSIVE"):
                            if self._should_bluff_river(board, opp_type, in_position, my_cards):
                                freq = {"DESPERATE": 0.22, "AGGRESSIVE": 0.18, "CONSERVATIVE": 0.06, "SLIGHTLY_TIGHT": 0.06}.get(gear, 0.12)
                                if random.random() < freq:
                                    bluff_size = self._get_bet_size(0, pot, street, opp_type, is_bluff=True)
                                    if bluff_size > 0 and RaiseAction in legal:
                                        min_r, max_r = round_state.raise_bounds()
                                        return self._do_raise(max(min_r, min(max_r, my_pip + bluff_size)))

                elif opp_type in ("CALLING_STATION", "LOOSE_PASSIVE") and equity >= 0.45:
                    if RaiseAction in legal:
                        min_r, max_r = round_state.raise_bounds()
                        return self._do_raise(max(min_r, min(max_r, my_pip + int(pot * 0.45))))

                return CheckAction()

            else:
                bet_size = self._get_bet_size(equity, pot, street, opp_type)
                if gear == "DESPERATE":
                    bet_size = int(bet_size * 1.15)
                elif gear == "AGGRESSIVE":
                    bet_size = int(bet_size * 1.08)
                elif gear in ("CONSERVATIVE", "SLIGHTLY_TIGHT"):
                    bet_size = int(bet_size * 0.90)

                if bet_size > 0 and RaiseAction in legal:
                    min_r, max_r = round_state.raise_bounds()
                    return self._do_raise(max(min_r, min(max_r, my_pip + bet_size)))

                return CheckAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
