/*
 * Fast Poker Hand Evaluator for Pokerbots
 * Compile: g++ -O3 -shared -fPIC -o fast_eval.so fast_eval.cpp
 *
 * This provides a fast equity calculation using 10,000 Monte Carlo simulations
 * instead of the ~40-100 Python can handle in the time limit.
 */

#include <algorithm>
#include <random>
#include <vector>
#include <string>
#include <cstring>

// Card representation: 0-51
// rank = card / 4 (0=2, 1=3, ..., 12=A)
// suit = card % 4 (0=h, 1=d, 2=c, 3=s)

static std::mt19937 rng(std::random_device{}());

int card_to_int(const char* card) {
    int rank;
    switch(card[0]) {
        case '2': rank = 0; break;
        case '3': rank = 1; break;
        case '4': rank = 2; break;
        case '5': rank = 3; break;
        case '6': rank = 4; break;
        case '7': rank = 5; break;
        case '8': rank = 6; break;
        case '9': rank = 7; break;
        case 'T': rank = 8; break;
        case 'J': rank = 9; break;
        case 'Q': rank = 10; break;
        case 'K': rank = 11; break;
        case 'A': rank = 12; break;
        default: return -1;
    }
    int suit;
    switch(card[1]) {
        case 'h': suit = 0; break;
        case 'd': suit = 1; break;
        case 'c': suit = 2; break;
        case 's': suit = 3; break;
        default: return -1;
    }
    return rank * 4 + suit;
}

// Hand evaluation - returns a comparable score
// Higher is better
// Format: (hand_type << 20) | tiebreaker
// hand_type: 1=high card, 2=pair, 3=two pair, 4=trips, 5=straight,
//            6=flush, 7=full house, 8=quads, 9=straight flush

int evaluate_hand(int* cards, int n) {
    if (n < 5) return 0;

    int ranks[13] = {0};
    int suits[4] = {0};
    int rank_list[7];
    int suit_list[7];

    for (int i = 0; i < n; i++) {
        int rank = cards[i] / 4;
        int suit = cards[i] % 4;
        ranks[rank]++;
        suits[suit]++;
        rank_list[i] = rank;
        suit_list[i] = suit;
    }

    // Check for flush
    int flush_suit = -1;
    for (int s = 0; s < 4; s++) {
        if (suits[s] >= 5) {
            flush_suit = s;
            break;
        }
    }

    // Check for straight
    int straight_high = -1;
    int consecutive = 0;
    for (int r = 12; r >= 0; r--) {
        if (ranks[r] > 0) {
            consecutive++;
            if (consecutive >= 5) {
                straight_high = r + 4;
                break;
            }
        } else {
            consecutive = 0;
        }
    }
    // Check A-2-3-4-5 straight
    if (straight_high < 0 && ranks[12] > 0 && ranks[0] > 0 && ranks[1] > 0 && ranks[2] > 0 && ranks[3] > 0) {
        straight_high = 3; // 5-high straight
    }

    // Check for straight flush
    if (flush_suit >= 0 && straight_high >= 0) {
        int flush_ranks[13] = {0};
        for (int i = 0; i < n; i++) {
            if (suit_list[i] == flush_suit) {
                flush_ranks[rank_list[i]]++;
            }
        }
        int sf_high = -1;
        consecutive = 0;
        for (int r = 12; r >= 0; r--) {
            if (flush_ranks[r] > 0) {
                consecutive++;
                if (consecutive >= 5) {
                    sf_high = r + 4;
                    break;
                }
            } else {
                consecutive = 0;
            }
        }
        if (sf_high < 0 && flush_ranks[12] > 0 && flush_ranks[0] > 0 &&
            flush_ranks[1] > 0 && flush_ranks[2] > 0 && flush_ranks[3] > 0) {
            sf_high = 3;
        }
        if (sf_high >= 0) {
            return (9 << 20) | sf_high;
        }
    }

    // Count pairs, trips, quads
    int quads = -1, trips = -1, pair1 = -1, pair2 = -1;
    for (int r = 12; r >= 0; r--) {
        if (ranks[r] == 4) {
            quads = r;
        } else if (ranks[r] == 3) {
            if (trips < 0) trips = r;
            else if (pair1 < 0) pair1 = r;
        } else if (ranks[r] == 2) {
            if (pair1 < 0) pair1 = r;
            else if (pair2 < 0) pair2 = r;
        }
    }

    // Get kickers
    std::vector<int> kickers;
    for (int r = 12; r >= 0; r--) {
        if (ranks[r] > 0 && r != quads && r != trips && r != pair1 && r != pair2) {
            for (int i = 0; i < ranks[r]; i++) {
                kickers.push_back(r);
            }
        }
    }

    // Four of a kind
    if (quads >= 0) {
        int kicker = kickers.empty() ? 0 : kickers[0];
        return (8 << 20) | (quads << 4) | kicker;
    }

    // Full house
    if (trips >= 0 && pair1 >= 0) {
        return (7 << 20) | (trips << 4) | pair1;
    }

    // Flush
    if (flush_suit >= 0) {
        std::vector<int> flush_cards;
        for (int i = 0; i < n; i++) {
            if (suit_list[i] == flush_suit) {
                flush_cards.push_back(rank_list[i]);
            }
        }
        std::sort(flush_cards.rbegin(), flush_cards.rend());
        int score = 0;
        for (int i = 0; i < 5 && i < (int)flush_cards.size(); i++) {
            score = score * 13 + flush_cards[i];
        }
        return (6 << 20) | score;
    }

    // Straight
    if (straight_high >= 0) {
        return (5 << 20) | straight_high;
    }

    // Three of a kind
    if (trips >= 0) {
        int k1 = kickers.size() > 0 ? kickers[0] : 0;
        int k2 = kickers.size() > 1 ? kickers[1] : 0;
        return (4 << 20) | (trips << 8) | (k1 << 4) | k2;
    }

    // Two pair
    if (pair1 >= 0 && pair2 >= 0) {
        int kicker = kickers.empty() ? 0 : kickers[0];
        return (3 << 20) | (pair1 << 8) | (pair2 << 4) | kicker;
    }

    // One pair
    if (pair1 >= 0) {
        int score = pair1 << 12;
        for (int i = 0; i < 3 && i < (int)kickers.size(); i++) {
            score |= kickers[i] << (8 - i * 4);
        }
        return (2 << 20) | score;
    }

    // High card
    int score = 0;
    for (int i = 0; i < 5 && i < (int)kickers.size(); i++) {
        score = score * 13 + kickers[i];
    }
    return (1 << 20) | score;
}

int best_five_of_seven(int* cards) {
    int best = 0;
    int combo[5];
    // Try all C(7,5) = 21 combinations
    for (int i = 0; i < 7; i++) {
        for (int j = i+1; j < 7; j++) {
            int idx = 0;
            for (int k = 0; k < 7; k++) {
                if (k != i && k != j) {
                    combo[idx++] = cards[k];
                }
            }
            int score = evaluate_hand(combo, 5);
            if (score > best) best = score;
        }
    }
    return best;
}

extern "C" {

// Calculate equity using Monte Carlo simulation
// my_cards: comma-separated string like "Ah,Kh"
// board: comma-separated string like "Qh,Jh,Th" (can be empty)
// street: 0=preflop, 4=turn, 5=river
// num_sims: number of simulations (recommend 1000-10000)
// Returns: equity as float 0.0 to 1.0
double calculate_equity(const char* my_cards_str, const char* board_str, int street, int num_sims) {
    // Parse my cards
    std::vector<int> my_cards;
    char buf[256];
    strncpy(buf, my_cards_str, 255);
    buf[255] = 0;
    char* token = strtok(buf, ",");
    while (token) {
        int c = card_to_int(token);
        if (c >= 0) my_cards.push_back(c);
        token = strtok(NULL, ",");
    }

    // Parse board
    std::vector<int> board;
    strncpy(buf, board_str, 255);
    buf[255] = 0;
    token = strtok(buf, ",");
    while (token) {
        int c = card_to_int(token);
        if (c >= 0) board.push_back(c);
        token = strtok(NULL, ",");
    }

    if (my_cards.empty()) return 0.5;

    // Build remaining deck
    bool used[52] = {false};
    for (int c : my_cards) used[c] = true;
    for (int c : board) used[c] = true;

    std::vector<int> deck;
    for (int i = 0; i < 52; i++) {
        if (!used[i]) deck.push_back(i);
    }

    // Target board size is 6
    int cards_needed = 6 - (int)board.size();
    // Opponent has 3 cards preflop, 2 after discard
    int opp_cards = (street == 0) ? 3 : 2;

    if ((int)deck.size() < opp_cards + cards_needed) return 0.5;

    double wins = 0.0;

    for (int sim = 0; sim < num_sims; sim++) {
        // Shuffle deck
        std::shuffle(deck.begin(), deck.end(), rng);

        // Deal opponent cards and future board
        std::vector<int> opp(deck.begin(), deck.begin() + opp_cards);
        std::vector<int> future_board(deck.begin() + opp_cards, deck.begin() + opp_cards + cards_needed);

        // Build full hands
        std::vector<int> full_board = board;
        for (int c : future_board) full_board.push_back(c);

        // Evaluate - combine hole cards with board
        int my_all[9], opp_all[9];
        int my_n = 0, opp_n = 0;
        for (int c : my_cards) my_all[my_n++] = c;
        for (int c : full_board) my_all[my_n++] = c;
        for (int c : opp) opp_all[opp_n++] = c;
        for (int c : full_board) opp_all[opp_n++] = c;

        // Find best 5-card hand from available cards
        int my_score = 0, opp_score = 0;

        // For my hand: try all 5-card combinations
        if (my_n >= 5) {
            for (int a = 0; a < my_n; a++) {
                for (int b = a+1; b < my_n; b++) {
                    for (int c = b+1; c < my_n; c++) {
                        for (int d = c+1; d < my_n; d++) {
                            for (int e = d+1; e < my_n; e++) {
                                int combo[5] = {my_all[a], my_all[b], my_all[c], my_all[d], my_all[e]};
                                int score = evaluate_hand(combo, 5);
                                if (score > my_score) my_score = score;
                            }
                        }
                    }
                }
            }
        }

        // For opponent hand
        if (opp_n >= 5) {
            for (int a = 0; a < opp_n; a++) {
                for (int b = a+1; b < opp_n; b++) {
                    for (int c = b+1; c < opp_n; c++) {
                        for (int d = c+1; d < opp_n; d++) {
                            for (int e = d+1; e < opp_n; e++) {
                                int combo[5] = {opp_all[a], opp_all[b], opp_all[c], opp_all[d], opp_all[e]};
                                int score = evaluate_hand(combo, 5);
                                if (score > opp_score) opp_score = score;
                            }
                        }
                    }
                }
            }
        }

        if (my_score > opp_score) wins += 1.0;
        else if (my_score == opp_score) wins += 0.5;
    }

    return wins / num_sims;
}

} // extern "C"
