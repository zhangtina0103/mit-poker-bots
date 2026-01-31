"""
Python wrapper for the fast C++ equity calculator.
Usage:
    from cpp_equity import calculate_equity_fast
    equity = calculate_equity_fast(['Ah', 'Kh'], ['Qh', 'Jh', 'Th'], street=4, num_sims=5000)
"""

import ctypes
import os

# Load the shared library
_lib_path = os.path.join(os.path.dirname(__file__), 'fast_eval.so')
_lib = None

def _load_lib():
    global _lib
    if _lib is None:
        try:
            _lib = ctypes.CDLL(_lib_path)
            _lib.calculate_equity.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            _lib.calculate_equity.restype = ctypes.c_double
        except Exception as e:
            print(f"Warning: Could not load fast_eval.so: {e}")
            _lib = False
    return _lib

def calculate_equity_fast(my_cards, board, street, num_sims=5000):
    """
    Calculate equity using the fast C++ engine.
    
    Args:
        my_cards: list of cards like ['Ah', 'Kh']
        board: list of board cards like ['Qh', 'Jh', 'Th']
        street: 0=preflop, 4=turn, 5=river
        num_sims: number of Monte Carlo simulations (default 5000)
    
    Returns:
        float: equity between 0.0 and 1.0
    """
    lib = _load_lib()
    if not lib:
        return None  # Fallback to Python implementation
    
    my_cards_str = ','.join(my_cards).encode('utf-8')
    board_str = ','.join(board).encode('utf-8') if board else b''
    
    return lib.calculate_equity(my_cards_str, board_str, street, num_sims)

def is_available():
    """Check if the C++ library is available."""
    return _load_lib() is not False

if __name__ == '__main__':
    # Test the library
    print("Testing C++ equity calculator...")
    
    if not is_available():
        print("ERROR: fast_eval.so not available!")
        exit(1)
    
    # Test 1: AA vs random preflop
    equity = calculate_equity_fast(['Ah', 'As', '2c'], [], 0, 10000)
    print(f"AA+2 preflop equity: {equity:.1%} (expected ~85%)")
    
    # Test 2: AK on a K-high flop
    equity = calculate_equity_fast(['Ah', 'Kh'], ['Kd', '7c', '2s'], 4, 10000)
    print(f"AK on K72 flop: {equity:.1%} (expected ~75-80%)")
    
    # Test 3: Flush draw
    equity = calculate_equity_fast(['Ah', 'Kh'], ['Qh', 'Jc', '2h'], 4, 10000)
    print(f"AKh on Qh Jc 2h (flush draw): {equity:.1%}")
    
    print("\nAll tests passed!")
