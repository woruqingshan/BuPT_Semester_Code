# -*- coding: utf-8 -*-

def compare_paths(paths):
    
    if not paths:
        return None
    min_length = float('inf')
    best_path = None
    for path in paths:
        length = len(path)
        if length < min_length:
            min_length = length
            best_path = path
    return best_path