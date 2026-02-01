import random
import heapq
from collections import deque
from typing import Tuple, Optional, List, Dict, Any

from game_constants import Team, TileType, FoodType, ShopCosts
from robot_controller import RobotController
from item import Pan, Plate, Food


class BotPlayer:
    """
    A complete bot implementation for Carnegie Cookoff.
    
    This bot uses a state machine approach to manage cooking tasks:
    - State 0: Initialize and check for pan
    - State 1: Buy and place pan on cooker
    - State 2: Buy meat
    - State 3: Place meat on counter
    - State 4: Chop meat  
    - State 5: Pickup chopped meat
    - State 6: Place meat in pan (starts cooking)
    - State 7: (skipped - cooking starts automatically)
    - State 8: Buy plate
    - State 9: Place plate on counter
    - State 10: Buy noodles
    - State 11: Add noodles to plate
    - State 12: Wait for meat to cook and take from pan
    - State 13: Add meat to plate
    - State 14: Pickup completed plate
    - State 15: Submit order
    - State 16: Trash (error recovery)
    
    EGG RECIPE (cook only, no chop):
    - State 20: Buy egg
    - State 21: Place egg in pan (starts cooking)
    - State 22: Wait for egg to cook and take from pan
    - State 23: Add egg to plate
    
    ONION RECIPE (chop only, no cook):
    - State 30: Buy onions
    - State 31: Place onions on counter
    - State 32: Chop onions
    - State 33: Pickup chopped onions
    - State 34: Add onions to plate
    
    SAUCE RECIPE (direct add, no processing):
    - State 40: Buy sauce
    - State 41: Add sauce to plate
    """
    
    def __init__(self, map_copy):
        self.map = map_copy
        
        # Pre-cache tile locations for O(1) lookup instead of O(n²) grid scans
        self._tile_cache = {}  # tile_name -> list of (x, y) positions
        self._num_counters = 0
        self._num_cookers = 0
        self._cache_initialized = False
        
        # Dual bot mode - track if we have enough resources for 2 bots
        self.dual_bot_mode = False
        self.bot_states = {}  # bot_id -> BotState dict
        
        # Resource allocation for dual bot mode
        self.bot_resources = {}  # bot_id -> {'counter': (x,y), 'cooker': (x,y)}
        
        # Turn counter for alternating bot execution
        self.turn_counter = 0
        
    def _create_bot_state(self) -> Dict[str, Any]:
        """Create a fresh state dict for a bot."""
        return {
            'assembly_counter': None,
            'chopping_counter': None,
            'cooker_loc': None,
            'state': 0,
            'current_order': None,
            'current_order_ingredients': [],
            'current_ingredient_index': 0,
            'plate_ready': False,
            'plate_temp_held': False,
            'held_food_for_chop': None,
            'single_counter_mode': False,
            'chopped_items_ready': [],
            'meat_in_pan': False,
            'meat_placed_turn': None,
            'last_move': None,  # (dx, dy) of last movement for momentum
            'stuck_counter': 0,  # Count turns stuck in same position
            'last_position': None,  # Last position for stuck detection
            'position_history': [],  # Track recent positions for oscillation detection
            'oscillation_counter': 0,  # Track how long we've been oscillating
        }
        
    def _init_tile_cache(self):
        """Initialize tile location cache for O(1) lookups."""
        if self._cache_initialized:
            return
        self._tile_cache = {}
        self._num_counters = 0
        self._num_cookers = 0
        self._num_walkable = 0
        for x in range(self.map.width):
            for y in range(self.map.height):
                tile = self.map.tiles[x][y]
                tile_name = tile.tile_name
                if tile_name not in self._tile_cache:
                    self._tile_cache[tile_name] = []
                self._tile_cache[tile_name].append((x, y))
                if tile_name == "COUNTER":
                    self._num_counters += 1
                if tile_name == "COOKER":
                    self._num_cookers += 1
                if tile.is_walkable:
                    self._num_walkable += 1
        
        # Check if we have enough resources for dual bot mode
        # Need at least 2 counters and 2 cookers for 2 bots to work simultaneously
        # Also need enough walkable space - narrow maps like orbit cause collisions
        self.dual_bot_mode = (self._num_counters >= 2 and self._num_cookers >= 2 and self._num_walkable >= 80)
        self._cache_initialized = True

    def get_bfs_path(self, controller: RobotController, start: Tuple[int, int], target_predicate, blocked_positions: set = None) -> Optional[Tuple[int, int]]:
        """
        Use Dijkstra's algorithm to find the shortest path.
        Returns the first step (dx, dy) to take, or None if no path exists.
        Diagonal moves cost sqrt(2), orthogonal moves cost 1.
        Optimized: uses parent pointers instead of storing full paths.
        blocked_positions: set of (x, y) tuples that should be treated as impassable (other bots)
        """
        SQRT2 = 1.41421356
        if blocked_positions is None:
            blocked_positions = set()
        
        # Priority queue: (cost, counter, x, y) - counter breaks ties for stable ordering
        counter = 0
        heap = [(0, counter, start[0], start[1])]
        # Maps (x, y) -> (best_cost, parent_x, parent_y, dx_from_parent, dy_from_parent)
        visited = {start: (0, None, None, 0, 0)}
        w, h = self.map.width, self.map.height
        team = controller.get_team()
        game_map = controller.get_map(team)

        while heap:
            cost, _, curr_x, curr_y = heapq.heappop(heap)
            
            # Skip if we've already found a better path to this node
            if visited[(curr_x, curr_y)][0] < cost:
                continue
            
            tile = controller.get_tile(team, curr_x, curr_y)
            if target_predicate(curr_x, curr_y, tile):
                # Reconstruct first step by walking back to start
                if (curr_x, curr_y) == start:
                    return (0, 0)
                # Walk back to find the first step
                px, py = curr_x, curr_y
                while True:
                    _, parent_x, parent_y, dx, dy = visited[(px, py)]
                    if parent_x is None or (parent_x, parent_y) == start:
                        return (dx, dy)
                    px, py = parent_x, parent_y

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = curr_x + dx, curr_y + dy
                    # Check walkable AND not blocked by another bot
                    if 0 <= nx < w and 0 <= ny < h and game_map.is_tile_walkable(nx, ny) and (nx, ny) not in blocked_positions:
                        # Calculate move cost: diagonal = sqrt(2), orthogonal = 1
                        move_cost = SQRT2 if (dx != 0 and dy != 0) else 1.0
                        new_cost = cost + move_cost
                        
                        # Only explore if we haven't found a better path
                        if (nx, ny) not in visited or visited[(nx, ny)][0] > new_cost:
                            visited[(nx, ny)] = (new_cost, curr_x, curr_y, dx, dy)
                            counter += 1
                            heapq.heappush(heap, (new_cost, counter, nx, ny))
        return None

    def get_shortest_path_length(self, controller: RobotController, start: Tuple[int, int], target_predicate) -> Optional[int]:
        """
        Returns the length (number of steps) of the shortest path to target.
        """
        queue = deque([(start, 0)]) 
        visited = set([start])
        w, h = self.map.width, self.map.height

        while queue:
            (curr_x, curr_y), dist = queue.popleft()
            tile = controller.get_tile(controller.get_team(), curr_x, curr_y)
            if target_predicate(curr_x, curr_y, tile):
                return dist

            for dx in [0, -1, 1]:
                for dy in [0, -1, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
                            visited.add((nx, ny))
                            queue.append(((nx, ny), dist + 1))
        return None

    def _get_other_bot_positions(self, controller: RobotController, exclude_bot_id: int) -> set:
        """Get positions of all other bots on our team."""
        team = controller.get_team()
        my_bots = controller.get_team_bot_ids(team)
        positions = set()
        for bot_id in my_bots:
            if bot_id != exclude_bot_id:
                bot_state = controller.get_bot_state(bot_id)
                if bot_state:
                    positions.add((bot_state['x'], bot_state['y']))
        return positions

    def move_towards(self, controller: RobotController, bot_id: int, target_x: int, target_y: int) -> bool:
        """
        Move the bot towards the target position, avoiding other bots.
        Returns True if already adjacent to target, False otherwise.
        Uses momentum to prefer continuing in the same direction to avoid oscillation.
        """
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
            
        bx, by = bot_state['x'], bot_state['y']
        bs = self.bot_states.get(bot_id, {})
        
        def is_adjacent_to_target(x, y, tile):
            return max(abs(x - target_x), abs(y - target_y)) <= 1
        
        if is_adjacent_to_target(bx, by, None):
            return True
        
        # Track if we're stuck (same position for multiple turns)
        if bs.get('last_position') == (bx, by):
            bs['stuck_counter'] = bs.get('stuck_counter', 0) + 1
        else:
            bs['stuck_counter'] = 0
        bs['last_position'] = (bx, by)
        
        # Get positions of other bots to avoid
        other_bot_positions = self._get_other_bot_positions(controller, bot_id)
        game_map = controller.get_map(controller.get_team())
        
        # First try to find path avoiding other bots
        step = self.get_bfs_path(controller, (bx, by), is_adjacent_to_target, other_bot_positions)
        
        # If no path found or if the step would move us AWAY from target (indicating blockage),
        # try pathfinding ignoring other bots - they might move
        if step:
            # Check if this step moves us further from target (sign of blocked path)
            new_x, new_y = bx + step[0], by + step[1]
            curr_dist = max(abs(bx - target_x), abs(by - target_y))
            new_dist = max(abs(new_x - target_x), abs(new_y - target_y))
            
            if new_dist > curr_dist and bs.get('stuck_counter', 0) > 0:
                # Path is blocked, try ignoring other bots
                ideal_step = self.get_bfs_path(controller, (bx, by), is_adjacent_to_target, set())
                if ideal_step:
                    ideal_x, ideal_y = bx + ideal_step[0], by + ideal_step[1]
                    if (ideal_x, ideal_y) not in other_bot_positions:
                        # The ideal path is not blocked, use it
                        step = ideal_step
                    else:
                        # Ideal path IS blocked by another bot, wait for them
                        return False
        
        if step and (step[0] != 0 or step[1] != 0):
            # Check momentum: if we have a last move and we're not stuck, prefer to continue that direction
            last_move = bs.get('last_move')
            
            # Check if continuing last move is still valid and makes progress
            if last_move:
                nx, ny = bx + last_move[0], by + last_move[1]
                if game_map.is_tile_walkable(nx, ny) and (nx, ny) not in other_bot_positions:
                    # Check if it gets us closer or at least not further
                    curr_dist = max(abs(bx - target_x), abs(by - target_y))
                    new_dist = max(abs(nx - target_x), abs(ny - target_y))
                    if new_dist <= curr_dist:
                        controller.move(bot_id, last_move[0], last_move[1])
                        return False
            
            # Use pathfinder's suggestion
            new_x, new_y = bx + step[0], by + step[1]
            if (new_x, new_y) not in other_bot_positions:
                controller.move(bot_id, step[0], step[1])
                bs['last_move'] = step
                return False
            else:
                # Blocked by another bot - just wait
                return False
        return False

    def find_nearest_tile(self, controller: RobotController, bot_x: int, bot_y: int, tile_name: str) -> Optional[Tuple[int, int]]:
        """
        Find the nearest tile of a given type using Chebyshev distance.
        Optimized: uses cached tile locations for O(k) instead of O(n²).
        """
        self._init_tile_cache()
        
        if tile_name not in self._tile_cache:
            return None
        
        best_dist = 9999
        best_pos = None
        
        for x, y in self._tile_cache[tile_name]:
            dist = max(abs(bot_x - x), abs(bot_y - y))
            if dist < best_dist:
                best_dist = dist
                best_pos = (x, y)
        return best_pos

    def find_empty_counter(self, controller: RobotController, bot_x: int, bot_y: int, exclude: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """
        Find the nearest empty counter tile, optionally excluding a specific position.
        Optimized: uses cached counter locations.
        """
        self._init_tile_cache()
        
        if "COUNTER" not in self._tile_cache:
            return None
        
        best_dist = 9999
        best_pos = None
        team = controller.get_team()
        
        for x, y in self._tile_cache["COUNTER"]:
            # Skip excluded position
            if exclude and (x, y) == exclude:
                continue
            actual_tile = controller.get_tile(team, x, y)
            if actual_tile and getattr(actual_tile, 'item', None) is None:
                dist = max(abs(bot_x - x), abs(bot_y - y))
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (x, y)
        return best_pos

    def find_empty_cooker(self, controller: RobotController, bot_x: int, bot_y: int) -> Optional[Tuple[int, int]]:
        """
        Find the nearest cooker with an empty pan or no pan.
        Optimized: uses cached cooker locations.
        """
        self._init_tile_cache()
        
        if "COOKER" not in self._tile_cache:
            return None
        
        best_dist = 9999
        best_pos = None
        team = controller.get_team()
        
        for x, y in self._tile_cache["COOKER"]:
            actual_tile = controller.get_tile(team, x, y)
            if actual_tile:
                pan = getattr(actual_tile, 'item', None)
                if pan is None or (isinstance(pan, Pan) and pan.food is None):
                    dist = max(abs(bot_x - x), abs(bot_y - y))
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (x, y)
        return best_pos

    def play_turn(self, controller: RobotController):
        """
        Main game loop - called every turn.
        Supports dual bot mode when enough resources are available.
        """
        self._init_tile_cache()
        
        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots:
            return
        
        # Initialize bot states if needed
        for bot_id in my_bots:
            if bot_id not in self.bot_states:
                self.bot_states[bot_id] = self._create_bot_state()
        
        # Determine how many bots can work simultaneously
        num_active_bots = 2 if (self.dual_bot_mode and len(my_bots) >= 2) else 1
        
        # Allocate resources to bots if not done yet
        if num_active_bots >= 2 and not self.bot_resources:
            self._allocate_resources(controller, my_bots[:2])
        
        # Only ONE bot acts per turn to avoid competing for money
        if num_active_bots >= 2:
            # Alternate: turn 0 -> bot 0, turn 1 -> bot 1, turn 2 -> bot 0, etc.
            active_bot_index = self.turn_counter % 2
            if active_bot_index < len(my_bots):
                self._run_bot(controller, my_bots[active_bot_index], active_bot_index)
        else:
            # Single bot mode - just run the one bot
            self._run_bot(controller, my_bots[0], 0)
        
        self.turn_counter += 1
        
        # Secondary bots beyond the active ones just stay out of the way
        for i in range(num_active_bots, len(my_bots)):
            self._handle_idle_bot(controller, my_bots[i])

    def _allocate_resources(self, controller: RobotController, bot_ids: List[int]):
        """Allocate counters and cookers to each bot based on proximity to shop."""
        counters = self._tile_cache.get("COUNTER", [])
        cookers = self._tile_cache.get("COOKER", [])
        shops = self._tile_cache.get("SHOP", [])
        
        if len(counters) < 2 or len(cookers) < 2:
            return
        
        # Find the shop location (use first one if multiple)
        if not shops:
            # Fallback to spatial sorting if no shop found
            counters_sorted = sorted(counters, key=lambda p: (p[0], p[1]))
            cookers_sorted = sorted(cookers, key=lambda p: (p[0], p[1]))
        else:
            shop_x, shop_y = shops[0]
            
            # Sort counters and cookers by distance to shop (closest first)
            counters_sorted = sorted(counters, key=lambda p: max(abs(p[0] - shop_x), abs(p[1] - shop_y)))
            cookers_sorted = sorted(cookers, key=lambda p: max(abs(p[0] - shop_x), abs(p[1] - shop_y)))
        
        # Assign first bot to closest resources, second bot to next closest
        self.bot_resources[bot_ids[0]] = {
            'counter': counters_sorted[0],
            'cooker': cookers_sorted[0]
        }
        # For second bot, pick resources that don't conflict with first bot
        # Use second closest counter and cooker
        second_counter = counters_sorted[1] if len(counters_sorted) > 1 else counters_sorted[0]
        second_cooker = cookers_sorted[1] if len(cookers_sorted) > 1 else cookers_sorted[0]
        
        self.bot_resources[bot_ids[1]] = {
            'counter': second_counter,
            'cooker': second_cooker
        }

    def _run_bot(self, controller: RobotController, bot_id: int, bot_index: int):
        """Run the state machine for a single bot."""
        bot_info = controller.get_bot_state(bot_id)
        if bot_info is None:
            return
        
        bx, by = bot_info['x'], bot_info['y']
        bs = self.bot_states[bot_id]
        
        # Initialize locations for this bot
        if bs['assembly_counter'] is None:
            if bot_id in self.bot_resources:
                bs['assembly_counter'] = self.bot_resources[bot_id]['counter']
            else:
                bs['assembly_counter'] = self.find_nearest_tile(controller, bx, by, "COUNTER")
        
        if bs['cooker_loc'] is None:
            if bot_id in self.bot_resources:
                bs['cooker_loc'] = self.bot_resources[bot_id]['cooker']
            else:
                bs['cooker_loc'] = self.find_nearest_tile(controller, bx, by, "COOKER")
        
        if not bs['assembly_counter'] or not bs['cooker_loc']:
            return
        
        cx, cy = bs['assembly_counter']
        kx, ky = bs['cooker_loc']
        
        # Track position history for oscillation detection
        position_history = bs.get('position_history', [])
        position_history.append((bx, by))
        if len(position_history) > 20:
            position_history = position_history[-20:]
        bs['position_history'] = position_history
        
        # Detect oscillation: if we've visited the same small set of positions repeatedly
        if len(position_history) >= 10:
            recent_positions = set(position_history[-10:])
            if len(recent_positions) <= 4:  # Only 4 or fewer unique positions in last 10 moves
                bs['oscillation_counter'] = bs.get('oscillation_counter', 0) + 1
            else:
                bs['oscillation_counter'] = 0
        
        # If oscillating for too long and holding something, trash and restart
        if bs.get('oscillation_counter', 0) > 15 and bot_info.get('holding'):
            # print(f"Bot {bot_id} detected oscillation, trashing and restarting")
            bs['oscillation_counter'] = 0
            bs['position_history'] = []
            bs['state'] = 16  # Go to trash
            return
        
        print(f"Bot {bot_id} state {bs['state']} at ({bx},{by}), turn={controller.get_turn()}, holding={bool(bot_info.get('holding'))}")
        
        # Error recovery: if we're holding something in certain states, trash it
        if bs['state'] in [2, 8, 10] and bot_info.get('holding'):
            # print(f"Bot {bot_id} error recovery: holding item in state {bs['state']}, going to trash")
            bs['state'] = 16
        
        # State machine for cooking workflow
        self._execute_state(controller, bot_id, bot_info, bx, by, cx, cy, kx, ky, bs, bot_index)

    def _handle_idle_bot(self, controller: RobotController, bot_id: int):
        """Handle bots that aren't actively working - just stay out of the way."""
        bot_info = controller.get_bot_state(bot_id)
        if bot_info is None:
            return
        
        bx, by = bot_info['x'], bot_info['y']
        holding = bot_info.get('holding')
        
        # If holding something, trash it
        if holding:
            trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
            if trash_pos:
                tx, ty = trash_pos
                if self.move_towards(controller, bot_id, tx, ty):
                    controller.trash(bot_id, tx, ty)
            return
        
        # Move randomly to stay out of the way
        possible_moves = []
        game_map = controller.get_map(controller.get_team())
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = bx + dx, by + dy
                if game_map.is_tile_walkable(nx, ny):
                    possible_moves.append((dx, dy))
        
        if possible_moves:
            dx, dy = random.choice(possible_moves)
            controller.move(bot_id, dx, dy)

    def _execute_state(self, controller: RobotController, bot_id: int, bot_info: Dict[str, Any],
                       bx: int, by: int, cx: int, cy: int, kx: int, ky: int,
                       bs: Dict[str, Any], bot_index: int):
        """
        Execute the current state in the cooking state machine.
        bs = bot state dict for this specific bot
        """
        team = controller.get_team()
        
        # State 0: Initialize - fetch orders and pick one to work on
        if bs['state'] == 0:
            # Get ALL orders and current turn for feasibility analysis
            orders = controller.get_orders(team)
            current_turn = controller.get_turn()
            current_money = controller.get_team_money(team)
            
            # Cost lookup for ingredients
            ingredient_costs = {
                'MEAT': 80,
                'NOODLES': 40,
                'ONIONS': 30,
                'EGG': 20,
                'SAUCE': 10,
            }
            
            def estimate_order_cost(ingredients, needs_pan=False):
                """Estimate total cost to complete an order."""
                cost = 2  # Plate cost
                if needs_pan:
                    cost += 4  # Pan cost
                for ing in ingredients:
                    cost += ingredient_costs.get(ing, 0)
                return cost
            
            def estimate_completion_time(ingredients):
                """Estimate turns needed to complete an order based on ingredients."""
                estimated_time = 0
                for ing in ingredients:
                    if ing == 'MEAT':
                        estimated_time += 8   # chop + cook (reduced from 12)
                    elif ing == 'EGG':
                        estimated_time += 5   # cook (reduced from 8)
                    elif ing == 'ONIONS':
                        estimated_time += 4   # chop (reduced from 6)
                    elif ing == 'NOODLES':
                        estimated_time += 2   # just buy and add (reduced from 3)
                    elif ing == 'SAUCE':
                        estimated_time += 2   # just buy and add
                # Add base time for plate + submit + movement (reduced from 6)
                estimated_time += 4
                return estimated_time
            
            def is_feasible(order):
                """Check if an order can realistically be completed in time."""
                expires = order.get('expires_turn', 9999)
                ingredients = order.get('required', [])
                time_left = expires - current_turn
                estimated_time = estimate_completion_time(ingredients)
                # Need at least some margin - if we can't make it, filter it out
                return time_left >= estimated_time * 0.8  # Allow 20% buffer for movement
            
            def can_afford(order):
                """Check if we have enough money to complete this order."""
                ingredients = order.get('required', [])
                needs_pan = any(ing in ['MEAT', 'EGG'] for ing in ingredients)
                # Check if cooker already has a pan
                tile = controller.get_tile(team, kx, ky)
                has_pan = tile and isinstance(getattr(tile, 'item', None), Pan)
                cost = estimate_order_cost(ingredients, needs_pan and not has_pan)
                return current_money >= cost
            
            def score_order(order):
                """
                Score an order based on reward, complexity, and time efficiency.
                Higher score = better order to pick.
                Optimized to strongly prefer simple, high-efficiency orders.
                """
                reward = order.get('reward', 0)
                expires = order.get('expires_turn', 9999)
                ingredients = order.get('required', [])
                
                time_left = expires - current_turn
                estimated_time = estimate_completion_time(ingredients)
                time_margin = time_left - estimated_time
                
                # Primary metric: reward per time unit (efficiency)
                efficiency = reward / max(estimated_time, 1)
                
                # Strong complexity penalty: each ingredient adds significant time/risk
                # 1 ingredient = 1.0, 2 = 0.7, 3 = 0.5, 4 = 0.4, 5 = 0.33
                num_ingredients = len(ingredients)
                complexity_factor = 1.0 / (1 + num_ingredients * 0.45)
                
                # Bonus for no-cook orders (NOODLES, SAUCE, ONIONS only - faster to complete)
                needs_cooking = any(ing in ['MEAT', 'EGG'] for ing in ingredients)
                cook_factor = 0.8 if needs_cooking else 1.2
                
                # Time margin - prefer orders with comfortable margin
                if time_margin < 0:
                    margin_factor = 0.1  # Very low score for likely-to-fail orders
                elif time_margin < estimated_time * 0.5:
                    margin_factor = 0.7  # Tight margin
                else:
                    margin_factor = 1.0  # Comfortable
                
                # Final score: heavily weighted toward efficiency
                score = efficiency * complexity_factor * cook_factor * margin_factor * 100
                
                return score
            
            # Step 1: Filter to only active, uncompleted orders
            candidate_orders = [o for o in orders if o.get('is_active') and o.get('completed_turn') is None]
            
            # Step 2: Filter out orders already being worked on by other bots
            for other_bot_id, other_bs in self.bot_states.items():
                if other_bot_id != bot_id and other_bs.get('current_order'):
                    other_order_id = other_bs['current_order'].get('order_id')
                    candidate_orders = [o for o in candidate_orders if o.get('order_id') != other_order_id]
            
            # Step 3: Filter out orders that CANNOT be completed in time
            feasible_orders = [o for o in candidate_orders if is_feasible(o)]
            
            # Step 4: Filter out orders we can't afford right now
            affordable_orders = [o for o in feasible_orders if can_afford(o)]
            
            # If no affordable orders, wait for money to accumulate
            if not affordable_orders:
                # Check if there are feasible orders we just can't afford yet
                if feasible_orders:
                    return  # Wait for money
                # Fall back to any candidate if nothing is feasible
                affordable_orders = [o for o in candidate_orders if can_afford(o)]
            
            if not affordable_orders:
                return  # No orders we can afford right now, wait for money
            
            # Step 5: Score and sort affordable orders by best value
            affordable_orders.sort(key=score_order, reverse=True)
            bs['current_order'] = affordable_orders[0]
            ingredients = bs['current_order'].get('required', [])
            
            # Check if we have only one counter - if so, reorder to do choppable items first
            # This way we chop before placing the plate
            num_counters = self._num_counters
            
            if num_counters <= 1:
                # Reorder: choppable items (MEAT, ONIONS) first, then cookable (EGG), then direct (NOODLES, SAUCE)
                choppable = [i for i in ingredients if i in ['MEAT', 'ONIONS']]
                cookable_only = [i for i in ingredients if i in ['EGG']]
                direct = [i for i in ingredients if i in ['NOODLES', 'SAUCE']]
                bs['current_order_ingredients'] = choppable + cookable_only + direct
            else:
                bs['current_order_ingredients'] = ingredients
            
            bs['current_ingredient_index'] = 0
            bs['plate_ready'] = False
            bs['plate_temp_held'] = False
            bs['single_counter_mode'] = (num_counters <= 1)
            bs['chopped_items_ready'] = []
            bs['meat_in_pan'] = False
            bs['meat_placed_turn'] = None
            
            # Check if we need a pan (any cookable ingredients)
            needs_pan = any(ing in ['MEAT', 'EGG'] for ing in bs['current_order_ingredients'])
            
            # For single counter mode with choppable items:
            # Do all chopping first, put in pan to cook, THEN buy plate
            has_choppable = any(ing in ['MEAT', 'ONIONS'] for ing in bs['current_order_ingredients'])
            
            # print(f"Bot {bot_id} state 0: Order {bs['current_order'].get('order_id')} with ingredients {bs['current_order_ingredients']}")
            # print(f"Bot {bot_id} state 0: needs_pan={needs_pan}, has_choppable={has_choppable}, single_counter={bs['single_counter_mode']}")
            
            if needs_pan:
                tile = controller.get_tile(team, kx, ky)
                has_pan = tile and isinstance(getattr(tile, 'item', None), Pan)
                # print(f"Bot {bot_id} state 0: Cooker at ({kx},{ky}) has_pan={has_pan}")
                if has_pan:
                    if bs['single_counter_mode'] and has_choppable:
                        # Single counter: start with first choppable ingredient before plate
                        next_state = self._get_state_for_ingredient(bs)
                        # print(f"Bot {bot_id} state 0 -> {next_state} (single counter, chop first)")
                        bs['state'] = next_state
                    else:
                        # print(f"Bot {bot_id} state 0 -> 8 (go buy plate)")
                        bs['state'] = 8  # Go buy plate
                else:
                    # print(f"Bot {bot_id} state 0 -> 1 (go buy pan)")
                    bs['state'] = 1  # Go buy pan first
            else:
                if bs['single_counter_mode'] and has_choppable:
                    # Single counter with choppable but no cookable - chop first
                    next_state = self._get_state_for_ingredient(bs)
                    # print(f"Bot {bot_id} state 0 -> {next_state} (no pan needed, chop first)")
                    bs['state'] = next_state
                else:
                    # print(f"Bot {bot_id} state 0 -> 8 (no pan needed, buy plate)")
                    bs['state'] = 8  # Go buy plate (no pan needed)

        # State 1: Buy pan and place on cooker
        elif bs['state'] == 1:
            holding = bot_info.get('holding')
            if holding:
                # We have a pan, place it on cooker
                if self.move_towards(controller, bot_id, kx, ky):
                    if controller.place(bot_id, kx, ky):
                        bs['state'] = 2
            else:
                # Buy a pan
                shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
                if not shop_pos:
                    return
                sx, sy = shop_pos
                if self.move_towards(controller, bot_id, sx, sy):
                    if controller.get_team_money(team) >= ShopCosts.PAN.buy_cost:
                        controller.buy(bot_id, ShopCosts.PAN, sx, sy)

        # State 2: Buy meat
        elif bs['state'] == 2:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                # print(f"Bot {bot_id} state 2: No shop found!")
                return
            sx, sy = shop_pos
            adjacent = self.move_towards(controller, bot_id, sx, sy)
            # print(f"Bot {bot_id} state 2: Shop at ({sx},{sy}), adjacent={adjacent}, money={controller.get_team_money(team)}, cost={FoodType.MEAT.buy_cost}")
            if adjacent:
                if controller.get_team_money(team) >= FoodType.MEAT.buy_cost:
                    result = controller.buy(bot_id, FoodType.MEAT, sx, sy)
                    # print(f"Bot {bot_id} state 2: Buy result={result}")
                    if result:
                        bs['state'] = 3
                else:
                    # print(f"Bot {bot_id} state 2: Not enough money!")
                    pass

        # State 3: Place meat on counter for chopping
        elif bs['state'] == 3:
            # Find a counter that's not where the plate is
            chop_counter = self.find_empty_counter(controller, bx, by, exclude=bs['assembly_counter'])
            
            # If no separate counter available, check if assembly counter is empty
            if not chop_counter:
                # Check if assembly counter is actually empty
                tile = controller.get_tile(team, cx, cy)
                if tile and getattr(tile, 'item', None) is None:
                    chop_counter = bs['assembly_counter']  # Use it if empty
                else:
                    # Try to find ANY empty counter
                    chop_counter = self.find_empty_counter(controller, bx, by)
            
            if not chop_counter:
                # No empty counter available - trash and restart
                bs['state'] = 16
                return
            bs['chopping_counter'] = chop_counter
            chx, chy = bs['chopping_counter']
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.place(bot_id, chx, chy):
                    bs['state'] = 4

        # State 4: Chop meat
        elif bs['state'] == 4:
            if not bs['chopping_counter']:
                bs['state'] = 3  # Go back to find counter
                return
            chx, chy = bs['chopping_counter']
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.chop(bot_id, chx, chy):
                    bs['state'] = 5

        # State 5: Pick up chopped meat
        elif bs['state'] == 5:
            if not bs['chopping_counter']:
                bs['state'] = 3
                return
            chx, chy = bs['chopping_counter']
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.pickup(bot_id, chx, chy):
                    bs['chopping_counter'] = None  # Clear after pickup
                    bs['state'] = 6

        # State 6: Place meat in pan on cooker (starts cooking automatically)
        elif bs['state'] == 6:
            if self.move_towards(controller, bot_id, kx, ky):
                if controller.place(bot_id, kx, ky):
                    # Track that meat is in pan and needs to be retrieved later
                    bs['meat_in_pan'] = True
                    bs['meat_placed_turn'] = controller.get_turn()
                    bs['current_ingredient_index'] += 1  # Meat is now cooking
                    bs['state'] = self._get_state_for_ingredient(bs)  # Get next state (may go to state 8 for plate)

        # State 7: (skipped - cooking starts automatically)
        elif bs['state'] == 7:
            bs['state'] = 8

        # State 8: Buy plate
        elif bs['state'] == 8:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                print(f"Bot {bot_id} state 8: NO SHOP FOUND!")
                return
            sx, sy = shop_pos
            print(f"Bot {bot_id} state 8: shop at ({sx},{sy}), money={controller.get_team_money(team)}")
            if self.move_towards(controller, bot_id, sx, sy):
                print(f"Bot {bot_id} state 8: adjacent to shop, trying to buy")
                if controller.get_team_money(team) >= ShopCosts.PLATE.buy_cost:
                    if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                        bs['state'] = 9
                        print(f"Bot {bot_id} state 8: bought plate, -> state 9")
                    else:
                        print(f"Bot {bot_id} state 8: buy FAILED!")
                else:
                    print(f"Bot {bot_id} state 8: not enough money for plate")

        # State 9: Place plate on counter, then dispatch to next ingredient
        elif bs['state'] == 9:
            # Check if assembly counter is empty, if not find a new one
            tile = controller.get_tile(team, cx, cy)
            if tile and getattr(tile, 'item', None) is not None:
                # Counter is occupied, find a new empty counter
                new_counter = self.find_empty_counter(controller, bx, by)
                if new_counter:
                    bs['assembly_counter'] = new_counter
                    cx, cy = new_counter
                else:
                    # No empty counter - trash what we're holding and restart
                    bs['state'] = 16
                    return
            
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    bs['plate_ready'] = True
                    # Don't reset index - continue from where we left off after chopping
                    # In single counter mode, chopping was done first
                    bs['state'] = self._get_state_for_ingredient(bs)

        # State 10: Buy noodles
        elif bs['state'] == 10:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(team) >= FoodType.NOODLES.buy_cost:
                    if controller.buy(bot_id, FoodType.NOODLES, sx, sy):
                        bs['state'] = 11

        # State 11: Add noodles to plate, then go to next ingredient
        elif bs['state'] == 11:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    bs['current_ingredient_index'] += 1
                    bs['state'] = self._get_state_for_ingredient(bs)

        # State 12: Wait for meat to cook and take from pan
        elif bs['state'] == 12:
            # Wait at least 20 ticks for meat to cook
            current_turn = controller.get_turn()
            if bs['meat_placed_turn'] is not None:
                ticks_waited = current_turn - bs['meat_placed_turn']
                if ticks_waited < 20:
                    # Still waiting for meat to cook - move towards cooker while waiting
                    self.move_towards(controller, bot_id, kx, ky)
                    return
            
            if self.move_towards(controller, bot_id, kx, ky):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan) and tile.item.food:
                    food = tile.item.food
                    if food.cooked_stage == 1:  # Cooked
                        if controller.take_from_pan(bot_id, kx, ky):
                            bs['state'] = 13
                    elif food.cooked_stage == 2:  # Burnt
                        if controller.take_from_pan(bot_id, kx, ky):
                            bs['state'] = 16  # Go to trash
                    # else: still raw, keep waiting
                else:
                    if bot_info.get('holding'):
                        bs['state'] = 16  # Something wrong, trash it
                    elif not bs['meat_in_pan']:
                        bs['state'] = 2  # No meat in pan, restart the cycle

        # State 13: Add meat to plate, then go to next ingredient
        elif bs['state'] == 13:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    bs['meat_in_pan'] = False  # Meat has been added to plate
                    bs['meat_placed_turn'] = None  # Clear timing
                    bs['state'] = self._get_state_for_ingredient(bs)

        # State 14: Pick up the completed plate
        elif bs['state'] == 14:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    bs['state'] = 15

        # State 15: Submit the order
        elif bs['state'] == 15:
            # Check if the order is still active - if not, trash the plate and restart
            if bs['current_order']:
                order_id = bs['current_order'].get('order_id')
                orders = controller.get_orders(team)
                current_order = next((o for o in orders if o.get('order_id') == order_id), None)
                if current_order is None or not current_order.get('is_active') or current_order.get('completed_turn') is not None:
                    # Order expired or completed, trash the plate
                    bs['state'] = 16
                    return
            
            submit_pos = self.find_nearest_tile(controller, bx, by, "SUBMIT")
            if not submit_pos:
                return
            ux, uy = submit_pos
            if self.move_towards(controller, bot_id, ux, uy):
                if controller.submit(bot_id, ux, uy):
                    # Reset order tracking
                    bs['current_order'] = None
                    bs['current_order_ingredients'] = []
                    bs['current_ingredient_index'] = 0
                    bs['plate_ready'] = False
                    bs['meat_in_pan'] = False
                    bs['meat_placed_turn'] = None
                    bs['chopped_items_ready'] = []
                    bs['state'] = 0  # Start a new cycle
                else:
                    # Submit failed - order might have expired, trash and restart
                    bs['state'] = 16

        # State 16: Trash (error recovery)
        elif bs['state'] == 16:
            # If not holding anything, skip trash and go straight to state 0
            if not bot_info.get('holding'):
                bs['current_order'] = None
                bs['current_order_ingredients'] = []
                bs['current_ingredient_index'] = 0
                bs['plate_ready'] = False
                bs['meat_in_pan'] = False
                bs['meat_placed_turn'] = None
                bs['chopped_items_ready'] = []
                bs['state'] = 0
                return
            
            trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
            if not trash_pos:
                return
            tx, ty = trash_pos
            if self.move_towards(controller, bot_id, tx, ty):
                if controller.trash(bot_id, tx, ty):
                    bs['current_order'] = None
                    bs['current_order_ingredients'] = []
                    bs['current_ingredient_index'] = 0
                    bs['plate_ready'] = False
                    bs['meat_in_pan'] = False
                    bs['meat_placed_turn'] = None
                    bs['chopped_items_ready'] = []
                    bs['state'] = 0  # Restart from beginning

        # ============== EGG RECIPE STATES (cook only, no chop) ==============
        
        # State 20: Buy egg
        elif bs['state'] == 20:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                # print(f"Bot {bot_id} state 20: No shop found!")
                return
            sx, sy = shop_pos
            adjacent = self.move_towards(controller, bot_id, sx, sy)
            # print(f"Bot {bot_id} state 20: Shop at ({sx},{sy}), adjacent={adjacent}, money={controller.get_team_money(team)}, cost={FoodType.EGG.buy_cost}")
            if adjacent:
                if controller.get_team_money(team) >= FoodType.EGG.buy_cost:
                    result = controller.buy(bot_id, FoodType.EGG, sx, sy)
                    # print(f"Bot {bot_id} state 20: Buy result={result}")
                    if result:
                        bs['state'] = 21
                else:
                    # print(f"Bot {bot_id} state 20: Not enough money!")
                    pass

        # State 21: Place egg in pan on cooker (starts cooking)
        elif bs['state'] == 21:
            if self.move_towards(controller, bot_id, kx, ky):
                if controller.place(bot_id, kx, ky):
                    bs['state'] = 22

        # State 22: Wait for egg to cook and take from pan
        elif bs['state'] == 22:
            if self.move_towards(controller, bot_id, kx, ky):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan) and tile.item.food:
                    food = tile.item.food
                    if food.cooked_stage == 1:  # Cooked
                        if controller.take_from_pan(bot_id, kx, ky):
                            bs['state'] = 23
                    elif food.cooked_stage == 2:  # Burnt
                        if controller.take_from_pan(bot_id, kx, ky):
                            bs['state'] = 16  # Go to trash
                else:
                    if bot_info.get('holding'):
                        bs['state'] = 16
                    else:
                        bs['state'] = 20  # Restart egg cycle

        # State 23: Add egg to plate, then go to next ingredient
        elif bs['state'] == 23:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    bs['current_ingredient_index'] += 1
                    bs['state'] = self._get_state_for_ingredient(bs)

        # ============== ONION RECIPE STATES (chop only, no cook) ==============
        
        # State 30: Buy onions
        elif bs['state'] == 30:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(team) >= FoodType.ONIONS.buy_cost:
                    if controller.buy(bot_id, FoodType.ONIONS, sx, sy):
                        bs['state'] = 31

        # State 31: Place onions on counter for chopping
        elif bs['state'] == 31:
            # Find a counter that's not where the plate is
            chop_counter = self.find_empty_counter(controller, bx, by, exclude=bs['assembly_counter'])
            
            # If no separate counter, use assembly counter (plate should not be there yet for single-counter)
            if not chop_counter:
                chop_counter = bs['assembly_counter']
            if not chop_counter:
                chop_counter = self.find_empty_counter(controller, bx, by)  # Final fallback
            if not chop_counter:
                return
            bs['chopping_counter'] = chop_counter
            chx, chy = bs['chopping_counter']
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.place(bot_id, chx, chy):
                    bs['state'] = 32

        # State 32: Chop onions
        elif bs['state'] == 32:
            if not bs['chopping_counter']:
                bs['state'] = 31
                return
            chx, chy = bs['chopping_counter']
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.chop(bot_id, chx, chy):
                    bs['state'] = 33

        # State 33: Pickup chopped onions
        elif bs['state'] == 33:
            if not bs['chopping_counter']:
                bs['state'] = 31
                return
            chx, chy = bs['chopping_counter']
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.pickup(bot_id, chx, chy):
                    bs['chopping_counter'] = None  # Clear after pickup
                    # In single counter mode without plate, store in box and go to next ingredient
                    if bs['single_counter_mode'] and not bs['plate_ready']:
                        bs['state'] = 35  # Store onions in box temporarily
                    else:
                        bs['state'] = 34  # Add to plate

        # State 34: Add onions to plate, then go to next ingredient
        elif bs['state'] == 34:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    bs['current_ingredient_index'] += 1
                    bs['state'] = self._get_state_for_ingredient(bs)

        # State 35: Store chopped onions in box (single counter mode)
        elif bs['state'] == 35:
            box_pos = self.find_nearest_tile(controller, bx, by, "BOX")
            if not box_pos:
                # No box, try to put in trash and restart (shouldn't happen in most maps)
                return
            bxx, bxy = box_pos
            if self.move_towards(controller, bot_id, bxx, bxy):
                if controller.place(bot_id, bxx, bxy):
                    bs['chopped_items_ready'].append(('ONIONS', box_pos))
                    bs['current_ingredient_index'] += 1
                    bs['state'] = self._get_state_for_ingredient(bs)

        # ============== SAUCE RECIPE STATES (direct add, no processing) ==============
        
        # State 40: Buy sauce
        elif bs['state'] == 40:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(team) >= FoodType.SAUCE.buy_cost:
                    if controller.buy(bot_id, FoodType.SAUCE, sx, sy):
                        bs['state'] = 41

        # State 41: Add sauce to plate, then go to next ingredient
        elif bs['state'] == 41:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    bs['current_ingredient_index'] += 1
                    bs['state'] = self._get_state_for_ingredient(bs)

        # ============== RETRIEVE PRE-CHOPPED ITEMS (single counter mode) ==============
        
        # State 50: Pickup pre-chopped item from box
        elif bs['state'] == 50:
            if not bs['chopped_items_ready']:
                bs['state'] = self._get_state_for_ingredient(bs)  # No more items, continue
                return
            item_type, box_pos = bs['chopped_items_ready'][0]
            bxx, bxy = box_pos
            if self.move_towards(controller, bot_id, bxx, bxy):
                if controller.pickup(bot_id, bxx, bxy):
                    bs['chopped_items_ready'].pop(0)  # Remove from list
                    bs['state'] = 51  # Go add to plate

        # State 51: Add retrieved item to plate
        elif bs['state'] == 51:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    bs['state'] = self._get_state_for_ingredient(bs)  # Check for more items or continue

    def _get_state_for_ingredient(self, bs: Dict[str, Any]) -> int:
        """
        Returns the starting state for processing the current ingredient.
        If all ingredients are done, returns state 14 (pickup plate).
        For single counter mode, after chopping items go to buy plate state.
        """
        # If plate is ready, first check if we have pre-chopped items to retrieve
        if bs['plate_ready'] and bs['chopped_items_ready']:
            return 50  # Go retrieve pre-chopped items from box
        
        # If meat is in pan, go retrieve it (applies to all modes)
        if bs['plate_ready'] and bs['meat_in_pan']:
            return 12  # Go take meat from pan (will wait for cooking)
        
        if bs['current_ingredient_index'] >= len(bs['current_order_ingredients']):
            return 14  # All ingredients done, pickup plate
        
        ingredient = bs['current_order_ingredients'][bs['current_ingredient_index']]
        
        # For single counter mode: if plate isn't ready and we hit a non-choppable item,
        # go buy plate first
        if bs['single_counter_mode'] and not bs['plate_ready']:
            if ingredient not in ['MEAT', 'ONIONS']:  # Non-choppable items
                return 8  # Go buy plate first
        
        # Map ingredient names to their starting states
        ingredient_states = {
            'MEAT': 2,      # Buy meat -> chop -> cook -> add
            'NOODLES': 10,  # Buy noodles -> add
            'EGG': 20,      # Buy egg -> cook -> add
            'ONIONS': 30,   # Buy onions -> chop -> add
            'SAUCE': 40,    # Buy sauce -> add
        }
        
        return ingredient_states.get(ingredient, 14)  # Default to pickup if unknown
