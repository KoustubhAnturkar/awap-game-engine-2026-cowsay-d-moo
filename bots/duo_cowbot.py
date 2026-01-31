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
        self.assembly_counter = None  # Counter for plate
        self.chopping_counter = None  # Counter for chopping (separate from plate)
        self.cooker_loc = None
        self.my_bot_id = None
        self.state = 0
        
        # Order tracking
        self.current_order = None  # Current order being processed
        self.current_order_ingredients = []  # List of ingredients for current order
        self.current_ingredient_index = 0  # Which ingredient we're working on
        self.plate_ready = False  # Whether plate is placed and ready for ingredients
        self.plate_temp_held = False  # True when we temporarily picked up plate for chopping
        self.held_food_for_chop = None  # Temporarily store what we're chopping when picking up plate
        self.single_counter_mode = False  # True when only one counter available
        self.chopped_items_ready = []  # Items that have been chopped and are waiting in box
        self.meat_in_pan = False  # True when meat has been placed in pan (for single counter mode)
        
    def get_bfs_path(self, controller: RobotController, start: Tuple[int, int], target_predicate) -> Optional[Tuple[int, int]]:
        """
        Use Dijkstra's algorithm to find the shortest path.
        Returns the first step (dx, dy) to take, or None if no path exists.
        Diagonal moves cost sqrt(2), orthogonal moves cost 1.
        """
        SQRT2 = 1.41421356
        
        # Priority queue: (cost, x, y, path)
        heap = [(0, start[0], start[1], [])]
        visited = {}  # Maps (x, y) -> best cost to reach it
        w, h = self.map.width, self.map.height

        while heap:
            cost, curr_x, curr_y, path = heapq.heappop(heap)
            
            # Skip if we've already found a better path to this node
            if (curr_x, curr_y) in visited and visited[(curr_x, curr_y)] < cost:
                continue
            
            tile = controller.get_tile(controller.get_team(), curr_x, curr_y)
            if target_predicate(curr_x, curr_y, tile):
                if not path:
                    return (0, 0) 
                return path[0] 

            for dx in [0, -1, 1]:
                for dy in [0, -1, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
                            # Calculate move cost: diagonal = sqrt(2), orthogonal = 1
                            move_cost = SQRT2 if (dx != 0 and dy != 0) else 1
                            new_cost = cost + move_cost
                            
                            # Only explore if we haven't found a better path
                            if (nx, ny) not in visited or visited[(nx, ny)] > new_cost:
                                visited[(nx, ny)] = new_cost
                                heapq.heappush(heap, (new_cost, nx, ny, path + [(dx, dy)]))
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

    def move_towards(self, controller: RobotController, bot_id: int, target_x: int, target_y: int) -> bool:
        """
        Move the bot towards the target position.
        Returns True if already adjacent to target, False otherwise.
        """
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
            
        bx, by = bot_state['x'], bot_state['y']
        
        def is_adjacent_to_target(x, y, tile):
            return max(abs(x - target_x), abs(y - target_y)) <= 1
        
        if is_adjacent_to_target(bx, by, None):
            return True
            
        step = self.get_bfs_path(controller, (bx, by), is_adjacent_to_target)
        if step and (step[0] != 0 or step[1] != 0):
            controller.move(bot_id, step[0], step[1])
            return False 
        return False 

    def find_nearest_tile(self, controller: RobotController, bot_x: int, bot_y: int, tile_name: str) -> Optional[Tuple[int, int]]:
        """
        Find the nearest tile of a given type using Chebyshev distance.
        """
        best_dist = 9999
        best_pos = None
        m = controller.get_map(controller.get_team())
        
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == tile_name:
                    dist = max(abs(bot_x - x), abs(bot_y - y))
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (x, y)
        return best_pos

    def find_empty_counter(self, controller: RobotController, bot_x: int, bot_y: int, exclude: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """
        Find the nearest empty counter tile, optionally excluding a specific position.
        """
        best_dist = 9999
        best_pos = None
        m = controller.get_map(controller.get_team())
        
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == "COUNTER":
                    # Skip excluded position
                    if exclude and (x, y) == exclude:
                        continue
                    actual_tile = controller.get_tile(controller.get_team(), x, y)
                    if actual_tile and getattr(actual_tile, 'item', None) is None:
                        dist = max(abs(bot_x - x), abs(bot_y - y))
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = (x, y)
        return best_pos

    def find_empty_cooker(self, controller: RobotController, bot_x: int, bot_y: int) -> Optional[Tuple[int, int]]:
        """
        Find the nearest cooker with an empty pan or no pan.
        """
        best_dist = 9999
        best_pos = None
        m = controller.get_map(controller.get_team())
        
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == "COOKER":
                    actual_tile = controller.get_tile(controller.get_team(), x, y)
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
        """
        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots:
            return
    
        # Primary bot handles cooking
        self.my_bot_id = my_bots[0]
        bot_id = self.my_bot_id
        
        bot_info = controller.get_bot_state(bot_id)
        if bot_info is None:
            return
            
        bx, by = bot_info['x'], bot_info['y']

        # Initialize locations on first run
        if self.assembly_counter is None:
            self.assembly_counter = self.find_nearest_tile(controller, bx, by, "COUNTER")
        if self.cooker_loc is None:
            self.cooker_loc = self.find_nearest_tile(controller, bx, by, "COOKER")

        if not self.assembly_counter or not self.cooker_loc:
            return

        cx, cy = self.assembly_counter
        kx, ky = self.cooker_loc

        # Error recovery: if we're holding something in certain states, trash it
        if self.state in [2, 8, 10] and bot_info.get('holding'):
            self.state = 16

        # State machine for cooking workflow
        self._execute_state(controller, bot_id, bot_info, bx, by, cx, cy, kx, ky)

        # Additional bots move randomly or assist
        for i in range(1, len(my_bots)):
            self._handle_secondary_bot(controller, my_bots[i])

    def _execute_state(self, controller: RobotController, bot_id: int, bot_info: Dict[str, Any],
                       bx: int, by: int, cx: int, cy: int, kx: int, ky: int):
        """
        Execute the current state in the cooking state machine.
        """
        team = controller.get_team()
        
        # State 0: Initialize - fetch orders and pick one to work on
        if self.state == 0:
            # Get active orders
            orders = controller.get_orders(team)
            active_orders = [o for o in orders if o.get('is_active') and o.get('completed_turn') is None]
            
            if not active_orders:
                return  # No orders to process
            
            # Pick the order expiring soonest
            active_orders.sort(key=lambda o: o.get('expires_turn', 9999))
            self.current_order = active_orders[0]
            ingredients = self.current_order.get('required', [])
            
            # Check if we have only one counter - if so, reorder to do choppable items first
            # This way we chop before placing the plate
            num_counters = sum(1 for x in range(self.map.width) for y in range(self.map.height) 
                              if self.map.tiles[x][y].tile_name == "COUNTER")
            
            if num_counters <= 1:
                # Reorder: choppable items (MEAT, ONIONS) first, then cookable (EGG), then direct (NOODLES, SAUCE)
                choppable = [i for i in ingredients if i in ['MEAT', 'ONIONS']]
                cookable_only = [i for i in ingredients if i in ['EGG']]
                direct = [i for i in ingredients if i in ['NOODLES', 'SAUCE']]
                self.current_order_ingredients = choppable + cookable_only + direct
            else:
                self.current_order_ingredients = ingredients
            
            self.current_ingredient_index = 0
            self.plate_ready = False
            self.plate_temp_held = False
            self.single_counter_mode = (num_counters <= 1)
            self.chopped_items_ready = []
            self.meat_in_pan = False
            
            # Check if we need a pan (any cookable ingredients)
            needs_pan = any(ing in ['MEAT', 'EGG'] for ing in self.current_order_ingredients)
            
            # For single counter mode with choppable items:
            # Do all chopping first, put in pan to cook, THEN buy plate
            has_choppable = any(ing in ['MEAT', 'ONIONS'] for ing in self.current_order_ingredients)
            
            if needs_pan:
                tile = controller.get_tile(team, kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan):
                    if self.single_counter_mode and has_choppable:
                        # Single counter: start with first choppable ingredient before plate
                        self.state = self._get_state_for_ingredient()
                    else:
                        self.state = 8  # Go buy plate
                else:
                    self.state = 1  # Go buy pan first
            else:
                if self.single_counter_mode and has_choppable:
                    # Single counter with choppable but no cookable - chop first
                    self.state = self._get_state_for_ingredient()
                else:
                    self.state = 8  # Go buy plate (no pan needed)

        # State 1: Buy pan and place on cooker
        elif self.state == 1:
            holding = bot_info.get('holding')
            if holding:
                # We have a pan, place it on cooker
                if self.move_towards(controller, bot_id, kx, ky):
                    if controller.place(bot_id, kx, ky):
                        self.state = 2
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
        elif self.state == 2:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(team) >= FoodType.MEAT.buy_cost:
                    if controller.buy(bot_id, FoodType.MEAT, sx, sy):
                        self.state = 3

        # State 3: Place meat on counter for chopping
        elif self.state == 3:
            # Find a counter that's not where the plate is
            chop_counter = self.find_empty_counter(controller, bx, by, exclude=self.assembly_counter)
            
            # If no separate counter available, we need to use the plate's counter
            if not chop_counter:
                # Check if plate is on assembly counter - if so, we need to do chopping BEFORE placing plate
                # For single counter scenario, do all chopping first, then buy plate
                # Since plate isn't placed yet in this order, just use the assembly counter
                chop_counter = self.assembly_counter  # Fallback to same counter
            
            if not chop_counter:
                chop_counter = self.find_empty_counter(controller, bx, by)  # Final fallback
            if not chop_counter:
                return
            self.chopping_counter = chop_counter
            chx, chy = self.chopping_counter
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.place(bot_id, chx, chy):
                    self.state = 4

        # State 4: Chop meat
        elif self.state == 4:
            if not self.chopping_counter:
                self.state = 3  # Go back to find counter
                return
            chx, chy = self.chopping_counter
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.chop(bot_id, chx, chy):
                    self.state = 5

        # State 5: Pick up chopped meat
        elif self.state == 5:
            if not self.chopping_counter:
                self.state = 3
                return
            chx, chy = self.chopping_counter
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.pickup(bot_id, chx, chy):
                    self.chopping_counter = None  # Clear after pickup
                    self.state = 6

        # State 6: Place meat in pan on cooker (starts cooking automatically)
        elif self.state == 6:
            if self.move_towards(controller, bot_id, kx, ky):
                if controller.place(bot_id, kx, ky):
                    # In single counter mode, mark that meat is in pan (to retrieve later)
                    if self.single_counter_mode and not self.plate_ready:
                        self.meat_in_pan = True
                    self.current_ingredient_index += 1  # Meat is now cooking
                    self.state = self._get_state_for_ingredient()  # Get next state (may go to state 8 for plate)

        # State 7: (skipped - cooking starts automatically)
        elif self.state == 7:
            self.state = 8

        # State 8: Buy plate
        elif self.state == 8:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(team) >= ShopCosts.PLATE.buy_cost:
                    if controller.buy(bot_id, ShopCosts.PLATE, sx, sy):
                        self.state = 9

        # State 9: Place plate on counter, then dispatch to next ingredient
        elif self.state == 9:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    self.plate_ready = True
                    # Don't reset index - continue from where we left off after chopping
                    # In single counter mode, chopping was done first
                    self.state = self._get_state_for_ingredient()

        # State 10: Buy noodles
        elif self.state == 10:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(team) >= FoodType.NOODLES.buy_cost:
                    if controller.buy(bot_id, FoodType.NOODLES, sx, sy):
                        self.state = 11

        # State 11: Add noodles to plate, then go to next ingredient
        elif self.state == 11:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.current_ingredient_index += 1
                    self.state = self._get_state_for_ingredient()

        # State 12: Wait for meat to cook and take from pan
        elif self.state == 12:
            if self.move_towards(controller, bot_id, kx, ky):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan) and tile.item.food:
                    food = tile.item.food
                    if food.cooked_stage == 1:  # Cooked
                        if controller.take_from_pan(bot_id, kx, ky):
                            self.state = 13
                    elif food.cooked_stage == 2:  # Burnt
                        if controller.take_from_pan(bot_id, kx, ky):
                            self.state = 16  # Go to trash
                else:
                    if bot_info.get('holding'):
                        self.state = 16  # Something wrong, trash it
                    else:
                        self.state = 2  # Restart the cycle

        # State 13: Add meat to plate, then go to next ingredient
        elif self.state == 13:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.meat_in_pan = False  # Meat has been added to plate
                    # Don't increment index if this was from stored meat_in_pan
                    if not self.single_counter_mode:
                        self.current_ingredient_index += 1
                    self.state = self._get_state_for_ingredient()

        # State 14: Pick up the completed plate
        elif self.state == 14:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.pickup(bot_id, cx, cy):
                    self.state = 15

        # State 15: Submit the order
        elif self.state == 15:
            submit_pos = self.find_nearest_tile(controller, bx, by, "SUBMIT")
            if not submit_pos:
                return
            ux, uy = submit_pos
            if self.move_towards(controller, bot_id, ux, uy):
                if controller.submit(bot_id, ux, uy):
                    # Reset order tracking
                    self.current_order = None
                    self.current_order_ingredients = []
                    self.current_ingredient_index = 0
                    self.plate_ready = False
                    self.meat_in_pan = False
                    self.chopped_items_ready = []
                    self.state = 0  # Start a new cycle

        # State 16: Trash (error recovery)
        elif self.state == 16:
            trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
            if not trash_pos:
                return
            tx, ty = trash_pos
            if self.move_towards(controller, bot_id, tx, ty):
                if controller.trash(bot_id, tx, ty):
                    self.state = 0  # Restart from beginning

        # ============== EGG RECIPE STATES (cook only, no chop) ==============
        
        # State 20: Buy egg
        elif self.state == 20:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(team) >= FoodType.EGG.buy_cost:
                    if controller.buy(bot_id, FoodType.EGG, sx, sy):
                        self.state = 21

        # State 21: Place egg in pan on cooker (starts cooking)
        elif self.state == 21:
            if self.move_towards(controller, bot_id, kx, ky):
                if controller.place(bot_id, kx, ky):
                    self.state = 22

        # State 22: Wait for egg to cook and take from pan
        elif self.state == 22:
            if self.move_towards(controller, bot_id, kx, ky):
                tile = controller.get_tile(controller.get_team(), kx, ky)
                if tile and isinstance(getattr(tile, 'item', None), Pan) and tile.item.food:
                    food = tile.item.food
                    if food.cooked_stage == 1:  # Cooked
                        if controller.take_from_pan(bot_id, kx, ky):
                            self.state = 23
                    elif food.cooked_stage == 2:  # Burnt
                        if controller.take_from_pan(bot_id, kx, ky):
                            self.state = 16  # Go to trash
                else:
                    if bot_info.get('holding'):
                        self.state = 16
                    else:
                        self.state = 20  # Restart egg cycle

        # State 23: Add egg to plate, then go to next ingredient
        elif self.state == 23:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.current_ingredient_index += 1
                    self.state = self._get_state_for_ingredient()

        # ============== ONION RECIPE STATES (chop only, no cook) ==============
        
        # State 30: Buy onions
        elif self.state == 30:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(team) >= FoodType.ONIONS.buy_cost:
                    if controller.buy(bot_id, FoodType.ONIONS, sx, sy):
                        self.state = 31

        # State 31: Place onions on counter for chopping
        elif self.state == 31:
            # Find a counter that's not where the plate is
            chop_counter = self.find_empty_counter(controller, bx, by, exclude=self.assembly_counter)
            
            # If no separate counter, use assembly counter (plate should not be there yet for single-counter)
            if not chop_counter:
                chop_counter = self.assembly_counter
            if not chop_counter:
                chop_counter = self.find_empty_counter(controller, bx, by)  # Final fallback
            if not chop_counter:
                return
            self.chopping_counter = chop_counter
            chx, chy = self.chopping_counter
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.place(bot_id, chx, chy):
                    self.state = 32

        # State 32: Chop onions
        elif self.state == 32:
            if not self.chopping_counter:
                self.state = 31
                return
            chx, chy = self.chopping_counter
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.chop(bot_id, chx, chy):
                    self.state = 33

        # State 33: Pickup chopped onions
        elif self.state == 33:
            if not self.chopping_counter:
                self.state = 31
                return
            chx, chy = self.chopping_counter
            if self.move_towards(controller, bot_id, chx, chy):
                if controller.pickup(bot_id, chx, chy):
                    self.chopping_counter = None  # Clear after pickup
                    # In single counter mode without plate, store in box and go to next ingredient
                    if self.single_counter_mode and not self.plate_ready:
                        self.state = 35  # Store onions in box temporarily
                    else:
                        self.state = 34  # Add to plate

        # State 34: Add onions to plate, then go to next ingredient
        elif self.state == 34:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.current_ingredient_index += 1
                    self.state = self._get_state_for_ingredient()

        # State 35: Store chopped onions in box (single counter mode)
        elif self.state == 35:
            box_pos = self.find_nearest_tile(controller, bx, by, "BOX")
            if not box_pos:
                # No box, try to put in trash and restart (shouldn't happen in most maps)
                return
            bxx, bxy = box_pos
            if self.move_towards(controller, bot_id, bxx, bxy):
                if controller.place(bot_id, bxx, bxy):
                    self.chopped_items_ready.append(('ONIONS', box_pos))
                    self.current_ingredient_index += 1
                    self.state = self._get_state_for_ingredient()

        # ============== SAUCE RECIPE STATES (direct add, no processing) ==============
        
        # State 40: Buy sauce
        elif self.state == 40:
            shop_pos = self.find_nearest_tile(controller, bx, by, "SHOP")
            if not shop_pos:
                return
            sx, sy = shop_pos
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(team) >= FoodType.SAUCE.buy_cost:
                    if controller.buy(bot_id, FoodType.SAUCE, sx, sy):
                        self.state = 41

        # State 41: Add sauce to plate, then go to next ingredient
        elif self.state == 41:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.current_ingredient_index += 1
                    self.state = self._get_state_for_ingredient()

        # ============== RETRIEVE PRE-CHOPPED ITEMS (single counter mode) ==============
        
        # State 50: Pickup pre-chopped item from box
        elif self.state == 50:
            if not self.chopped_items_ready:
                self.state = self._get_state_for_ingredient()  # No more items, continue
                return
            item_type, box_pos = self.chopped_items_ready[0]
            bxx, bxy = box_pos
            if self.move_towards(controller, bot_id, bxx, bxy):
                if controller.pickup(bot_id, bxx, bxy):
                    self.chopped_items_ready.pop(0)  # Remove from list
                    self.state = 51  # Go add to plate

        # State 51: Add retrieved item to plate
        elif self.state == 51:
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.add_food_to_plate(bot_id, cx, cy):
                    self.state = self._get_state_for_ingredient()  # Check for more items or continue

    def _get_state_for_ingredient(self) -> int:
        """
        Returns the starting state for processing the current ingredient.
        If all ingredients are done, returns state 14 (pickup plate).
        For single counter mode, after chopping items go to buy plate state.
        """
        # If plate is ready, first check if we have pre-chopped items to retrieve
        if self.plate_ready and self.chopped_items_ready:
            return 50  # Go retrieve pre-chopped items from box
        
        # If plate is ready and meat is in pan, go retrieve it
        if self.plate_ready and self.meat_in_pan:
            return 12  # Go take meat from pan
        
        if self.current_ingredient_index >= len(self.current_order_ingredients):
            return 14  # All ingredients done, pickup plate
        
        ingredient = self.current_order_ingredients[self.current_ingredient_index]
        
        # For single counter mode: if plate isn't ready and we hit a non-choppable item,
        # go buy plate first
        if self.single_counter_mode and not self.plate_ready:
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

    def _handle_secondary_bot(self, controller: RobotController, bot_id: int):
        """
        Handle secondary bots - they can assist with simpler tasks or move randomly.
        """
        bot_info = controller.get_bot_state(bot_id)
        if bot_info is None:
            return
            
        bx, by = bot_info['x'], bot_info['y']
        holding = bot_info.get('holding')
        
        # If holding something, try to trash it
        if holding:
            trash_pos = self.find_nearest_tile(controller, bx, by, "TRASH")
            if trash_pos:
                tx, ty = trash_pos
                if self.move_towards(controller, bot_id, tx, ty):
                    controller.trash(bot_id, tx, ty)
                return
        
        # Otherwise, move randomly to stay out of the way
        possible_moves = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = bx + dx, by + dy
                if controller.get_map(controller.get_team()).is_tile_walkable(nx, ny):
                    possible_moves.append((dx, dy))
        
        if possible_moves:
            dx, dy = random.choice(possible_moves)
            controller.move(bot_id, dx, dy)
