# Grid-Based Movement in pygame
# Enhanced AI: Alpha-Beta + Iterative Deepening (HARD), Expectimax (MEDIUM)
# Includes quiescence extensions, move ordering, transposition table, and better evaluation
# Requires Pygame 2.1.3dev8+ for Vector2.move_towards

import pygame
import sys
import random
import time

# Constants
TILE_SIZE = 50
WINDOW_SIZE = 500
TREASURE_POS = (9, 9)
GRID_SIZE = WINDOW_SIZE // TILE_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

icon = pygame.image.load("images/background.png")
pygame.display.set_icon(icon)

# Game States
class GameState:
    MENU = "MENU"
    RUNNING = "RUNNING"
    WIN = "WIN"
    LOSE = "LOSE"


class Button:
    def __init__(self, text, center, size, callback, base_color, hover_color):
        self.text = text
        self.callback = callback
        self.base_color = base_color
        self.hover_color = hover_color
        self.rect = pygame.Rect(0, 0, size[0], size[1])
        self.rect.center = center
        self.is_hovered = False
        self.radius = 16

    def draw(self, surface, font):
        color = self.hover_color if self.is_hovered else self.base_color

        # Shadow
        shadow_rect = self.rect.move(0, 6)
        shadow_surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0, 0, 0, 90), shadow_surf.get_rect(), border_radius=self.radius)
        surface.blit(shadow_surf, shadow_rect.topleft)

        # Button body
        pygame.draw.rect(surface, color, self.rect, border_radius=self.radius)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2, border_radius=self.radius)

        # Text
        text_surf = font.render(self.text, True, (20, 20, 20))
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()


class Menu:
    def __init__(self, game):
        self.game = game

        # Fonts (smaller title)
        self.title_font = pygame.font.Font(None, 56)
        self.subtitle_font = pygame.font.Font(None, 26)
        self.button_font = pygame.font.Font(None, 34)

        # Background
        self.background = pygame.image.load("images/background.png").convert()
        self.background = pygame.transform.scale(self.background, (WINDOW_SIZE, WINDOW_SIZE))

        # Panel (container) rect
        self.panel_rect = pygame.Rect(0, 0, 480, 440)
        self.panel_rect.center = (WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 10)
        self.panel_radius = 28

        # Button colors (normal, hover)
        green = ((144, 238, 144), (124, 220, 124))
        blue = ((173, 216, 230), (153, 200, 214))
        red = ((255, 160, 122), (240, 140, 100))

        # Button geometry (centers inside the panel)
        cx = self.panel_rect.centerx
        top = self.panel_rect.top + 185
        w, h, step = 320, 62, 86

        self.buttons = [
            Button("EASY - Greedy Chase", (cx, top + 0 * step), (w, h),
                   lambda: self.start_game("EASY"), *green),
            Button("MEDIUM - Expectimax", (cx, top + 1 * step), (w, h),
                   lambda: self.start_game("MEDIUM"), *blue),
            Button("HARD - Alpha-Beta+ID", (cx, top + 2 * step), (w, h),
                   lambda: self.start_game("HARD"), *red),
        ]

    def start_game(self, difficulty):
        self.game.difficulty = difficulty
        self.game.state = GameState.RUNNING
        self.game.world = World(difficulty)

    def draw(self, surface):
        # Background
        surface.blit(self.background, (0, 0))

        # Panel shadow
        shadow = self.panel_rect.move(8, 8)
        shadow_surf = pygame.Surface(self.panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0, 0, 0, 110), shadow_surf.get_rect(), border_radius=self.panel_radius)
        surface.blit(shadow_surf, shadow.topleft)

        # Panel (semi-transparent white)
        panel_surf = pygame.Surface(self.panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (255, 255, 255, 215), panel_surf.get_rect(), border_radius=self.panel_radius)
        surface.blit(panel_surf, self.panel_rect.topleft)
        pygame.draw.rect(surface, (210, 210, 210), self.panel_rect, 2, border_radius=self.panel_radius)

        # Title with subtle shadow
        title = "Guard vs Thief"
        t_shadow = self.title_font.render(title, True, (0, 0, 0))
        t_text = self.title_font.render(title, True, (255, 215, 0))
        t_pos = (self.panel_rect.centerx, self.panel_rect.top + 70)
        surface.blit(t_shadow, t_shadow.get_rect(center=(t_pos[0] + 2, t_pos[1] + 2)))
        surface.blit(t_text, t_text.get_rect(center=t_pos))

        # Subtitle
        subtitle = "Steal the treasure before the pirate catches you!"
        s_surf = self.subtitle_font.render(subtitle, True, (30, 30, 30))
        surface.blit(s_surf, s_surf.get_rect(center=(self.panel_rect.centerx, self.panel_rect.top + 115)))

        # Buttons
        for b in self.buttons:
            b.draw(surface, self.button_font)

    def handle_event(self, event):
        for b in self.buttons:
            b.handle_event(event)


class Player(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int], *groups: pygame.sprite.AbstractGroup, barriers=None):
        super().__init__(*groups)
        self.image = pygame.image.load("images/thief.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (TILE_SIZE, TILE_SIZE))
        self.rect = self.image.get_rect(topleft=pos)
        self.direction = pygame.math.Vector2()
        self.pos = pygame.math.Vector2(self.rect.center)
        self.moving = False
        self.speed = 295
        self.barriers = barriers if barriers is not None else set()

    def get_input(self):
        keys = pygame.key.get_pressed()

        grid_x = int(self.pos.x // TILE_SIZE)
        grid_y = int(self.pos.y // TILE_SIZE)
        target = None
        if keys[pygame.K_UP]:
            target = (grid_x, grid_y - 1)
        elif keys[pygame.K_DOWN]:
            target = (grid_x, grid_y + 1)
        elif keys[pygame.K_LEFT]:
            target = (grid_x - 1, grid_y)
        elif keys[pygame.K_RIGHT]:
            target = (grid_x + 1, grid_y)
        if target:
            if (0 <= target[0] < GRID_SIZE and 0 <= target[1] < GRID_SIZE and target not in self.barriers):
                self.direction = pygame.math.Vector2(target[0] * TILE_SIZE + TILE_SIZE // 2,
                                                     target[1] * TILE_SIZE + TILE_SIZE // 2)
                self.moving = True
            else:
                self.direction = pygame.math.Vector2()
        else:
            self.direction = pygame.math.Vector2()

    def move(self, dt):
        if self.direction.magnitude() != 0:
            self.pos = self.pos.move_towards(self.direction, self.speed * dt)
        if self.pos == self.direction:
            self.moving = False
        self.rect.center = tuple(map(int, self.pos))

    def update(self, dt):
        if not self.moving:
            self.get_input()
        self.move(dt)


class Treasure(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int], *groups: pygame.sprite.AbstractGroup):
        super().__init__(*groups)
        self.image = pygame.image.load("images/treasure.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (TILE_SIZE, TILE_SIZE))
        self.rect = self.image.get_rect(topleft=(pos[0] * TILE_SIZE, pos[1] * TILE_SIZE))


class Guard(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int], *groups: pygame.sprite.AbstractGroup, barriers=None, forbidden=None,
                 difficulty="EASY"):
        super().__init__(*groups)
        self.image = pygame.image.load("images/guard.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (TILE_SIZE, TILE_SIZE))
        self.rect = self.image.get_rect(topleft=(pos[0] * TILE_SIZE, pos[1] * TILE_SIZE))
        self.grid_pos = pos
        self.barriers = barriers if barriers is not None else set()
        self.forbidden = forbidden if forbidden is not None else set()
        self.difficulty = difficulty

        # Search params
        self.time_budget_s = 0.020  # ~20ms per guard move on HARD
        self.quiescence_bonus = 2    # extend depth when tactically hot

        # Eval weights 
        self.W_BLOCKED = 300          
        self.W_P_PATH = -40           
        self.W_INTERCEPT = 12         
        self.W_PRESSURE = -8          
        self.W_RING = 20              
        self.W_LOS_PEN = 25           
        self.W_COVER_BONUS = 15       

        
        self._sp_cache = {}

    # Utils 
    def manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def in_bounds(self, p):
        return 0 <= p[0] < GRID_SIZE and 0 <= p[1] < GRID_SIZE

    def neighbors4(self, p):
        x, y = p
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            q = (x+dx, y+dy)
            if self.in_bounds(q):
                yield q

    def bfs_path(self, start, goal, blocked=frozenset()):
        from collections import deque
        if start == goal:
            return [start]
        seen = {start}
        q = deque([(start, None)])
        parent = {}
        while q:
            v, _ = q.popleft()
            for w in self.neighbors4(v):
                if w in seen or w in self.barriers or w in blocked:
                    continue
                seen.add(w)
                parent[w] = v
                if w == goal:
                    # reconstruct
                    path = [w]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append((w, v))
        return None

    def shortest_path_len(self, start, goal, extra_blocked=None):
        
        key = (start, goal, tuple(sorted(extra_blocked or ())))
        if key in self._sp_cache:
            return self._sp_cache[key]
        path = self.bfs_path(start, goal, blocked=frozenset(extra_blocked or ()))
        val = None if path is None else len(path)-1
        self._sp_cache[key] = val
        return val

    # Line of Sight / Raycast 
    def _bresenham_line(self, a, b):
        """Клетки по линија (вклучувајќи старт/цел) со Bresenham 4-соседи."""
        (x0, y0), (x1, y1) = a, b
        dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        out = [(x, y)]
        while (x, y) != (x1, y1):
            e2 = 2 * err
            if e2 >= dy:
                err += dy; x += sx
            if e2 <= dx:
                err += dx; y += sy
            out.append((x, y))
        return out

    def has_line_of_sight(self, a, b):
        
        for cell in self._bresenham_line(a, b):
            if cell == a or cell == b:
                continue
            if cell in self.barriers:
                return False
        return True

    def is_covered_from(self, target, observer):
        """Дали target е 'покриен' (без LoS) од observer."""
        return not self.has_line_of_sight(target, observer)

    # Move gens 
    def get_guard_moves(self, pos, player_pos):
        moves = []
        for q in self.neighbors4(pos):
            if q in self.barriers:
                continue
            # guard cannot step onto treasure (keeps it protected), unless it's the player square to capture
            if q in self.forbidden and q != player_pos:
                continue
            moves.append(q)
        return moves

    def get_player_moves(self, pos, guard_pos):
        moves = []
        for q in self.neighbors4(pos):
            if q in self.barriers:
                continue
            if q == guard_pos:
                # walking into guard would be immediate capture 
                continue
            # player IS allowed to step onto treasure
            moves.append(q)
        return moves

   
    def evaluate(self, g, p):
        # Терминали
        if g == p:
            return 1_000_000
        if p == TREASURE_POS:
            return -1_000_000

        # 1)  player to treasure
        p_path_len = self.shortest_path_len(p, TREASURE_POS, extra_blocked={g})
        if p_path_len is None:
            block_term = self.W_BLOCKED
        else:
            block_term = self.W_P_PATH * p_path_len

        # 2)колку порано guard ја среќава player-патеката до treasure
        p_path = self.bfs_path(p, TREASURE_POS)
        intercept_term = 0
        if p_path is not None:
            best_margin = 999
            for i, node in enumerate(p_path):
                margin = self.manhattan(g, node) - i  # guard до node минуси player до node
                if margin < best_margin:
                    best_margin = margin
            intercept_term = self.W_INTERCEPT * (-best_margin)

        # 3) pressure: near player
        pressure = self.W_PRESSURE * self.manhattan(g, p)

        # 4) Treasure ring: if player е near treasure, award i nie da sme blisku
        p_to_t = self.manhattan(p, TREASURE_POS)
        ring_bonus = 0
        if p_to_t <= 3:
            ring_bonus = max(0, self.W_RING * (3 - self.manhattan(g, TREASURE_POS)))

        # 5) Line-of-sight / cover
        los_guard_to_player = self.has_line_of_sight(g, p)
        cover_bonus = self.W_COVER_BONUS if not los_guard_to_player else 0

        # kazna ako player ima clean LoS to treasure 
        los_player_to_treasure = self.has_line_of_sight(p, TREASURE_POS)
        los_penalty = -self.W_LOS_PEN if los_player_to_treasure else 0

        return block_term + intercept_term + pressure + ring_bonus + cover_bonus + los_penalty

    #  Quiescence trigger
    def is_tactical(self, g, p):
        return (
            self.manhattan(g, p) <= 2
            or self.manhattan(p, TREASURE_POS) <= 2
            or self.has_line_of_sight(g, p)
        )

    # ---- Move ordering ----
    def order_guard_moves(self, moves, g, p):
        def key(m):
            # Prefer moves which => player to treasure,
            
            p_len_after = self.shortest_path_len(p, TREASURE_POS, extra_blocked={m})
            if p_len_after is None:
                pen = -999  # strongest (best for guard)
            else:
                pen = p_len_after
            los_break = 0 if self.has_line_of_sight(m, p) else -1  
            ring_dist = self.manhattan(m, TREASURE_POS)
            return (
                pen,
                los_break,
                self.manhattan(m, p),
                ring_dist
            )
        return sorted(moves, key=key)

    def order_player_moves(self, moves, g):
        def key(m):
            return (self.manhattan(m, TREASURE_POS), -self.manhattan(m, g))
        return sorted(moves, key=key)

    # Alpha-Beta with iterative deepening 
    def alphabeta(self, g, p, depth, alpha, beta, maximizing, start_time, time_budget, tt, qdepth=0):
        # Time cutoff
        if time.perf_counter() - start_time > time_budget:
            return self.evaluate(g, p), None, True  # (value, move, timed_out)

        # Terminal or depth 
        if depth == 0 or g == p or p == TREASURE_POS:
            # Quiescence: extend ONLY a limited number of extra plies
            if depth == 0 and qdepth > 0 and self.is_tactical(g, p):
                depth = 1
                qdepth -= 1
            else:
                return self.evaluate(g, p), None, False

        key = (g, p, depth, maximizing)
        if key in tt:
            val, mv, flag = tt[key]
            return val, mv, False

        if maximizing:
            best_val = float('-inf')
            best_mv = None
            moves = self.get_guard_moves(g, p)
            if not moves:
                return self.evaluate(g, p), None, False
            moves = self.order_guard_moves(moves, g, p)
            for mv in moves:
                val, _, to = self.alphabeta(mv, p, depth-1, alpha, beta, False, start_time, time_budget, tt, qdepth)
                if to:
                    return best_val if best_mv is not None else self.evaluate(g, p), best_mv, True
                if val > best_val:
                    best_val, best_mv = val, mv
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            tt[key] = (best_val, best_mv, 'EXACT')
            return best_val, best_mv, False
        else:
            best_val = float('inf')
            best_mv = None
            moves = self.get_player_moves(p, g)
            if not moves:
                return self.evaluate(g, p), None, False
            moves = self.order_player_moves(moves, g)
            for mv in moves:
                val, _, to = self.alphabeta(g, mv, depth-1, alpha, beta, True, start_time, time_budget, tt, qdepth)
                if to:
                    return best_val if best_mv is not None else self.evaluate(g, p), best_mv, True
                if val < best_val:
                    best_val, best_mv = val, mv
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            tt[key] = (best_val, best_mv, 'EXACT')
            return best_val, best_mv, False

    def move_minimax(self, player_pos):
        # reset мал локален cache за shortest_path_len во ова барање
        self._sp_cache.clear()

        start = time.perf_counter()
        time_budget = self.time_budget_s
        tt = {}

        best_move = None
        best_val = float('-inf')

        # Iterative deepening
        depth = 2
        while True:
            val, mv, to = self.alphabeta(
                self.grid_pos, player_pos, depth,
                float('-inf'), float('inf'), True,
                start, time_budget, tt, qdepth=self.quiescence_bonus
            )
            if not to and mv is not None:
                best_move, best_val = mv, val

            depth += 1
            if time.perf_counter() - start > time_budget or depth > 12:
                break

        if best_move is None:
            self.move_greedy(player_pos)
            return
        self.grid_pos = best_move
        self.rect.topleft = (best_move[0] * TILE_SIZE, best_move[1] * TILE_SIZE)

    # Expectimax (chance over player moves via softmax) 
    def expectimax(self, g, p, depth, maximizing, start_time, time_budget, cache):
        if time.perf_counter() - start_time > time_budget:
            return self.evaluate(g, p), None, True
        if depth == 0 or g == p or p == TREASURE_POS:
            return self.evaluate(g, p), None, False

        key = ('ex', g, p, depth, maximizing)
        if key in cache:
            return cache[key]

        if maximizing:
            best_val = float('-inf')
            best_mv = None
            moves = self.get_guard_moves(g, p)
            if not moves:
                return self.evaluate(g, p), None, False
            moves = self.order_guard_moves(moves, g, p)
            for mv in moves:
                val, _, to = self.expectimax(mv, p, depth-1, False, start_time, time_budget, cache)
                if to:
                    return best_val if best_mv is not None else self.evaluate(g, p), best_mv, True
                if val > best_val:
                    best_val, best_mv = val, mv
            cache[key] = (best_val, best_mv, False)
            return cache[key]
        else:
            moves = self.get_player_moves(p, g)
            if not moves:
                return self.evaluate(g, p), None, False
            logits = []
            for mv in moves:
                toward_t = -self.manhattan(mv, TREASURE_POS)
                away_g = self.manhattan(mv, g)
                logits.append(0.9*toward_t + 0.4*away_g)
            # numerical stable softmax
            m = max(logits)
            exps = [pow(2.718281828, L - m) for L in logits]
            s = sum(exps)
            probs = [e/s for e in exps]

            exp_val = 0.0
            for mv, pprob in zip(moves, probs):
                val, _, to = self.expectimax(g, mv, depth-1, True, start_time, time_budget, cache)
                if to:
                    return exp_val, None, True
                exp_val += pprob * val
            cache[key] = (exp_val, None, False)
            return cache[key]

    def move_expectimax(self, player_pos):
        # reset local cache
        self._sp_cache.clear()

        start = time.perf_counter()
        cache = {}
        time_budget = self.time_budget_s
        best_move = None
        best_val = float('-inf')

        # Iterative deepening on depth as well
        for depth in range(2, 8):
            val, mv, to = self.expectimax(self.grid_pos, player_pos, depth, True, start, time_budget, cache)
            if to:
                break
            if mv is not None:
                best_move, best_val = mv, val
            if time.perf_counter() - start > time_budget:
                break

        if best_move is None:
            self.move_greedy(player_pos)
            return
        self.grid_pos = best_move
        self.rect.topleft = (best_move[0] * TILE_SIZE, best_move[1] * TILE_SIZE)

    # Simple baselines
    def bfs_shortest_path(self, start, goal):
        # kept for EASY greedy
        return self.bfs_path(start, goal)

    def move_greedy(self, player_pos):
        path = self.bfs_shortest_path(self.grid_pos, player_pos)
        if path and len(path) > 1:
            next_pos = path[1]
            self.grid_pos = next_pos
            self.rect.topleft = (next_pos[0] * TILE_SIZE, next_pos[1] * TILE_SIZE)
        else:
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            best_move = None
            best_distance = float('inf')
            for dx, dy in moves:
                nx, ny = self.grid_pos[0] + dx, self.grid_pos[1] + dy
                nxt = (nx, ny)
                if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and nxt not in self.barriers and nxt not in self.forbidden):
                    d = self.manhattan(nxt, player_pos)
                    if d < best_distance:
                        best_distance = d
                        best_move = nxt
            if best_move:
                self.grid_pos = best_move
                self.rect.topleft = (best_move[0] * TILE_SIZE, best_move[1] * TILE_SIZE)

    def move_randomly(self, player_pos):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(moves)
        for dx, dy in moves:
            nx, ny = self.grid_pos[0] + dx, self.grid_pos[1] + dy
            nxt = (nx, ny)
            if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and nxt not in self.barriers and nxt not in self.forbidden):
                self.grid_pos = nxt
                self.rect.topleft = (nx * TILE_SIZE, ny * TILE_SIZE)
                break

    #Difficulty switch 
    def move(self, player_pos):
        if self.difficulty == "EASY":
            self.move_greedy(player_pos)
        elif self.difficulty == "MEDIUM":
            self.move_expectimax(player_pos)
        elif self.difficulty == "HARD":
            self.move_minimax(player_pos)
        else:
            self.move_randomly(player_pos)


class World:
    """
    The World class takes care of our World information.
    It contains our player and the current world.
    """

    def __init__(self, difficulty="EASY"):
        self.difficulty = difficulty
        self.barriers = set()

        # regenerate until constraints satisfied
        while True:
            new_barriers = self.generate_barriers()
            if not self.has_path((0, 0), TREASURE_POS, new_barriers):
                continue
            self.barriers = new_barriers
            guard_start = self.find_guard_start()
            if self.has_path(guard_start, (0, 0), new_barriers):
                break

        # sprites
        self.player = pygame.sprite.GroupSingle()
        Player((0, 0), self.player, barriers=self.barriers)
        self.treasure = pygame.sprite.GroupSingle()
        Treasure(TREASURE_POS, self.treasure)

        self.guard = pygame.sprite.GroupSingle()
        forbidden = {(0, 0), TREASURE_POS}
        Guard(guard_start, self.guard, barriers=self.barriers, forbidden=forbidden, difficulty=difficulty)

        self.game_won = False
        self.game_lost = False
        self.font = pygame.font.Font(None, 74)
        self.last_player_grid_pos = (0, 0)

    def find_guard_start(self):
        tx, ty = TREASURE_POS
        candidates = [
            (tx + dx, ty + dy)
            for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            if not (dx == 0 and dy == 0)
        ]
        valid = [c for c in candidates if 0 <= c[0] < GRID_SIZE and 0 <= c[1] < GRID_SIZE and c not in self.barriers and c != (0, 0)]
        if valid:
            return random.choice(valid)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) not in self.barriers and (x, y) != (0, 0) and (x, y) != TREASURE_POS:
                    return (x, y)
        return (1, 1)

    def has_path(self, start, end, barriers):
        rows = GRID_SIZE
        queue = [(start[0], start[1])]
        visited = {start}
        while queue:
            x, y = queue.pop(0)
            if (x, y) == end:
                return True
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                q = (nx, ny)
                if (0 <= nx < rows and 0 <= ny < rows and q not in barriers and q not in visited):
                    queue.append(q)
                    visited.add(q)
        return False

    def generate_barriers(self):
        rows = GRID_SIZE
        all_coords = [(x, y) for x in range(rows) for y in range(rows)]
        forbidden = {(0, 0), TREASURE_POS}
        barriers = set()

        initial_count = random.randint(7, 8)
        candidates = [c for c in all_coords if c not in forbidden]
        initial_barriers = random.sample(candidates, initial_count)
        barriers.update(initial_barriers)

        for bx, by in list(barriers):
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = bx+dx, by+dy
                if (0 <= nx < rows and 0 <= ny < rows and (nx, ny) not in barriers and (nx, ny) not in forbidden):
                    if random.randint(0, 99) >= 60:
                        barriers.add((nx, ny))
                        if not self.has_path((0, 0), TREASURE_POS, barriers):
                            barriers.remove((nx, ny))
        return barriers

    def check_win_condition(self):
        if not self.game_won and self.player.sprite:
            player_grid_pos = (
                int(self.player.sprite.pos.x // TILE_SIZE),
                int(self.player.sprite.pos.y // TILE_SIZE)
            )
            if player_grid_pos == TREASURE_POS:
                self.game_won = True
        if not self.game_lost and self.guard.sprite and self.player.sprite:
            guard_pos = self.guard.sprite.grid_pos
            player_grid_pos = (
                int(self.player.sprite.pos.x // TILE_SIZE),
                int(self.player.sprite.pos.y // TILE_SIZE)
            )
            if guard_pos == player_grid_pos:
                self.game_lost = True

    def draw_game_over_message(self):
        if self.game_won or self.game_lost:
            display = pygame.display.get_surface()
            text = self.font.render('YOU WIN!' if self.game_won else 'GAME OVER!',
                                    True, YELLOW if self.game_won else RED)
            text_rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
            display.blit(text, text_rect)

            font_small = pygame.font.Font(None, 36)
            restart_text = font_small.render('Press SPACE to return to menu', True, BLACK)
            restart_rect = restart_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 50))
            display.blit(restart_text, restart_rect)

    def update(self, dt):
        display = pygame.display.get_surface()
        self.player.update(dt)
        player_grid_pos = (
            int(self.player.sprite.pos.x // TILE_SIZE),
            int(self.player.sprite.pos.y // TILE_SIZE)
        )
        if player_grid_pos != self.last_player_grid_pos and not self.player.sprite.moving:
            self.guard.sprite.move(player_grid_pos)
            self.last_player_grid_pos = player_grid_pos
            guard_pos = self.guard.sprite.grid_pos
            if guard_pos == player_grid_pos:
                self.game_lost = True
        self.check_win_condition()
        self.player.draw(display)
        self.treasure.draw(display)
        self.guard.draw(display)
        self.draw_game_over_message()


class Game:
    """
    Initializes pygame and handles events.
    """

    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode([WINDOW_SIZE, WINDOW_SIZE], pygame.SCALED)
        pygame.display.set_caption("Guard vs Thief")
        self.clock = pygame.time.Clock()
        self.state = GameState.MENU
        self.menu = Menu(self)
        self.world = None
        self.difficulty = None
        self.running = True
        self.show_grid = True

    @staticmethod
    def draw_grid(barriers=None):
        rows = int(WINDOW_SIZE / TILE_SIZE)
        display = pygame.display.get_surface()

        tile_img = pygame.image.load("images/sand-background-tile.png").convert()
        tile_img = pygame.transform.scale(tile_img, (TILE_SIZE, TILE_SIZE))

        for y in range(rows):
            for x in range(rows):
                display.blit(tile_img, (x * TILE_SIZE, y * TILE_SIZE))

        if barriers:
            for bx, by in barriers:
                rect = pygame.Rect(bx * TILE_SIZE, by * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(display, (80, 40, 40), rect)

        gap = WINDOW_SIZE // rows
        for i in range(rows + 1):
            pygame.draw.line(display, "brown", (0, i * gap), (WINDOW_SIZE, i * gap))
            pygame.draw.line(display, "brown", (i * gap, 0), (i * gap, WINDOW_SIZE))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d and self.state == GameState.RUNNING:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_SPACE:
                    if self.world and (self.world.game_won or self.world.game_lost):
                        self.state = GameState.MENU

            if self.state == GameState.MENU:
                self.menu.handle_event(event)

    def update(self, dt):
        if self.state == GameState.MENU:
            self.menu.draw(self.window)
        elif self.state == GameState.RUNNING:
            if self.show_grid:
                self.draw_grid(getattr(self.world, 'barriers', None))
            self.world.update(dt)

            if self.world.game_won:
                self.state = GameState.WIN
            elif self.world.game_lost:
                self.state = GameState.LOSE

    def run(self):
        while self.running:
            dt = self.clock.tick() / 1000
            self.handle_events()
            self.update(dt)
            pygame.display.update()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()
