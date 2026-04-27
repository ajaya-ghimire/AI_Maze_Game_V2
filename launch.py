import os
import sys
import subprocess
import importlib.util
import pygame

# ── locate game file (same folder as this script) ────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_GAME_FILE = os.path.join(_HERE, "final_mazeT.py")
_AUTH_FILE = os.path.join(_HERE, "face_auth.py")

if not os.path.exists(_GAME_FILE):
    print(f"ERROR: Cannot find final_mazeT.py next to launch.py")
    print(f"       Expected at: {_GAME_FILE}")
    sys.exit(1)

if not os.path.exists(_AUTH_FILE):
    print(f"ERROR: Cannot find face_auth.py next to launch.py")
    print(f"       Expected at: {_AUTH_FILE}")
    sys.exit(1)


def _user_stats_file(username):
    """Return a per-user stats JSON filename."""
    if not username:
        return "ai_maze_stats_guest.json"
    safe = "".join(c if c.isalnum() else "_" for c in username.strip().lower())
    return f"ai_maze_stats_{safe}.json"


def main():
    # ── initialise Pygame (shared between login and game) ─────────────────────
    pygame.init()
    screen = pygame.display.set_mode((1280, 780))
    pygame.display.set_caption("AI Maze  —  Login")
    clock  = pygame.time.Clock()

    # ── run face auth ─────────────────────────────────────────────────────────
    from face_auth import FaceAuthSystem
    auth = FaceAuthSystem()
    try:
        username = auth.run(screen, clock)   # str or None
    finally:
        auth.shutdown()

    # ── tell the user what's happening ───────────────────────────────────────
    display_name = username if username else "Guest"
    print(f"[launch] Logged in as: {display_name}")

    # ── inject per-user save-file into the game module before it runs ─────────
    stats_file = _user_stats_file(username)
    print(f"[launch] Using stats file: {stats_file}")

    # Load final_mazeT as a module (keeps the same process / Pygame window)
    spec = importlib.util.spec_from_file_location("maze_game", _GAME_FILE)
    game_mod = importlib.util.module_from_spec(spec)
    sys.modules["maze_game"] = game_mod

    # Patch SAVE_FILE before the module executes
    game_mod.SAVE_FILE = stats_file   # module attr set before exec
    spec.loader.exec_module(game_mod)  # this runs the game's top-level code

    # After exec_module the module's globals are set — override SAVE_FILE
    # (some code paths may have already grabbed the default; this covers the rest)
    game_mod.SAVE_FILE = stats_file

    # Update the window caption with the player name
    pygame.display.set_caption(f"AI Maze  —  {display_name}")

    # ── monkey-patch build_menu / draw_menu to show player badge ─────────────
    Game = game_mod.Game
    _orig_build_menu = Game.build_menu
    _orig_draw_menu  = Game.draw_menu

    def _patched_build_menu(self):
        _orig_build_menu(self)
        self._launch_username = display_name   # stash for draw_menu

    def _patched_draw_menu(self, events):
        _orig_draw_menu(self, events)
        # Player badge — top-right corner
        f = self.f_ui
        badge_txt = f"\u25b6  {self._launch_username}" if hasattr(self, "_launch_username") else ""
        if badge_txt:
            W = self.screen.get_width()
            col = (30, 210, 240) if username else (90, 105, 125)
            lbl = f.render(badge_txt, True, col)
            self.screen.blit(lbl, (W - lbl.get_width() - 22, 20))
        pygame.display.flip()

    Game.build_menu = _patched_build_menu
    Game.draw_menu  = _patched_draw_menu

    # ── launch the game (calls Game().run() via __main__ guard) ───────────────
    # The game's __main__ block has already been skipped because __name__ !=
    # "__main__" for the loaded module.  We call it directly instead:
    game_mod.Game().run()


if __name__ == "__main__":
    main()
