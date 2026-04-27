import pygame
import random
import math
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Optional

# ═══════════════════════════════════════════════════
#  FACE RECOGNITION
# ═══════════════════════════════════════════════════
FACE_ENCODINGS_FILE = "face_encodings.json"

def _try_import_face():
    try:
        import face_recognition
        import cv2
        import numpy as np
        return face_recognition, cv2, np
    except ImportError:
        return None, None, None

def face_recognition_available() -> bool:
    fr, _, _ = _try_import_face()
    return fr is not None

def _load_encodings_db():
    fr, cv2, np = _try_import_face()
    if fr is None or not os.path.exists(FACE_ENCODINGS_FILE):
        return {}
    try:
        with open(FACE_ENCODINGS_FILE) as f:
            raw = json.load(f)
        return {k: [np.array(e) for e in v] for k, v in raw.items()}
    except Exception:
        return {}

def _save_encodings_db(db: dict):
    fr, cv2, np = _try_import_face()
    if np is None:
        return
    serialisable = {k: [e.tolist() for e in v] for k, v in db.items()}
    with open(FACE_ENCODINGS_FILE, "w") as f:
        json.dump(serialisable, f)

def face_login(timeout_frames: int = 150) -> Optional[str]:
    """Scan camera, return matched player name or None.

    Uses a small downscaled frame for fast HOG detection and only runs
    face_recognition every few grabbed frames so the thread finishes in
    ~5-8 s instead of several minutes.
    """
    fr, cv2, np = _try_import_face()
    if fr is None:
        return None
    db = _load_encodings_db()
    if not db:
        return None
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return None
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        result      = None
        grab_count  = 0
        SKIP        = 4          # run recognition every N grabs (skip cheap reads)
        MAX_RECOG   = 20         # max recognition attempts (~10-15 s wall-clock)
        recog_count = 0
        import time as _time
        deadline = _time.monotonic() + 12.0   # hard 12-second wall-clock cap
        while recog_count < MAX_RECOG and _time.monotonic() < deadline:
            ret, frame = cam.read()
            if not ret:
                break
            grab_count += 1
            if grab_count % SKIP != 0:
                continue          # grab-and-discard to keep camera buffer fresh
            recog_count += 1
            # Shrink to half size — HOG is O(n²) in pixels, 2x faster at 0.5x
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = fr.face_locations(rgb, model="hog")
            if not locations:
                continue
            for enc in fr.face_encodings(rgb, locations):
                for name, known in db.items():
                    if any(fr.compare_faces(known, enc, tolerance=0.52)):
                        result = name
                        break
                if result:
                    break
            if result:
                break
        cam.release()
        return result
    except Exception:
        return None

def register_face(name: str, capture_frames: int = 5) -> bool:
    """Capture face from camera and save under name. Returns True on success."""
    fr, cv2, np = _try_import_face()
    if fr is None:
        return False
    db = _load_encodings_db()
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return False
        captured = []
        for _ in range(300):
            ret, frame = cam.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = fr.face_locations(rgb, model="hog")
            if locs:
                encs = fr.face_encodings(rgb, locs)
                if encs:
                    captured.append(encs[0])
                    if len(captured) >= capture_frames:
                        break
        cam.release()
        if not captured:
            return False
        db[name] = db.get(name, []) + captured
        _save_encodings_db(db)
        return True
    except Exception:
        return False

# ═══════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════
WIDTH, HEIGHT = 1280, 780
FPS           = 60
PLAYERS_FILE  = "ai_maze_players.json"
SAVE_FILE     = "ai_maze_stats.json"
CELL_MIN, CELL_MAX = 12, 28

BG        = (6,   6,  12)
PANEL     = (10,  10,  22)
TEXT      = (220, 228, 240)
MUTED     = (90,  105, 125)
PURPLE    = (130,  55, 240)
BLUE      = ( 50, 125, 250)
CYAN      = ( 30, 210, 240)
GREEN     = ( 30, 200,  90)
GREEN_DIM = (  6,  55,  22)
RED       = (240,  60,  60)
ORANGE    = (250, 110,  20)
YELLOW    = (248, 160,   8)
GOLD      = (255, 200,  40)
WALL_C    = ( 18,  28,  55)
WALL_LIT  = ( 32,  48,  88)
FLOOR_C   = ( 11,  11,  24)
FOG_C     = (  2,   2,   5)
SPOTLIGHT_COL = (255, 240, 200)

MODE_NAMES = {
    "maze":     "LOST IN A MAZE",
    "survival": "SURVIVAL",
    "nightout": "NIGHT OUT",
}

# ═══════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════
def clamp(x, a, b): return max(a, min(b, x))
def lerp(a, b, t):  return a + (b - a) * t
def manhattan(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def blend(c1, c2, t):
    t = clamp(t, 0, 1)
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))

# ═══════════════════════════════════════════════════
#  SOUND
# ═══════════════════════════════════════════════════
class SoundSystem:
    def __init__(self):
        self.enabled  = False
        self.music_on = True
        self._cache: Dict[str, object] = {}
        self._music_ch = None
        self._cur_track: Optional[str] = None
        try:
            pygame.mixer.pre_init(44100, -16, 2, 1024)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(16)
            self.enabled = True
            self._music_ch = pygame.mixer.Channel(15)
        except Exception:
            pass

    def _gen(self, name):
        try:
            import numpy as np
            sr = 44100
            def make(dur, fn):
                t = np.linspace(0, dur, int(sr*dur), False)
                w = np.clip(fn(t), -32000, 32000).astype(np.int16)
                return pygame.sndarray.make_sound(np.column_stack([w, w]))
            recipes = {
                "move":        (0.030, lambda t: np.sin(2*np.pi*420*t)*np.exp(-t*110)*4000),
                "wall":        (0.06,  lambda t: np.sin(2*np.pi*160*t+np.sin(60*t)*3)*np.exp(-t*70)*5500),
                "coin":        (0.18,  lambda t: np.sin(2*np.pi*(600+500*t)*t)*np.exp(-t*22)*12000),
                "damage":      (0.30,  lambda t: np.sin(2*np.pi*110*t+5*np.sin(14*t))*np.exp(-t*10)*15000),
                "exit":        (0.60,  lambda t: np.sin(2*np.pi*(400+300*np.sin(4*t))*t)*np.exp(-t*3.5)*14000),
                "death":       (0.80,  lambda t: np.sin(2*np.pi*80/(1+t*3)*t)*np.exp(-t*4)*16000),
                "spawn":       (0.10,  lambda t: np.sin(2*np.pi*200*t)*(1-t/0.10)*5000),
                "heal":        (0.35,  lambda t: (np.sin(2*np.pi*520*t)+np.sin(2*np.pi*780*t)*0.5)*np.exp(-t*8)*9000),
                "levelup":     (0.70,  lambda t: np.sin(2*np.pi*(300+500*t/0.70)*t)*np.exp(-t*2)*13000),
                "wave_alert":  (0.45,  lambda t: np.sin(2*np.pi*220*t)*(np.sin(2*np.pi*3*t)*0.5+0.5)*np.exp(-t*4)*12000),
                "attack_miss": (0.12,  lambda t: (np.random.default_rng(0).random(len(t))*2-1)*np.exp(-t*40)*6000),
                "menu_click":  (0.07,  lambda t: np.sin(2*np.pi*700*t)*np.exp(-t*80)*7000),
                "powerup":     (0.50,  lambda t: np.sin(2*np.pi*(250+600*(t/0.50)**2)*t)*np.exp(-t*3)*13000),
                "wave_clear":  (0.55,  lambda t: (np.sin(2*np.pi*440*t)+np.sin(2*np.pi*550*t)+np.sin(2*np.pi*660*t))*np.exp(-t*4)*5000),
                "attack_hit":  (0.18,  lambda t: (np.sin(2*np.pi*140*t)+np.random.default_rng(1).random(len(t))*0.3)*np.exp(-t*18)*13000),
            }
            if name not in recipes: return None
            dur, fn = recipes[name]
            return make(dur, fn)
        except Exception:
            return None

    def _gen_music(self, track: str):
        try:
            import numpy as np
            sr = 44100; dur = 4.0; n = int(sr*dur)
            def note(freq, start, end, amp=1.0, shape="sine"):
                env = np.zeros(n)
                s, e = int(start*sr), int(end*sr); ln = e-s
                if ln <= 0: return env
                tt = np.linspace(0, end-start, ln, False)
                atk = min(int(0.02*sr), ln//4); rel = min(int(0.04*sr), ln//3)
                seg = np.ones(ln)
                seg[:atk] = np.linspace(0,1,atk); seg[-rel:] = np.linspace(1,0,rel)
                if shape=="sine":   w = np.sin(2*np.pi*freq*tt)
                elif shape=="square": w = np.sign(np.sin(2*np.pi*freq*tt))*0.4
                elif shape=="saw":  w = (2*(freq*tt%1)-1)*0.5
                else:               w = np.sin(2*np.pi*freq*tt)
                env[s:e] = w*seg*amp; return env
            mix = np.zeros(n)
            if track == "menu":
                notes=[261.6,329.6,392.0,523.3,392.0,329.6]; step=dur/len(notes)
                for i,f in enumerate(notes):
                    mix+=note(f,i*step,(i+1)*step-0.04,0.35)
                    mix+=note(f*2,i*step+0.05,(i+1)*step-0.02,0.12)
                mix+=note(65.4,0,dur,0.18)
            elif track == "maze":
                pat=[220,261.6,329.6,261.6,220,196,220,261.6]; step=dur/len(pat)
                for i,f in enumerate(pat):
                    mix+=note(f,i*step,(i+1)*step-0.03,0.28)
                    mix+=note(f*2,i*step+0.02,(i+1)*step-0.02,0.10)
                bf=[110,110,98,110]; bs=dur/len(bf)
                for i,f in enumerate(bf): mix+=note(f,i*bs,i*bs+0.18,0.30,"square")
                for f in [220,277.2,329.6]: mix+=note(f,0,dur,0.06)
            elif track == "survival":
                for kt in [0,.5,1,1.5,2,2.5,3,3.5]:
                    mix+=note(80,kt,kt+0.12,0.50); mix+=note(55,kt,kt+0.20,0.30)
                bass=[146.8,146.8,130.8,146.8,164.8,146.8,130.8,110.0]; bs=dur/len(bass)
                for i,f in enumerate(bass): mix+=note(f,i*bs,(i+1)*bs-0.04,0.28,"square")
                mel=[293.7,329.6,293.7,261.6,293.7,329.6,349.2,293.7]; ms=dur/len(mel)
                for i,f in enumerate(mel): mix+=note(f,i*ms,(i+1)*ms-0.03,0.16,"saw")
            elif track == "nightout":
                mix+=note(82.4,0,dur,0.14); mix+=note(164.8,0,dur,0.07)
                for st,f in [(0,164.8),(.75,196),(1.5,185),(2.25,174.6),(3,164.8),(3.5,155.6)]:
                    mix+=note(f,st,st+0.5,0.22)
                for i in range(8): mix+=note(1320+i*40,i*0.5,i*0.5+0.08,0.06)
            pk=np.max(np.abs(mix))
            if pk>0: mix=mix/pk*18000
            mix=np.clip(mix,-32000,32000).astype(np.int16)
            return pygame.sndarray.make_sound(np.column_stack([mix,mix]))
        except Exception:
            return None

    def play_music(self, track: str):
        if not self.enabled or not self.music_on: return
        if track == self._cur_track: return
        self._cur_track = track
        key = f"_music_{track}"
        if key not in self._cache: self._cache[key] = self._gen_music(track)
        s = self._cache.get(key)
        if s and self._music_ch:
            try: self._music_ch.set_volume(0.45); self._music_ch.play(s, loops=-1)
            except Exception: pass

    def stop_music(self):
        if self._music_ch:
            try: self._music_ch.stop()
            except Exception: pass
        self._cur_track = None

    def play(self, name):
        if not self.enabled: return
        if name not in self._cache: self._cache[name] = self._gen(name)
        s = self._cache.get(name)
        if s:
            try: s.play()
            except Exception: pass

# ═══════════════════════════════════════════════════
#  A*
# ═══════════════════════════════════════════════════
def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = {start}; came = {}
    g = {start: 0}; f = {start: manhattan(start, goal)}
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    while open_set:
        cur = min(open_set, key=lambda n: f.get(n, 1e9))
        if cur == goal:
            path = []
            while cur in came: path.append(cur); cur = came[cur]
            path.reverse(); return path
        open_set.remove(cur)
        for dr,dc in dirs:
            nr,nc = cur[0]+dr, cur[1]+dc
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==0:
                nxt=(nr,nc); tg=g[cur]+1
                if tg < g.get(nxt,1e9):
                    came[nxt]=cur; g[nxt]=tg
                    f[nxt]=tg+manhattan(nxt,goal); open_set.add(nxt)
    return []

# ═══════════════════════════════════════════════════
#  MAZE GENERATION
# ═══════════════════════════════════════════════════
def generate_maze(rows, cols, loop_factor=0.0):
    if rows%2==0: rows+=1
    if cols%2==0: cols+=1
    sys.setrecursionlimit(max(10000, rows*cols*2))
    grid    = [[1]*cols for _ in range(rows)]
    visited = [[False]*cols for _ in range(rows)]
    def carve(r, c):
        visited[r][c]=True; grid[r][c]=0
        dirs=[(0,2),(0,-2),(2,0),(-2,0)]; random.shuffle(dirs)
        for dr,dc in dirs:
            nr,nc=r+dr,c+dc
            if 0<nr<rows-1 and 0<nc<cols-1 and not visited[nr][nc]:
                grid[r+dr//2][c+dc//2]=0; carve(nr,nc)
    carve(1,1); grid[1][1]=grid[rows-2][cols-2]=0
    if loop_factor>0:
        walls=[(r,c) for r in range(1,rows-1) for c in range(1,cols-1) if grid[r][c]==1]
        random.shuffle(walls)
        for r,c in walls[:int(len(walls)*loop_factor)]: grid[r][c]=0
    return grid

def generate_open_field(rows, cols, obstacle_rate=0.12):
    grid=[[0]*cols for _ in range(rows)]
    for r in range(rows): grid[r][0]=grid[r][cols-1]=1
    for c in range(cols): grid[0][c]=grid[rows-1][c]=1
    for _ in range(int(rows*cols*obstacle_rate)):
        r=random.randint(1,rows-2); c=random.randint(1,cols-2); grid[r][c]=1
    cr,cc=rows//2,cols//2
    for r in range(cr-3,cr+4):
        for c in range(cc-3,cc+4):
            if 0<=r<rows and 0<=c<cols: grid[r][c]=0
    return grid

# ═══════════════════════════════════════════════════
#  ADAPTIVE AI
# ═══════════════════════════════════════════════════
@dataclass
class LevelResult:
    mode: str; level: int; time_s: int
    wrong: int; efficiency: float; score: int

@dataclass
class StatsData:
    history: List[LevelResult] = field(default_factory=list)
    avg_time: float=30.0; avg_wrong: float=5.0
    avg_eff: float=0.5;   levels: int=0

def compute_profile(stats):
    if stats.levels < 2: return "Beginner"
    if stats.avg_eff > 0.80 and stats.avg_time < 20: return "SpeedRunner"
    if stats.avg_wrong < 3  and stats.avg_eff > 0.70: return "LogicalPlanner"
    if stats.avg_wrong > 12: return "Explorer"
    return "Balanced"

PROFILE_COLORS = {
    "Beginner":       (90,180,90),
    "Balanced":       (50,130,250),
    "SpeedRunner":    (245,158,11),
    "LogicalPlanner": (130,55,240),
    "Explorer":       (30,210,240),
}

def difficulty_params(profile, level):
    base = {
        "Beginner":       dict(rows=17,cols=19,enemies=0,fog=False,speed=1.0),
        "Balanced":       dict(rows=19,cols=21,enemies=1,fog=False,speed=1.0),
        "SpeedRunner":    dict(rows=21,cols=23,enemies=2,fog=False,speed=1.5),
        "LogicalPlanner": dict(rows=21,cols=23,enemies=1,fog=True, speed=1.0),
        "Explorer":       dict(rows=23,cols=25,enemies=2,fog=True, speed=0.9),
    }.get(profile, dict(rows=19,cols=21,enemies=1,fog=False,speed=1.0))
    grow=(level-1)//2*2
    return dict(
        rows    = min(base["rows"]+grow, 33),
        cols    = min(base["cols"]+grow, 37),
        enemies = min(base["enemies"]+level//3, 4),
        speed   = base["speed"]+(level-1)*0.08,
        fog     = level>=5,
    )

class AdaptiveDirector:
    def __init__(self):
        self.skill=50.0; self.difficulty=1.0
        self.recent_damage=self.recent_kills=self.recent_coins=0
        self.time_alive=0.0; self._history: List[float]=[]
        self.death_streak=0; self.wave_streak=0; self.runs_this_session=0

    def reset(self):
        ds,ws,runs,sk,di,hi=(self.death_streak,self.wave_streak,
            self.runs_this_session,self.skill,self.difficulty,self._history)
        self.__init__()
        self.death_streak=ds; self.wave_streak=ws; self.runs_this_session=runs
        self.skill=sk; self.difficulty=di; self._history=hi

    def tick(self,dt): self.time_alive+=dt
    def report_damage(self,n=1): self.recent_damage+=n
    def report_kill(self,n=1):   self.recent_kills+=n
    def report_coin(self,n=1):   self.recent_coins+=n

    def report_death(self):
        self.death_streak+=1; self.wave_streak=0; self.runs_this_session+=1
        self.skill=clamp(self.skill-(12+self.death_streak*6),0,100); self._commit()

    def report_wave_cleared(self):
        self.wave_streak+=1; self.death_streak=0
        self.skill=clamp(self.skill+(8+self.wave_streak*4),0,100); self._commit()

    def _commit(self):
        target=0.40+(self.skill/100.0)*1.20
        self.difficulty=lerp(self.difficulty,target,0.35)
        self._history.append(self.difficulty)
        if len(self._history)>80: self._history.pop(0)

    def update_every(self):
        delta=self.recent_kills*3+self.recent_coins*1.5-self.recent_damage*5
        self.skill=clamp(lerp(self.skill,self.skill+delta,0.30),0,100)
        self._commit()
        self.recent_damage=self.recent_kills=self.recent_coins=0

    @property
    def diff_history(self): return self._history

    @property
    def ease_label(self):
        if self.skill<25:  return "EASY MODE"
        if self.skill<45:  return "EASING..."
        if self.skill<60:  return "BALANCED"
        if self.skill<80:  return "RAMPING"
        return "HARD MODE"

# ═══════════════════════════════════════════════════
#  SAVE / LOAD
# ═══════════════════════════════════════════════════
def _load_all_players() -> dict:
    if os.path.exists(PLAYERS_FILE):
        try:
            with open(PLAYERS_FILE) as f: return json.load(f)
        except Exception: pass
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE) as f: data=json.load(f)
            if data.get("history"): return {"Player 1": {"history": data["history"]}}
        except Exception: pass
    return {}

def _save_all_players(db: dict):
    with open(PLAYERS_FILE, "w") as f: json.dump(db, f, indent=2)

def load_stats_for(name: str, db: dict) -> StatsData:
    data=db.get(name, {})
    history=[LevelResult(**x) for x in data.get("history", [])]
    s=StatsData(history=history); _rollup(s); return s

def save_stats_for(name: str, stats: StatsData, db: dict):
    db[name]={"history": [asdict(h) for h in stats.history]}
    _save_all_players(db)

def _rollup(s):
    if not s.history:
        s.avg_time=30.0; s.avg_wrong=5.0; s.avg_eff=0.5; s.levels=0; return
    n=len(s.history); s.levels=n
    s.avg_time =sum(h.time_s     for h in s.history)/n
    s.avg_wrong=sum(h.wrong      for h in s.history)/n
    s.avg_eff  =sum(h.efficiency for h in s.history)/n

# ═══════════════════════════════════════════════════
#  UI PRIMITIVES
# ═══════════════════════════════════════════════════
class Button:
    def __init__(self,rect,text,primary=False,color=None):
        self.rect=pygame.Rect(rect); self.text=text
        self.primary=primary; self._color=color
        self.hover=False; self._anim=0.0

    def draw(self,surf,font):
        if self._color:
            bg=blend(self._color,(0,0,0),0.65); bgh=self._color; bord=self._color
        elif self.primary:
            bg=(60,15,150); bgh=(95,35,215); bord=(140,65,255)
        else:
            bg=(16,15,38); bgh=(28,25,70); bord=(42,40,82)
        self._anim=lerp(self._anim,1.0 if self.hover else 0.0,0.20)
        pygame.draw.rect(surf,blend(bg,bgh,self._anim),self.rect,border_radius=12)
        pygame.draw.rect(surf,bord,self.rect,2,border_radius=12)
        lbl=font.render(self.text,True,TEXT)
        surf.blit(lbl,lbl.get_rect(center=self.rect.center))

    def handle(self,events):
        self.hover=self.rect.collidepoint(pygame.mouse.get_pos())
        return any(e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self.hover for e in events)

class Particle:
    def __init__(self,x,y,vx,vy,color,life):
        self.x=float(x); self.y=float(y); self.vx=vx; self.vy=vy
        self.color=color; self.life=self.max_life=life
    def update(self,dt):
        self.x+=self.vx*dt; self.y+=self.vy*dt; self.vy+=55*dt; self.life-=dt
        return self.life>0
    def draw(self,surf):
        t=self.life/self.max_life
        pygame.draw.circle(surf,blend((0,0,0),self.color,t),(int(self.x),int(self.y)),max(1,int(t*4)))

# ═══════════════════════════════════════════════════
#  ENEMY FSM
# ═══════════════════════════════════════════════════
class FSMEnemy:
    def __init__(self,r,c,kind,speed,color):
        self.r=r; self.c=c; self.kind=kind; self.speed=speed; self.color=color
        self.move_timer=0.0; self.state="WANDER"
        self.wander_path=[]; self.patrol_path=[]; self.patrol_idx=0
        self.stun_timer=0.0; self.flash_timer=0.0
        self.visual_x=float(c); self.visual_y=float(r)

    def update_visual(self,dt):
        self.visual_x=lerp(self.visual_x,float(self.c),min(1.0,dt*14))
        self.visual_y=lerp(self.visual_y,float(self.r),min(1.0,dt*14))
        if self.flash_timer>0: self.flash_timer-=dt

    def get_draw_color(self):
        if self.flash_timer>0: return blend(self.color,(255,255,255),min(1.0,self.flash_timer*6))
        if self.state=="CHASE":  return blend(self.color,RED,0.35)
        if self.state=="WANDER": return blend(self.color,(30,30,30),0.35)
        if self.state=="PATROL": return blend(self.color,YELLOW,0.25)
        return self.color

    def decide_state(self,player,difficulty):
        d=manhattan((self.r,self.c),player)
        if self.stun_timer>0: self.state="WANDER"; return
        if self.kind=="hunter":
            self.state="ATTACK" if d<=2 else ("CHASE" if d<=int(9*difficulty) else "WANDER")
        elif self.kind=="blocker":
            self.state="CHASE" if d<=14 else "PATROL"
        elif self.kind=="zombie":
            self.state="CHASE" if d<=int(6*difficulty) else "WANDER"
        elif self.kind=="sentinel":
            self.state="ATTACK" if d<=3 else ("CHASE" if d<=7 else "PATROL")

    def step(self,grid,player,exit_pos,dt,difficulty):
        if self.stun_timer>0: self.stun_timer-=dt; return
        self.move_timer+=dt
        if self.move_timer < max(0.07,0.30/(self.speed*difficulty)): return
        self.move_timer=0.0
        rows,cols=len(grid),len(grid[0])
        if self.state in ("CHASE","ATTACK"):
            target=player
            if self.kind=="blocker" and exit_pos!=(-1,-1):
                pp=astar(grid,player,exit_pos)
                target=pp[min(3,len(pp)-1)] if len(pp)>=2 else (pp[-1] if pp else player)
            path=astar(grid,(self.r,self.c),target)
            if path: self.r,self.c=path[0]
        elif self.state=="WANDER":
            if not self.wander_path:
                for _ in range(25):
                    dr=random.randint(3,9); dc=random.randint(3,9)
                    nr=clamp(self.r+random.choice([-dr,dr]),1,rows-2)
                    nc=clamp(self.c+random.choice([-dc,dc]),1,cols-2)
                    if grid[nr][nc]==0:
                        p=astar(grid,(self.r,self.c),(nr,nc))
                        if p: self.wander_path=p; break
            if self.wander_path: self.r,self.c=self.wander_path.pop(0)
        elif self.state=="PATROL":
            if not self.patrol_path:
                pts=[]; base=(self.r,self.c)
                for _ in range(5):
                    dr=random.randint(2,7)*random.choice([-1,1])
                    dc=random.randint(2,7)*random.choice([-1,1])
                    nr=clamp(self.r+dr,1,rows-2); nc=clamp(self.c+dc,1,cols-2)
                    if grid[nr][nc]==0: pts.append((nr,nc))
                if pts:
                    full=[]; prev=base
                    for pt in pts: full+=astar(grid,prev,pt); prev=pt
                    self.patrol_path=full; self.patrol_idx=0
            if self.patrol_path:
                self.patrol_idx%=len(self.patrol_path)
                self.r,self.c=self.patrol_path[self.patrol_idx]
                self.patrol_idx=(self.patrol_idx+1)%len(self.patrol_path)

    def stun(self,dur=0.9):
        self.stun_timer=dur; self.flash_timer=0.45; self.state="WANDER"

# ═══════════════════════════════════════════════════
#  SPOTLIGHT
# ═══════════════════════════════════════════════════
def make_spotlight(radius, w, h, px, py):
    dark=pygame.Surface((w,h),pygame.SRCALPHA); dark.fill((0,0,0,220))
    hole=pygame.Surface((w,h),pygame.SRCALPHA); hole.fill((0,0,0,0))
    for i in range(radius,0,-1):
        a=int(220*(i/radius)**1.6)
        pygame.draw.circle(hole,(0,0,0,a),(px,py),i)
    dark.blit(hole,(0,0),special_flags=pygame.BLEND_RGBA_MIN)
    return dark

# ═══════════════════════════════════════════════════
#  GAME
# ═══════════════════════════════════════════════════
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Maze  —  Adaptive Director")
        self.screen=pygame.display.set_mode((WIDTH,HEIGHT))
        self.clock =pygame.time.Clock()

        def _font(size,bold=False):
            for name in ("segoeui","helveticaneue","arial","consolas"):
                try:
                    f=pygame.font.SysFont(name,size,bold=bold)
                    if f: return f
                except Exception: pass
            return pygame.font.Font(None,size)

        self.f_title=_font(58,bold=True); self.f_big=_font(44,bold=True)
        self.f_med=_font(22,bold=True);   self.f_ui=_font(17)
        self.f_small=_font(13)

        self.snd=SoundSystem()
        self.players_db: dict = _load_all_players()
        self.current_player: str = ""

        # ── face-login state ─────────────────────
        self.face_state      = "scanning"   # scanning | register_prompt | register_name_input | registering | failed
        self.face_status_msg = "Looking for your face…"
        self.face_scan_done  = False
        self.face_scan_result: Optional[str] = None
        self._reg_name_input = ""
        self._reg_status     = ""
        self._scan_start_ms  = 0

        # game state
        self.running=True; self.screen_state="face_login"
        self.mode=None; self.select_mode=None
        self.adaptive_on=True; self.sound_on=True

        self.stats  =StatsData()
        self.profile=compute_profile(self.stats)
        self.director=AdaptiveDirector(); self._dir_t=0.0

        self.level=1; self.grid=None; self.rows=self.cols=0
        self.cell=20; self.grid_origin=(30,100)
        self.player=(1,1); self.exit=(1,1)
        self.enemies: List[FSMEnemy]=[]
        self.coins:   List[Tuple[int,int]]=[]
        self.hearts:  List[Tuple[int,int]]=[]
        self.particles: List[Particle]=[]
        self.heatmap: Optional[List[List[int]]]=None

        self.fog=False; self.fog_radius=4
        self.nightout_radius=80

        self.start_ticks=0; self.elapsed_s=0
        self.wrong=0; self.path_len=0; self.opt_len=1; self.score=0
        self.hp=5; self.max_hp=5
        self.spawn_timer=0.0; self.wave=1; self.wave_timer=0.0
        self.invuln_timer=0.0
        self.attacks_this_wave=0; self.max_attacks_per_wave=5
        self._mv_cd=0.0; self._mv_rate=0.09
        self.apoc_difficulty="Medium"
        self.wave_alert_shown=False

        # Skip in-game face scan — auth already handled by launch.py
        self.face_scan_done  = True
        self.face_scan_result = None
        self.screen_state = "menu"
        self.build_menu()

    # ═══════════════════════════════════════════════
    #  FACE LOGIN FLOW
    # ═══════════════════════════════════════════════
    def _start_face_scan(self):
        import threading
        self.face_scan_done   = False
        self.face_scan_result = None
        self.face_state       = "scanning"
        self.face_status_msg  = "Looking for your face…"
        self._scan_start_ms   = pygame.time.get_ticks()
        def _scan():
            self.face_scan_result = face_login()   # wall-clock capped inside
            self.face_scan_done   = True
        threading.Thread(target=_scan, daemon=True).start()

    def _start_register(self, name: str):
        import threading
        self.face_scan_done  = False
        self.face_scan_result= None
        self.face_state      = "registering"
        self.face_status_msg = f"Registering face for  {name}…  look at the camera."
        def _reg():
            ok = register_face(name, capture_frames=5)
            if ok:
                self.face_scan_result = name
                self.face_state       = "reg_done"
            else:
                self.face_state       = "failed"
                self.face_status_msg  = "Could not capture face. Is your camera connected?"
            self.face_scan_done = True
        threading.Thread(target=_reg, daemon=True).start()

    def _complete_login(self, name: str):
        """Load this player's stats and go straight to the menu."""
        self.current_player = name
        if name not in self.players_db:
            self.players_db[name] = {"history": []}
            _save_all_players(self.players_db)
        self.stats   = load_stats_for(name, self.players_db)
        self.profile = compute_profile(self.stats)
        self.screen_state = "menu"
        self.build_menu()

    def draw_face_login(self, events):
        self.screen.fill(BG)
        t = pygame.time.get_ticks() / 1000.0

        # star field
        rng = random.Random(17)
        for _ in range(110):
            sx=rng.randint(0,WIDTH); sy=rng.randint(0,HEIGHT)
            br=int((math.sin(t*rng.uniform(0.4,1.8)+rng.uniform(0,6))+1)*0.5*160)+40
            pygame.draw.circle(self.screen,(br,br,min(br+40,255)),(sx,sy),1)

        cx = WIDTH // 2
        glow=(math.sin(t*1.8)+1)*0.5
        self.blit_c("AI  MAZE  GAME", cx, 44, self.f_title, blend(CYAN,PURPLE,glow))
        self.blit_c("Face Login", cx, 130, self.f_med, MUTED)

        # ── handle background thread results ─────
        if self.face_scan_done:
            if self.face_state == "scanning":
                if self.face_scan_result:
                    # recognised — log straight in, no input needed
                    self._complete_login(self.face_scan_result)
                    return
                else:
                    # not recognised
                    self.face_state      = "register_prompt"
                    self.face_status_msg = "Face not recognised."
                    self.face_scan_done  = False

            elif self.face_state == "registering" and self.face_scan_result:
                # registration succeeded — log straight in
                self._complete_login(self.face_scan_result)
                return

        # ── scanning / registering — show spinner ─
        if self.face_state in ("scanning", "registering"):
            dots = "." * (int(t * 2) % 4)
            self.blit_c(self.face_status_msg + dots, cx, 210, self.f_med, CYAN)
            # animated camera icon
            cr = 48
            pygame.draw.rect(self.screen, blend(CYAN,(0,0,0),0.7),
                             (cx-cr-10, 270, cr*2+20, cr+20), border_radius=10)
            pygame.draw.rect(self.screen, CYAN,
                             (cx-cr-10, 270, cr*2+20, cr+20), 2, border_radius=10)
            pulse=(math.sin(t*4)+1)*0.5
            lens_col=blend((20,80,120),CYAN,pulse)
            pygame.draw.circle(self.screen, lens_col, (cx, 270+cr//2+10), cr//2)
            pygame.draw.circle(self.screen, blend(lens_col,(255,255,255),0.4),
                               (cx, 270+cr//2+10), cr//2, 2)

            # elapsed timer + skip button (always shown so user is never locked)
            elapsed_s = (pygame.time.get_ticks() - getattr(self, '_scan_start_ms', pygame.time.get_ticks())) / 1000.0
            elapsed_lbl = self.f_small.render(f"Scanning… {int(elapsed_s)}s  (max ~12s)", True, MUTED)
            self.screen.blit(elapsed_lbl, (cx - elapsed_lbl.get_width()//2, 348))
            skip_btn = Button((cx - 160, 384, 320, 46), "SKIP  →  PLAY AS GUEST")
            skip_btn.draw(self.screen, self.f_med)
            if skip_btn.handle(events):
                # cancel the background thread gracefully (it will finish on its own)
                self.face_scan_done  = True
                self.face_scan_result = None
                self.face_state      = "register_prompt"
                self.face_status_msg = "Skipped — play as guest or register your face."

        # ── not recognised — offer register or retry ─
        elif self.face_state == "register_prompt":
            self.blit_c(self.face_status_msg, cx, 200, self.f_med, ORANGE)
            self.blit_c("First time here? Register your face to create a profile.",
                        cx, 238, self.f_ui, MUTED)
            reg_btn   = Button((cx-200, 286, 400, 56), "REGISTER MY FACE", primary=True)
            retry_btn = Button((cx-200, 358, 400, 56), "TRY SCAN AGAIN")
            reg_btn.draw(self.screen, self.f_med)
            retry_btn.draw(self.screen, self.f_med)
            if reg_btn.handle(events):
                self.face_state      = "register_name_input"
                self._reg_name_input = ""
                self._reg_status     = ""
            if retry_btn.handle(events):
                self._start_face_scan()

        # ── type your name once ─────────────────
        elif self.face_state == "register_name_input":
            self.blit_c("Enter your name, then press Enter:", cx, 196, self.f_med, CYAN)
            bw, bh = 440, 54
            box = pygame.Rect(cx - bw//2, 244, bw, bh)
            border_col = blend(CYAN, PURPLE, (math.sin(t*2)+1)*0.5)
            pygame.draw.rect(self.screen, (14,14,34), box, border_radius=10)
            pygame.draw.rect(self.screen, border_col, box, 2, border_radius=10)
            cursor = "|" if int(t*2)%2==0 else " "
            ns = self.f_med.render(self._reg_name_input + cursor, True, TEXT)
            self.screen.blit(ns, (box.x+12, box.y+(bh-ns.get_height())//2))
            if not self._reg_name_input:
                ph = self.f_ui.render("Type your name…", True, MUTED)
                self.screen.blit(ph, (box.x+12, box.y+(bh-ph.get_height())//2))

            can = bool(self._reg_name_input.strip())
            ok_btn = Button((cx-150, 314, 300, 52), "CONFIRM & SCAN FACE", primary=can)
            ok_btn.draw(self.screen, self.f_med)
            if self._reg_status:
                self.blit_c(self._reg_status, cx, 378, self.f_small, RED)

            def _try_confirm():
                name = self._reg_name_input.strip()
                if name in self.players_db:
                    self._reg_status = f"'{name}' already exists. Pick a different name."
                else:
                    self._start_register(name)

            for ev in events:
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_BACKSPACE:
                        self._reg_name_input = self._reg_name_input[:-1]
                    elif ev.key == pygame.K_RETURN and can:
                        _try_confirm()
                    elif ev.unicode.isprintable() and len(self._reg_name_input) < 20:
                        self._reg_name_input += ev.unicode
            if ok_btn.handle(events) and can:
                _try_confirm()

        # ── camera error ─────────────────────────
        elif self.face_state == "failed":
            self.blit_c(self.face_status_msg, cx, 220, self.f_med, RED)
            retry_btn = Button((cx-180, 290, 360, 56), "TRY AGAIN", primary=True)
            retry_btn.draw(self.screen, self.f_med)
            if retry_btn.handle(events):
                self.face_state      = "register_name_input"
                self.face_scan_done  = False
                self._reg_name_input = ""
                self._reg_status     = ""

        pygame.display.flip()

    # ═══════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════
    def blit(self,txt,pos,font,color):
        self.screen.blit(font.render(txt,True,color),pos)

    def blit_c(self,txt,cx,y,font,color):
        s=font.render(txt,True,color)
        self.screen.blit(s,(cx-s.get_width()//2,y))

    def _cell_center(self,r,c):
        ox,oy=self.grid_origin
        return ox+c*self.cell+self.cell//2, oy+r*self.cell+self.cell//2

    def _emit(self,x,y,color,n=10,spd=90):
        for _ in range(n):
            a=random.uniform(0,math.tau); v=random.uniform(18,spd)
            self.particles.append(Particle(x,y,math.cos(a)*v,math.sin(a)*v,color,random.uniform(0.3,0.7)))

    # ═══════════════════════════════════════════════
    #  MENU
    # ═══════════════════════════════════════════════
    def build_menu(self):
        cx=WIDTH//2; top=270; w,h,gap=400,54,15
        self.menu_btns=[
            Button((cx-w//2,top+0*(h+gap),w,h),"SELECT LEVEL",primary=True),
            Button((cx-w//2,top+1*(h+gap),w,h),"MY STATS"),
            Button((cx-w//2,top+2*(h+gap),w,h),"SCOREBOARD"),
            Button((cx-w//2,top+3*(h+gap),w,h),f"ADAPTIVE AI:  {'ON ' if self.adaptive_on else 'OFF'}"),
            Button((cx-w//2,top+4*(h+gap),w,h),f"SOUND:  {'ON ' if self.sound_on else 'OFF'}"),
            Button((cx-w//2,top+5*(h+gap),w,h),"SWITCH PLAYER  (re-scan face)"),
            Button((cx-w//2,top+6*(h+gap),w,h),"EXIT"),
        ]

    def draw_menu(self,events):
        self.screen.fill(BG)
        t=pygame.time.get_ticks()/1000.0
        rng=random.Random(42)
        for _ in range(130):
            sx=rng.randint(0,WIDTH); sy=rng.randint(0,HEIGHT)
            br=int((math.sin(t*rng.uniform(0.4,1.8)+rng.uniform(0,6))+1)*0.5*180)+40
            pygame.draw.circle(self.screen,(br,br,min(br+30,255)),(sx,sy),rng.randint(1,2))

        glow=(math.sin(t*1.8)+1)*0.5
        title_col=blend(CYAN,PURPLE,glow)
        shadow=self.f_title.render("AI  MAZE  GAME",True,(0,0,0))
        self.screen.blit(shadow,(WIDTH//2-shadow.get_width()//2+3,55))
        self.blit_c("AI  MAZE  GAME",WIDTH//2,52,self.f_title,title_col)
        self.blit_c("A* Pathfinding  ·  Adaptive AI Director  ·  FSM Enemies  ·  Player Profiling",
                    WIDTH//2,122,self.f_small,MUTED)

        pcol=PROFILE_COLORS.get(self.profile,MUTED)
        name_tag=f"  {self.current_player}  ·  {self.profile}  "
        bs=self.f_med.render(name_tag,True,pcol)
        bx=WIDTH//2-bs.get_width()//2; by=142
        pygame.draw.rect(self.screen,blend(pcol,(0,0,0),0.88),(bx-7,by-5,bs.get_width()+14,bs.get_height()+10),border_radius=10)
        pygame.draw.rect(self.screen,pcol,(bx-7,by-5,bs.get_width()+14,bs.get_height()+10),1,border_radius=10)
        self.screen.blit(bs,(bx,by))

        for i,b in enumerate(self.menu_btns):
            if i==3: b.text=f"ADAPTIVE AI:  {'ON ' if self.adaptive_on else 'OFF'}"
            if i==4: b.text=f"SOUND:  {'ON ' if self.sound_on else 'OFF'}"
            b.draw(self.screen,self.f_med)

        if self.menu_btns[0].handle(events): self.snd.play("menu_click"); self.select_mode=None; self.screen_state="select_level"
        if self.menu_btns[1].handle(events): self.snd.play("menu_click"); self.screen_state="stats"
        if self.menu_btns[2].handle(events): self.snd.play("menu_click"); self.screen_state="scoreboard"
        if self.menu_btns[3].handle(events): self.snd.play("menu_click"); self.adaptive_on=not self.adaptive_on
        if self.menu_btns[4].handle(events): self.snd.play("menu_click"); self.sound_on=not self.sound_on; self.snd.enabled=self.sound_on
        if self.menu_btns[5].handle(events): self.snd.play("menu_click"); self._start_face_scan(); self.screen_state="face_login"
        if self.menu_btns[6].handle(events): self.running=False

        self.snd.play_music("menu")
        pygame.display.flip()

    # ═══════════════════════════════════════════════
    #  SELECT LEVEL
    # ═══════════════════════════════════════════════
    def draw_select_level(self,events):
        self.screen.fill(BG)
        cx=WIDTH//2; w,h,gap=500,76,18; top=215

        if self.select_mode is None:
            self.blit_c("SELECT LEVEL",cx,78,self.f_big,CYAN)
            self.blit_c("Choose a game mode:",cx,145,self.f_med,MUTED)
            modes=[
                ("LOST IN A MAZE","maze",    PURPLE, "7 levels  ·  procedural maze  ·  many paths to explore"),
                ("SURVIVAL",      "survival",RED,    "Zombie waves  ·  collect coins  ·  adaptive difficulty"),
                ("NIGHT OUT",     "nightout",(180,120,255),"Pitch dark  ·  spotlight only  ·  find coins in the dark"),
            ]
            for i,(label,key,col,desc) in enumerate(modes):
                rect=pygame.Rect(cx-w//2,top+i*(h+gap),w,h)
                hov=rect.collidepoint(pygame.mouse.get_pos())
                pygame.draw.rect(self.screen,blend(col,(0,0,0),0.50 if hov else 0.77),rect,border_radius=14)
                pygame.draw.rect(self.screen,col,rect,2,border_radius=14)
                self.blit_c(label,cx,rect.y+10,self.f_med,TEXT)
                self.blit_c(desc, cx,rect.y+44,self.f_small,blend(col,(215,215,215),0.4))
                for e in events:
                    if e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and rect.collidepoint(e.pos):
                        self.select_mode=key
            back_y=top+3*(h+gap)+16
            back=Button((cx-115,back_y,230,52),"BACK")
            back.draw(self.screen,self.f_med)
            if back.handle(events): self.screen_state="menu"; self.build_menu()
        else:
            titles={"maze":("LOST IN A MAZE",PURPLE),"survival":("SURVIVAL",RED),"nightout":("NIGHT OUT",(180,120,255))}
            title,tcol=titles[self.select_mode]
            self.blit_c(title,cx,78,self.f_big,tcol)
            self.blit_c("Select difficulty / starting level:",cx,145,self.f_med,MUTED)

            if self.select_mode=="maze":
                opts=[
                    ("LEVEL 1  —  Easy",   1,GREEN,  "Larger maze · extra corridors · no enemies"),
                    ("LEVEL 3  —  Medium", 3,CYAN,   "Big maze · many paths · one hunter"),
                    ("LEVEL 5  —  Hard",   5,ORANGE, "Huge maze · fog of war · two enemies"),
                ]
                for i,(label,lvl,col,desc) in enumerate(opts):
                    rect=pygame.Rect(cx-w//2,top+i*(h+gap),w,h)
                    hov=rect.collidepoint(pygame.mouse.get_pos())
                    pygame.draw.rect(self.screen,blend(col,(0,0,0),0.50 if hov else 0.77),rect,border_radius=14)
                    pygame.draw.rect(self.screen,col,rect,2,border_radius=14)
                    self.blit_c(label,cx,rect.y+10,self.f_med,TEXT)
                    self.blit_c(desc, cx,rect.y+44,self.f_small,blend(col,(215,215,215),0.4))
                    for e in events:
                        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and rect.collidepoint(e.pos):
                            self.profile=compute_profile(self.stats)
                            self.init_maze(lvl); self.screen_state="play"; self.select_mode=None
            else:
                opts=[
                    ("EASY",  "Easy",  GREEN,  "0.7× spawn rate  ·  forgiving  ·  good for learning"),
                    ("MEDIUM","Medium",ORANGE, "1.0× spawn rate  ·  balanced  ·  recommended"),
                    ("HARD",  "Hard",  RED,    "1.4× spawn rate  ·  relentless  ·  true challenge"),
                ]
                for i,(label,key,col,desc) in enumerate(opts):
                    rect=pygame.Rect(cx-w//2,top+i*(h+gap),w,h)
                    hov=rect.collidepoint(pygame.mouse.get_pos())
                    sel=self.apoc_difficulty==key
                    bg2=blend(col,(0,0,0),0.40 if sel else (0.50 if hov else 0.77))
                    pygame.draw.rect(self.screen,bg2,rect,border_radius=14)
                    pygame.draw.rect(self.screen,col,rect,3 if sel else 2,border_radius=14)
                    self.blit_c(label,cx,rect.y+10,self.f_med,TEXT)
                    self.blit_c(desc, cx,rect.y+44,self.f_small,blend(col,(215,215,215),0.4))
                    for e in events:
                        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and rect.collidepoint(e.pos):
                            self.apoc_difficulty=key
                            if self.select_mode=="survival": self.init_survival()
                            else: self.init_nightout()
                            self.screen_state="play"; self.select_mode=None
            back_y=top+3*(h+gap)+16
            back=Button((cx-115,back_y,230,52),"BACK")
            back.draw(self.screen,self.f_med)
            if back.handle(events): self.select_mode=None

        pygame.display.flip()

    # ═══════════════════════════════════════════════
    #  INIT MODES
    # ═══════════════════════════════════════════════
    def init_maze(self,lvl):
        self.mode="maze"; self.level=lvl
        prof=self.profile if self.adaptive_on else "Balanced"
        p=difficulty_params(prof,lvl)
        self.rows,self.cols=p["rows"],p["cols"]
        self.grid=generate_maze(self.rows,self.cols,loop_factor=0.22)
        self.player=(1,1); self.exit=(self.rows-2,self.cols-2)
        self.grid[self.exit[0]][self.exit[1]]=0
        opt=astar(self.grid,self.player,self.exit); self.opt_len=max(1,len(opt))
        self.enemies=[]
        for i in range(p["enemies"]):
            kind="hunter" if i==0 else ("blocker" if i==1 else "sentinel")
            color=RED if kind=="hunter" else (ORANGE if kind=="blocker" else YELLOW)
            er,ec=self._place_far(8)
            self.enemies.append(FSMEnemy(er,ec,kind,p["speed"],color))
        self.fog=p["fog"]; self.coins=[]
        self._scatter_coins(5)
        self._reset_run(keep_score=True)
        self.snd.play_music("maze")

    def init_survival(self):
        self.mode="survival"; self.level=1; self.wave=1
        dm={"Easy":0.7,"Medium":1.0,"Hard":1.4}.get(self.apoc_difficulty,1.0)
        self.rows,self.cols=31,43
        self.grid=generate_open_field(self.rows,self.cols,0.13)
        self.player=(self.rows//2,self.cols//2)
        self.grid[self.player[0]][self.player[1]]=0
        self.exit=(-1,-1); self.enemies=[]; self.coins=[]
        self.hp=self.max_hp=5; self.fog=False
        self.director.reset(); self.director.difficulty=1.0*dm
        self._reset_run(keep_score=False)
        self.spawn_timer=self.wave_timer=0.0
        self._scatter_coins(10); self._scatter_hearts(7)
        for _ in range(3): self._spawn_survival("zombie")
        self.snd.play_music("survival")

    def init_nightout(self):
        self.mode="nightout"; self.level=1; self.wave=1
        dm={"Easy":0.7,"Medium":1.0,"Hard":1.4}.get(self.apoc_difficulty,1.0)
        self.rows,self.cols=33,47
        self.grid=generate_open_field(self.rows,self.cols,0.08)
        self.player=(self.rows//2,self.cols//2)
        self.grid[self.player[0]][self.player[1]]=0
        self.exit=(-1,-1); self.enemies=[]; self.coins=[]
        self.hp=self.max_hp=5
        self.director.reset(); self.director.difficulty=1.0*dm
        self._reset_run(keep_score=False)
        self.spawn_timer=self.wave_timer=0.0
        self._scatter_coins(14); self._scatter_hearts(4)
        for _ in range(2): self._spawn_survival("zombie")
        self.snd.play_music("nightout")

    def _reset_run(self,keep_score=True):
        self.start_ticks=pygame.time.get_ticks()
        self.elapsed_s=self.wrong=self.path_len=0
        self.invuln_timer=0.0; self.particles=[]; self.hearts=[]
        self.wave_alert_shown=False
        if hasattr(self,'_wave_banner_t'): self._wave_banner_t=0.0
        self.attacks_this_wave=0
        if not keep_score: self.score=0
        self.heatmap=[[0]*self.cols for _ in range(self.rows)]
        mw=WIDTH-30; mh=HEIGHT-140
        self.cell=clamp(min(mw//self.cols,mh//self.rows),CELL_MIN,CELL_MAX)
        grid_w=self.cols*self.cell; grid_h=self.rows*self.cell
        self.grid_origin=((WIDTH-grid_w)//2,105)
        self.nightout_radius=int(self.cell*6)

    def _place_far(self,safe=10):
        for _ in range(400):
            r=random.randint(1,self.rows-2); c=random.randint(1,self.cols-2)
            if self.grid[r][c]==1: continue
            if (r,c)==self.player or (r,c)==self.exit: continue
            if manhattan((r,c),self.player)<safe: continue
            if astar(self.grid,(r,c),self.player): return r,c
        return self.rows-3,self.cols-3

    def _scatter_coins(self,n):
        attempts=0
        while len(self.coins)<n and attempts<n*8:
            attempts+=1
            r=random.randint(1,self.rows-2); c=random.randint(1,self.cols-2)
            if self.grid[r][c]==0 and (r,c)!=self.player and (r,c)!=self.exit \
               and (r,c) not in self.coins:
                self.coins.append((r,c))

    def _scatter_hearts(self,n):
        attempts=0
        while len(self.hearts)<n and attempts<n*10:
            attempts+=1
            r=random.randint(1,self.rows-2); c=random.randint(1,self.cols-2)
            if (self.grid[r][c]==0 and (r,c)!=self.player
                    and (r,c) not in self.coins and (r,c) not in self.hearts):
                self.hearts.append((r,c))

    def _in_fog(self,r,c):
        if not self.fog: return True
        pr,pc=self.player
        return abs(r-pr)<=self.fog_radius and abs(c-pc)<=self.fog_radius

    def _in_spotlight(self,r,c):
        pr,pc=self.player
        return abs(r-pr)<=self.fog_radius+1 and abs(c-pc)<=self.fog_radius+1

    def _spawn_survival(self,kind="zombie"):
        diff=self.director.difficulty if self.adaptive_on else 1.0
        speed=0.8+(diff-0.7)*0.75
        color=RED if kind=="zombie" else ORANGE
        for _ in range(300):
            side=random.choice(["top","bot","left","right"])
            r=(1 if side=="top" else self.rows-2 if side=="bot" else random.randint(1,self.rows-2))
            c=(1 if side=="left" else self.cols-2 if side=="right" else random.randint(1,self.cols-2))
            if self.grid[r][c]==1: continue
            if manhattan((r,c),self.player)<14: continue
            e=FSMEnemy(r,c,kind,speed,color); e.state="WANDER"
            self.enemies.append(e); self.snd.play("spawn"); return

    # ═══════════════════════════════════════════════
    #  MOVEMENT / COMBAT
    # ═══════════════════════════════════════════════
    def _try_move(self,dr,dc):
        nr=self.player[0]+dr; nc=self.player[1]+dc
        if nr<0 or nr>=self.rows or nc<0 or nc>=self.cols or self.grid[nr][nc]==1:
            self.wrong+=1; self.snd.play("wall"); return
        self.player=(nr,nc); self.path_len+=1
        if self.heatmap and nr<len(self.heatmap) and nc<len(self.heatmap[nr]):
            self.heatmap[nr][nc]+=1
        if (nr,nc) in self.coins:
            self.coins.remove((nr,nc)); self.score+=50
            self.director.report_coin(1)
            cx2,cy2=self._cell_center(nr,nc)
            self._emit(cx2,cy2,GOLD,14,110); self.snd.play("coin")
            if self.mode in ("survival","nightout"): self._scatter_coins(1)
        if (nr,nc) in self.hearts:
            self.hearts.remove((nr,nc))
            if self.hp<self.max_hp:
                self.hp+=1; cx2,cy2=self._cell_center(nr,nc)
                self._emit(cx2,cy2,(255,80,120),14,90); self.snd.play("heal")
            if self.mode in ("survival","nightout"): self._scatter_hearts(1)
        if self.invuln_timer<=0 and any(e.r==nr and e.c==nc for e in self.enemies):
            self._take_damage()
        if self.mode=="maze" and self.player==self.exit:
            self.snd.play("levelup"); self._finish_maze()

    def _take_damage(self):
        if self.invuln_timer>0: return
        self.hp-=1; self.invuln_timer=2.0
        self.director.report_damage(1)
        pr,pc=self.player; cx2,cy2=self._cell_center(pr,pc)
        self._emit(cx2,cy2,RED,18,130); self.snd.play("damage")
        if self.hp<=0:
            self.snd.play("death"); self.snd.stop_music()
            if self.adaptive_on: self.director.report_death()
            if self.mode in ("survival","nightout"):
                eff=min(self.path_len/max(1,self.path_len+self.wrong),1.0)
                result=LevelResult(self.mode,self.wave,self.elapsed_s,self.wrong,eff,self.score)
                self.stats.history.append(result); _rollup(self.stats)
                save_stats_for(self.current_player,self.stats,self.players_db)
                self.profile=compute_profile(self.stats)
            self.screen_state="gameover"

    def _player_attack(self):
        if self.attacks_this_wave>=self.max_attacks_per_wave:
            self.snd.play("attack_miss"); return
        if not self.enemies: self.snd.play("attack_miss"); return
        pr,pc=self.player
        in_range=sorted(
            [(manhattan((pr,pc),(e.r,e.c)),i,e) for i,e in enumerate(self.enemies)
             if manhattan((pr,pc),(e.r,e.c))<=7],
            key=lambda x: x[0]
        )
        if not in_range: self.snd.play("attack_miss"); return
        killed_idx=set()
        for _,i,e in in_range[:3]:
            cx2,cy2=self._cell_center(e.r,e.c)
            self._emit(cx2,cy2,RED,22,150); self.score+=150
            self.director.report_kill(1)
            if random.random()<0.65: self.coins.append((e.r,e.c))
            killed_idx.add(i)
        self.enemies=[e for i,e in enumerate(self.enemies) if i not in killed_idx]
        self.attacks_this_wave+=1; self.snd.play("attack_hit")

    def _update_enemies(self,dt):
        diff=self.director.difficulty if self.adaptive_on else 1.0
        for e in self.enemies:
            e.update_visual(dt); e.decide_state(self.player,diff)
            e.step(self.grid,self.player,self.exit,dt,diff)
        if self.invuln_timer<=0:
            for e in self.enemies:
                if e.r==self.player[0] and e.c==self.player[1]:
                    if self.mode=="maze":
                        self.snd.play("damage")
                        eff=min(self.opt_len/max(1,self.path_len),1.0)
                        result=LevelResult("maze",self.level,self.elapsed_s,self.wrong,eff,0)
                        self.stats.history.append(result); _rollup(self.stats)
                        save_stats_for(self.current_player,self.stats,self.players_db)
                        self.profile=compute_profile(self.stats)
                        self.screen_state="gameover"
                    else: self._take_damage()
                    break

    def _finish_maze(self):
        eff=min(self.opt_len/max(1,self.path_len),1.0)
        s=max(0,1400-self.elapsed_s*10-self.wrong*5+int(eff*350))
        self.score+=s
        self.stats.history.append(LevelResult("maze",self.level,self.elapsed_s,self.wrong,eff,s))
        _rollup(self.stats)
        save_stats_for(self.current_player,self.stats,self.players_db)
        self.profile=compute_profile(self.stats); self.level+=1
        if self.level>7: self.screen_state="win"
        else: self.init_maze(self.level)

    # ═══════════════════════════════════════════════
    #  DRAW GRID
    # ═══════════════════════════════════════════════
    def draw_grid(self):
        ox,oy=self.grid_origin; cell=self.cell
        t=pygame.time.get_ticks()/1000.0
        is_night=(self.mode=="nightout")
        for r in range(self.rows):
            for c in range(self.cols):
                vis=self._in_spotlight(r,c) if is_night else self._in_fog(r,c)
                x=ox+c*cell; y=oy+r*cell; rect=pygame.Rect(x,y,cell,cell)
                if not vis:
                    pygame.draw.rect(self.screen,FOG_C,rect); continue
                if self.grid[r][c]==1:
                    wc=blend(WALL_C,(30,20,8),0.3) if is_night else WALL_C
                    pygame.draw.rect(self.screen,wc,rect)
                    pygame.draw.line(self.screen,WALL_LIT,(x,y),(x+cell-1,y))
                    pygame.draw.line(self.screen,WALL_LIT,(x,y),(x,y+cell-1))
                else:
                    fc=FLOOR_C
                    if self.heatmap and r<len(self.heatmap) and c<len(self.heatmap[r]):
                        hv=self.heatmap[r][c]
                        if hv>0:
                            heat=min(hv/8.0,1.0)
                            fc=blend(FLOOR_C,(30,10,50) if not is_night else (18,8,4),heat)
                    pygame.draw.rect(self.screen,fc,rect)

        if self.mode=="maze":
            er,ec=self.exit
            if self._in_fog(er,ec):
                x=ox+ec*cell; y=oy+er*cell; rect=pygame.Rect(x,y,cell,cell)
                pulse=(math.sin(t*4)+1)*0.5
                pygame.draw.rect(self.screen,blend(GREEN_DIM,GREEN,pulse),rect)
                pygame.draw.rect(self.screen,GREEN,rect.inflate(-cell*.35,-cell*.35),border_radius=3)

        for (cr2,cc2) in self.coins:
            vis2=self._in_spotlight(cr2,cc2) if is_night else self._in_fog(cr2,cc2)
            if not vis2: continue
            cx2,cy2=self._cell_center(cr2,cc2)
            pulse2=(math.sin(t*7+cr2*0.9)+1)*0.5
            rad2=max(2,int(cell*0.22+pulse2*2))
            pygame.draw.circle(self.screen,GOLD,(cx2,cy2),rad2)
            pygame.draw.circle(self.screen,(255,230,120),(cx2,cy2),max(1,rad2-2))

        for (hr2,hc2) in self.hearts:
            vis_h=self._in_spotlight(hr2,hc2) if is_night else self._in_fog(hr2,hc2)
            if not vis_h: continue
            hpx,hpy=self._cell_center(hr2,hc2)
            pulse_h=(math.sin(t*5+hr2*1.3)+1)*0.5
            r_h=max(2,int(cell*0.18+pulse_h*2))
            hcol2=blend((200,50,80),(255,100,140),pulse_h)
            pygame.draw.circle(self.screen,hcol2,(hpx-r_h//2,hpy-1),r_h//2+1)
            pygame.draw.circle(self.screen,hcol2,(hpx+r_h//2,hpy-1),r_h//2+1)
            pygame.draw.polygon(self.screen,hcol2,[(hpx-r_h,hpy),(hpx+r_h,hpy),(hpx,hpy+r_h+1)])

        for e in self.enemies:
            vis3=self._in_spotlight(e.r,e.c) if is_night else self._in_fog(e.r,e.c)
            if not vis3: continue
            vx=ox+e.visual_x*cell+cell//2; vy=oy+e.visual_y*cell+cell//2
            col=e.get_draw_color(); rad=max(3,cell//3)
            if e.kind in ("hunter","zombie"):
                pygame.draw.circle(self.screen,col,(int(vx),int(vy)),rad)
                pygame.draw.circle(self.screen,blend(col,(255,255,255),0.4),(int(vx),int(vy)),max(1,rad-3),2)
                ex2=int(vx+(self.player[1]-e.c)*2); ey2=int(vy+(self.player[0]-e.r)*2)
                pygame.draw.circle(self.screen,(255,255,255),(ex2,ey2),max(1,rad//3))
            elif e.kind=="blocker":
                r2=pygame.Rect(int(vx)-rad,int(vy)-rad,rad*2,rad*2)
                pygame.draw.rect(self.screen,col,r2,border_radius=5)
                pygame.draw.rect(self.screen,blend(col,(255,255,255),0.4),r2,2,border_radius=5)
            elif e.kind=="sentinel":
                pts=[(int(vx),int(vy)-rad),(int(vx)+rad,int(vy)+rad//2),(int(vx)-rad,int(vy)+rad//2)]
                pygame.draw.polygon(self.screen,col,pts)
                pygame.draw.polygon(self.screen,blend(col,(255,255,255),0.4),pts,2)
            tag={"CHASE":"!","ATTACK":"!!","WANDER":"~","PATROL":"P"}.get(e.state,"")
            if tag:
                ts=self.f_small.render(tag,True,blend(col,(255,255,255),0.55))
                self.screen.blit(ts,(int(vx)-ts.get_width()//2,int(vy)-rad-ts.get_height()))

        pr,pc=self.player; px,py=self._cell_center(pr,pc)
        rp=max(3,cell//3); t2=pygame.time.get_ticks()/1000.0
        pulse3=(math.sin(t2*5)+1)*0.5
        if not (self.invuln_timer>0 and int(t2*12)%2==0):
            pcol=SPOTLIGHT_COL if is_night else BLUE
            pygame.draw.circle(self.screen,pcol,(px,py),rp)
            pygame.draw.circle(self.screen,blend(pcol,(255,255,255),0.5),(px,py),rp,2)
            glow_r=rp+int(pulse3*4)
            gs=pygame.Surface((glow_r*2+6,glow_r*2+6),pygame.SRCALPHA)
            gcol=(255,240,150,35) if is_night else (50,120,250,38)
            pygame.draw.circle(gs,gcol,(glow_r+3,glow_r+3),glow_r)
            self.screen.blit(gs,(px-glow_r-3,py-glow_r-3))

        if is_night:
            sp=make_spotlight(self.nightout_radius,WIDTH,HEIGHT,px,py)
            self.screen.blit(sp,(0,0))

        alive=[]
        for p in self.particles:
            if p.update(1/60): p.draw(self.screen); alive.append(p)
        self.particles=alive

    # ═══════════════════════════════════════════════
    #  HUD
    # ═══════════════════════════════════════════════
    def _draw_heart(self,cx,cy,r,filled):
        hcol  = RED           if filled else (50,15,15)
        hcol2 = (255,140,160) if filled else (80,25,25)
        pygame.draw.circle(self.screen,hcol,(cx-r//2,cy-r//4),r//2+1)
        pygame.draw.circle(self.screen,hcol,(cx+r//2,cy-r//4),r//2+1)
        pygame.draw.polygon(self.screen,hcol,[(cx-r,cy-r//4+1),(cx+r,cy-r//4+1),(cx,cy+r)])
        if filled:
            pygame.draw.circle(self.screen,hcol2,(cx-r//2,cy-r//4),r//2+1,1)
            pygame.draw.circle(self.screen,hcol2,(cx+r//2,cy-r//4),r//2+1,1)

    def draw_hud(self):
        pygame.draw.rect(self.screen,PANEL,(0,0,WIDTH,92))
        pygame.draw.line(self.screen,(38,36,78),(0,92),(WIDTH,92),2)

        mode_label=MODE_NAMES.get(self.mode,"")
        pcol=PROFILE_COLORS.get(self.profile,MUTED)
        self.blit(mode_label,(18,14),self.f_med,CYAN)
        self.blit(f"Profile: {self.profile}",(18,48),self.f_small,pcol)
        pname_s=self.f_small.render(f" {self.current_player} ",True,pcol)
        px2=18; py2=68
        pygame.draw.rect(self.screen,blend(pcol,(0,0,0),0.82),(px2-2,py2-2,pname_s.get_width()+4,pname_s.get_height()+4),border_radius=5)
        self.screen.blit(pname_s,(px2,py2))

        if self.mode=="maze":
            items=[
                (220,f"SCORE  {self.score}",           CYAN),
                (420,f"TIME  {self.elapsed_s}s",       YELLOW),
                (580,f"LVL  {self.level}/7",           PURPLE),
                (700,f"WRONG  {self.wrong}",           RED),
            ]
            for xpos,txt,col in items: self.blit(txt,(xpos,30),self.f_med,col)
            eff=min(self.opt_len/max(1,self.path_len),1.0)
            self.blit(f"EFF  {int(eff*100)}%",(900,30),self.f_med,GREEN if eff>0.6 else YELLOW)
        else:
            heart_start_x=210; heart_r=9
            for i in range(self.max_hp):
                hcx=heart_start_x+i*(heart_r*2+6)+heart_r
                self._draw_heart(hcx,32,heart_r,i<self.hp)
            cx_icon=heart_start_x+self.max_hp*(heart_r*2+6)+14
            pygame.draw.circle(self.screen,GOLD,(cx_icon+9,32),9)
            pygame.draw.circle(self.screen,(200,155,20),(cx_icon+9,32),9,2)
            cc_s=self.f_med.render(f" {len(self.coins)}",True,GOLD)
            self.screen.blit(cc_s,(cx_icon+22,22))

            diff=self.director.difficulty if self.adaptive_on else 1.0
            el=self.director.ease_label
            el_col=GREEN if "EASY" in el else (ORANGE if ("EASE" in el or "BAL" in el) else RED)
            right_items=[
                (500,f"SCORE  {self.score}",   CYAN),
                (680,f"TIME  {self.elapsed_s}s",YELLOW),
                (820,f"WAVE  {self.wave}",      ORANGE),
                (960,f"DIFF  {diff:.2f}",       RED if diff>1.3 else (ORANGE if diff>1.0 else GREEN)),
                (1080,f"{el}",                  el_col),
            ]
            for xpos,txt,col in right_items:
                self.blit(txt,(xpos,16),self.f_med if xpos<1080 else self.f_small,col)
            self.blit(f"ENEMIES  {len(self.enemies)}",(500,54),self.f_small,MUTED)
            attacks_left=self.max_attacks_per_wave-self.attacks_this_wave
            atk_col=GREEN if attacks_left>2 else (YELLOW if attacks_left>0 else RED)
            self.blit(f"ATTACKS  {attacks_left}/{self.max_attacks_per_wave}",(660,54),self.f_small,atk_col)

        ai_c=GREEN if self.adaptive_on else RED
        self.blit(f"AI:{'ON' if self.adaptive_on else 'OFF'}",(WIDTH-70,16),self.f_small,ai_c)

        hist=self.director.diff_history
        if len(hist)>1 and self.mode!="maze":
            gx,gy,gw,gh=WIDTH-68,36,56,44
            pygame.draw.rect(self.screen,(4,4,10),(gx,gy,gw,gh))
            sampled=hist[-(gw):]
            n=len(sampled)
            pts=[(gx+int(i/max(1,n-1)*(gw-1)),gy+gh-int((v-0.4)/1.2*gh))
                 for i,v in enumerate(sampled)]
            if len(pts)>1: pygame.draw.lines(self.screen,CYAN,False,pts,1)

    # ═══════════════════════════════════════════════
    #  STATS
    # ═══════════════════════════════════════════════
    def draw_stats(self,events):
        self.screen.fill(BG)
        pcol=PROFILE_COLORS.get(self.profile,MUTED)
        self.blit("PLAYER STATS",(40,22),self.f_big,PURPLE)
        nb=self.f_med.render(f"  {self.current_player}  ",True,pcol)
        nx=WIDTH-nb.get_width()-50; ny=22
        pygame.draw.rect(self.screen,blend(pcol,(0,0,0),0.85),(nx-6,ny-4,nb.get_width()+12,nb.get_height()+8),border_radius=8)
        pygame.draw.rect(self.screen,pcol,(nx-6,ny-4,nb.get_width()+12,nb.get_height()+8),1,border_radius=8)
        self.screen.blit(nb,(nx,ny))

        self.blit(f"Profile: {self.profile}",(40,82),self.f_med,pcol)
        for i,(lbl,val) in enumerate([
            ("Avg Time",     f"{self.stats.avg_time:.1f}s"),
            ("Avg Wrong Moves",f"{self.stats.avg_wrong:.1f}"),
            ("Avg Efficiency", f"{int(self.stats.avg_eff*100)}%"),
            ("Total Runs",     str(self.stats.levels)),
        ]):
            self.blit(f"{lbl}:  {val}",(40,118+i*24),self.f_ui,MUTED)

        descs={"Beginner":"Still learning. Mazes will be gentle.",
               "Balanced":"Well-rounded. Steady challenge ahead.",
               "SpeedRunner":"Fast & efficient! Bigger mazes incoming.",
               "LogicalPlanner":"Minimal wrong turns. Fog & interceptors ahead.",
               "Explorer":"You roam freely. Larger maps await."}
        self.blit(f'"{descs.get(self.profile,"")}"',(40,216),self.f_small,blend(pcol,(200,200,200),0.5))

        y=248
        hdrs=["MODE","LVL","TIME","WRONG","EFF","SCORE"]
        xs  =[40,   215,  310,   425,    535,  650]
        for xh,hd in zip(xs,hdrs): self.blit(hd,(xh,y),self.f_ui,MUTED)
        pygame.draw.line(self.screen,MUTED,(40,y+22),(790,y+22),1); y+=28

        history=self.stats.history[-14:]
        if not history:
            self.blit_c("No runs yet — play a game to see your stats here!",WIDTH//2,y+40,self.f_ui,MUTED)
        for h in history:
            self.blit(MODE_NAMES.get(h.mode,h.mode),(xs[0],y),self.f_small,TEXT)
            self.blit(str(h.level),                  (xs[1],y),self.f_small,PURPLE)
            self.blit(f"{h.time_s}s",                (xs[2],y),self.f_small,YELLOW)
            self.blit(str(h.wrong),                  (xs[3],y),self.f_small,RED if h.wrong>10 else TEXT)
            ec=GREEN if h.efficiency>0.7 else (YELLOW if h.efficiency>0.4 else RED)
            self.blit(f"{int(h.efficiency*100)}%",   (xs[4],y),self.f_small,ec)
            self.blit(str(h.score),                  (xs[5],y),self.f_small,CYAN)
            y+=22

        clr=Button((860,248,160,34),"CLEAR DATA",color=(80,20,20))
        clr.draw(self.screen,self.f_small)
        if clr.handle(events):
            self.stats=StatsData()
            save_stats_for(self.current_player,self.stats,self.players_db)
            self.profile="Beginner"

        back=Button((40,HEIGHT-68,200,50),"BACK")
        back.draw(self.screen,self.f_med)
        if back.handle(events): self.screen_state="menu"; self.build_menu()
        pygame.display.flip()

    # ═══════════════════════════════════════════════
    #  SCOREBOARD
    # ═══════════════════════════════════════════════
    def draw_scoreboard(self,events):
        self.screen.fill(BG)
        self.blit("SCOREBOARD",(40,24),self.f_big,GOLD)
        self.blit_c("All-time highest scores across all players",WIDTH//2,86,self.f_small,MUTED)

        entries=[]
        for pname,pdata in self.players_db.items():
            history=[LevelResult(**x) for x in pdata.get("history",[])]
            if not history: continue
            ps=StatsData(history=history); _rollup(ps)
            prof=compute_profile(ps)
            mode_best: Dict[str,int]={}
            for h in history: mode_best[h.mode]=max(mode_best.get(h.mode,0),h.score)
            for mode,score in mode_best.items():
                entries.append({"player":pname,"mode":mode,"score":score,"profile":prof,"runs":ps.levels})
        entries.sort(key=lambda e: e["score"],reverse=True)

        xs=[50,130,350,560,750,920]
        hdrs=["RANK","PLAYER","MODE","SCORE","PROFILE","RUNS"]
        hdr_cols=[MUTED,MUTED,MUTED,GOLD,MUTED,MUTED]
        y=120
        for xh,hd,hc in zip(xs,hdrs,hdr_cols): self.blit(hd,(xh,y),self.f_ui,hc)
        pygame.draw.line(self.screen,MUTED,(40,y+24),(WIDTH-40,y+24),1); y+=32

        medal_colors={1:GOLD,2:(192,192,192),3:(205,127,50)}
        if not entries:
            self.blit_c("No scores recorded yet — play some games!",WIDTH//2,y+60,self.f_ui,MUTED)
        else:
            for rank,e in enumerate(entries[:18],1):
                pcol=PROFILE_COLORS.get(e["profile"],MUTED)
                is_me=(e["player"]==self.current_player)
                if is_me:
                    row_rect=pygame.Rect(36,y-3,WIDTH-72,26)
                    pygame.draw.rect(self.screen,blend(PURPLE,(0,0,0),0.72),row_rect,border_radius=6)
                    pygame.draw.rect(self.screen,PURPLE,row_rect,1,border_radius=6)
                rank_col=medal_colors.get(rank,TEXT)
                rank_txt={1:"1",2:"2",3:"3"}.get(rank,str(rank))
                self.blit(rank_txt,(xs[0],y),self.f_small,rank_col)
                name_col=blend(CYAN,TEXT,0.3) if is_me else TEXT
                self.blit(e["player"],(xs[1],y),self.f_small,name_col)
                mode_col=PURPLE if e["mode"]=="maze" else (RED if e["mode"]=="survival" else (180,120,255))
                self.blit(MODE_NAMES.get(e["mode"],e["mode"]),(xs[2],y),self.f_small,mode_col)
                self.blit(str(e["score"]),(xs[3],y),self.f_small,GOLD if rank==1 else CYAN)
                self.blit(e["profile"],(xs[4],y),self.f_small,pcol)
                self.blit(str(e["runs"]),(xs[5],y),self.f_small,MUTED)
                y+=26

        back=Button((40,HEIGHT-68,200,50),"BACK")
        back.draw(self.screen,self.f_med)
        if back.handle(events): self.screen_state="menu"; self.build_menu()
        pygame.display.flip()

    # ═══════════════════════════════════════════════
    #  GAME OVER
    # ═══════════════════════════════════════════════
    def draw_gameover(self,events,title="GAME OVER",color=RED):
        self.screen.fill(BG)
        t=pygame.time.get_ticks()/1000.0
        rng=random.Random(13)
        for _ in range(80):
            sx=rng.randint(0,WIDTH); sy=rng.randint(0,HEIGHT)
            br=int((math.sin(t*rng.uniform(0.5,2.0)+rng.uniform(0,6))+1)*0.5*120)+30
            pygame.draw.circle(self.screen,(br,br,min(br+40,255)),(sx,sy),1)

        self.blit_c(title,WIDTH//2,110,self.f_big,color)
        pcol=PROFILE_COLORS.get(self.profile,MUTED)
        self.blit_c(f"{self.current_player}",WIDTH//2,172,self.f_med,pcol)
        self.blit_c(f"Score:  {self.score}",WIDTH//2,210,self.f_big,CYAN)
        if self.mode!="maze":
            self.blit_c(f"Survived  {self.elapsed_s}s   ·   Wave {self.wave}",WIDTH//2,268,self.f_ui,MUTED)

        all_scores=[]
        for pdata in self.players_db.values():
            for h in pdata.get("history",[]):
                all_scores.append(h.get("score",0) if isinstance(h,dict) else h.score)
        all_scores.sort(reverse=True)
        if self.score>0 and self.score in all_scores:
            rank=all_scores.index(self.score)+1
            rank_col=GOLD if rank==1 else ((192,192,192) if rank==2 else (CYAN if rank<=5 else MUTED))
            rank_msg={1:"#1 All-Time!",2:"#2 All-Time!",3:"#3 All-Time!"}.get(rank,f"Rank #{rank} on the Scoreboard")
            self.blit_c(rank_msg,WIDTH//2,300,self.f_med,rank_col)

        btn_y=345
        retry=Button((WIDTH//2-205,btn_y,      410,56),"RETRY",primary=True)
        score=Button((WIDTH//2-205,btn_y+70,   410,56),"VIEW SCOREBOARD")
        menu =Button((WIDTH//2-205,btn_y+140,  410,56),"MAIN MENU")
        retry.draw(self.screen,self.f_med)
        score.draw(self.screen,self.f_med)
        menu.draw(self.screen,self.f_med)

        if retry.handle(events):
            if   self.mode=="maze":     self.init_maze(1)
            elif self.mode=="survival": self.init_survival()
            else:                       self.init_nightout()
            self.screen_state="play"
        if score.handle(events): self.screen_state="scoreboard"
        if menu.handle(events):  self.screen_state="menu"; self.build_menu()
        pygame.display.flip()

    def draw_pause_hint(self):
        hints="ESC = menu   ·   WASD / Arrows = move"
        if self.mode in ("nightout","survival"): hints+="   ·   SPACE = attack  (5 per wave, range 7, hits up to 3)"
        self.blit(hints,(30,HEIGHT-28),self.f_small,MUTED)

    # ═══════════════════════════════════════════════
    #  MAIN LOOP
    # ═══════════════════════════════════════════════
    def run(self):
        while self.running:
            dt=self.clock.tick(FPS)/1000.0
            events=pygame.event.get()
            for e in events:
                if e.type==pygame.QUIT: self.running=False
                if e.type==pygame.KEYDOWN:
                    if e.key==pygame.K_ESCAPE:
                        if self.screen_state in ("play","select_level"):
                            self.screen_state="menu"; self.build_menu()
                        elif self.screen_state in ("scoreboard","stats"):
                            self.screen_state="menu"; self.build_menu()
                    if e.key==pygame.K_SPACE and self.screen_state=="play" \
                       and self.mode in ("nightout","survival"):
                        self._player_attack()

            if self.screen_state=="face_login":
                self.draw_face_login(events); continue
            if self.screen_state=="menu":
                self.draw_menu(events); continue
            if self.screen_state=="scoreboard":
                self.draw_scoreboard(events); continue
            if self.screen_state=="stats":
                self.draw_stats(events); continue
            if self.screen_state=="select_level":
                self.draw_select_level(events); continue
            if self.screen_state=="gameover":
                self.draw_gameover(events); continue
            if self.screen_state=="win":
                self.draw_gameover(events,"YOU ESCAPED!  7/7",GREEN); continue

            # ═══ PLAY ════════════════════════════
            self.screen.fill(BG)
            self.draw_hud()
            self.elapsed_s=(pygame.time.get_ticks()-self.start_ticks)//1000
            self.invuln_timer=max(0.0,self.invuln_timer-dt)
            self._mv_cd=max(0.0,self._mv_cd-dt)

            keys=pygame.key.get_pressed()
            if self._mv_cd<=0:
                moved=False
                if   keys[pygame.K_UP]    or keys[pygame.K_w]: self._try_move(-1,0); moved=True
                elif keys[pygame.K_DOWN]  or keys[pygame.K_s]: self._try_move(1,0);  moved=True
                elif keys[pygame.K_LEFT]  or keys[pygame.K_a]: self._try_move(0,-1); moved=True
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: self._try_move(0,1);  moved=True
                if moved: self._mv_cd=self._mv_rate; self.snd.play("move")

            self.director.tick(dt); self._dir_t+=dt
            if self._dir_t>=2.5:
                self._dir_t=0.0
                if self.adaptive_on: self.director.update_every()

            if self.mode=="maze":
                self._update_enemies(dt)
            else:
                diff=self.director.difficulty if self.adaptive_on else 1.0
                self.spawn_timer+=dt
                si={"survival":2.8,"nightout":3.0}.get(self.mode,2.8)
                cap={"survival":20,"nightout":16}.get(self.mode,18)
                if self.spawn_timer>=max(0.3,si/diff):
                    self.spawn_timer=0.0
                    if len(self.enemies)<cap:
                        kind="zombie" if self.mode=="survival" else random.choice(["zombie","sentinel"])
                        self._spawn_survival(kind)
                self.wave_timer+=dt
                wd={"survival":20.0,"nightout":18.0}.get(self.mode,20.0)
                if self.wave_timer>=wd:
                    self.wave_timer=0.0; self.wave+=1; self.score+=200
                    self.attacks_this_wave=0; self.wave_alert_shown=True
                    self.snd.play("wave_clear")
                    if self.mode=="survival" and self.wave%3==0:
                        self.hp=min(self.hp+1,self.max_hp); self.snd.play("powerup")
                    if self.adaptive_on:
                        self.director.report_kill(2); self.director.report_wave_cleared()
                    self._scatter_coins(3)
                    if self.mode=="survival": self._scatter_hearts(3)
                    else: self._scatter_hearts(1)
                self._update_enemies(dt)

            self.draw_grid()
            self.draw_pause_hint()
            if getattr(self,'wave_alert_shown',False):
                if not hasattr(self,'_wave_banner_t'): self._wave_banner_t=0.0
                self._wave_banner_t+=dt
                alpha=clamp(1.0-self._wave_banner_t*1.4,0,1)
                if alpha>0:
                    msg=f"  WAVE {self.wave}  "
                    bs=self.f_big.render(msg,True,ORANGE)
                    bx=WIDTH//2-bs.get_width()//2; by=HEIGHT//2-40
                    surf=pygame.Surface((bs.get_width()+20,bs.get_height()+14),pygame.SRCALPHA)
                    surf.fill((20,8,0,int(180*alpha)))
                    self.screen.blit(surf,(bx-10,by-7))
                    col2=blend(ORANGE,TEXT,1-alpha)
                    self.blit_c(msg,WIDTH//2,by,self.f_big,col2)
                else:
                    self.wave_alert_shown=False; self._wave_banner_t=0.0
            pygame.display.flip()

        pygame.quit()

# ═══════════════════════════════════════════════════
if __name__=="__main__":
    Game().run()
    
