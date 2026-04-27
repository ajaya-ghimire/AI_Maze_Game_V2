import os, cv2, pickle, threading, time, math
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, List

# storage 
FACES_DIR       = "face_data"
USERS_FILE      = os.path.join(FACES_DIR, "users.pkl")
SAMPLES_NEEDED  = 15     # frames grabbed during registration
MATCH_THRESHOLD = 0.32   # combined similarity threshold
ORB_RATIO       = 0.55   # kept for reference
LOW_LIGHT_THRESH = 45    # mean pixel brightness below this = warn
BOOST_GAMMA      = 0.45  # gamma <1 brightens dark frames

# palette (mirrors game colours) 
BG      = (6,   6,  12)
PANEL   = (14,  14,  32)
TEXT    = (220, 228, 240)
MUTED   = (90,  105, 125)
PURPLE  = (130,  55, 240)
BLUE    = ( 50, 125, 250)
CYAN    = ( 30, 210, 240)
GREEN   = ( 30, 200,  90)
RED     = (240,  60,  60)
ORANGE  = (250, 110,  20)
DARK_PU = ( 22,   8,  50)
YELLOW  = (240, 210,  40)
AMBER   = (255, 160,  20)
TORCH   = (255, 252, 235)   

def _blend(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))

def _font(size, bold=False):
    for name in ("segoeui","helveticaneue","arial","consolas"):
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f: return f
        except Exception:
            pass
    return pygame.font.Font(None, size)


# ══════════════════════════════════════════════════════════════════════════════
#  FACE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class FaceEngine:
    SIZE = 160

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.orb   = cv2.ORB_create(nfeatures=400)
        self.bf    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._glut = self._build_lut(BOOST_GAMMA)

    @staticmethod
    def _build_lut(g):
        return np.array([((i/255.0)**(1.0/g))*255 for i in range(256)], dtype=np.uint8)

    def boost(self, gray):
        try:
            return cv2.LUT(gray, self._glut) if gray.mean() < LOW_LIGHT_THRESH else gray
        except Exception:
            return gray

    @staticmethod
    def brightness(gray):
        try:    return float(gray.mean())
        except: return 255.0

    def detect(self, gray):
        try:
            b = self.boost(gray)
            f = self.cascade.detectMultiScale(b, scaleFactor=1.1, minNeighbors=5, minSize=(55,55))
            if len(f) == 0:
                f = self.cascade.detectMultiScale(b, scaleFactor=1.2, minNeighbors=3, minSize=(45,45))
            return list(f) if len(f) else []
        except Exception:
            return []

    def _crop(self, gray, rect):
        try:
            x,y,w,h = int(rect[0]),int(rect[1]),int(rect[2]),int(rect[3])
            px,py = max(1,int(w*.18)), max(1,int(h*.18))
            crop = gray[max(0,y-py):min(gray.shape[0],y+h+py),
                        max(0,x-px):min(gray.shape[1],x+w+px)]
            if crop.size < 100: return None
            return self.clahe.apply(cv2.resize(crop,(self.SIZE,self.SIZE),
                                               interpolation=cv2.INTER_LINEAR))
        except Exception:
            return None

    @staticmethod
    def _lbp(img):
        try:
            h,w = img.shape
            res = np.zeros((h,w),dtype=np.uint8)
            c   = img[1:-1,1:-1].astype(np.int16)
            code= np.zeros_like(c,dtype=np.uint8)
            for bit,(dy,dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
                nb = img[1+dy:h-1+dy,1+dx:w-1+dx].astype(np.int16)
                code |= ((nb>=c).astype(np.uint8)<<bit)
            res[1:-1,1:-1]=code
            step=h//4; parts=[]
            for gy in range(4):
                for gx in range(4):
                    hc,_=np.histogram(res[gy*step:(gy+1)*step,gx*step:(gx+1)*step].ravel(),
                                      bins=32,range=(0,256))
                    parts.append(hc.astype(np.float32))
            feat=np.concatenate(parts); n=np.linalg.norm(feat)
            return feat/(n+1e-7)
        except Exception:
            return np.zeros(512,dtype=np.float32)

    @staticmethod
    def _mhist(img):
        try:
            h,w=img.shape; sy,sx=h//3,w//3; parts=[]
            for gy in range(3):
                for gx in range(3):
                    hc=cv2.calcHist([img[gy*sy:(gy+1)*sy,gx*sx:(gx+1)*sx]],
                                    [0],None,[32],[0,256]).flatten()
                    parts.append(hc/(np.linalg.norm(hc)+1e-7))
            return np.concatenate(parts).astype(np.float32)
        except Exception:
            return np.zeros(288,dtype=np.float32)

    def extract(self, gray, rect):
        try:
            crop=self._crop(self.boost(gray),rect)
            if crop is None: return None
            return {"lbp":self._lbp(crop),"mhist":self._mhist(crop),
                    "des":self.orb.detectAndCompute(crop,None)[1]}
        except Exception:
            return None

    @staticmethod
    def _cos(a,b):
        try:
            n=np.linalg.norm(a)*np.linalg.norm(b)
            return float(np.dot(a,b)/(n+1e-7)) if n>0 else 0.0
        except Exception:
            return 0.0

    def similarity(self, a, b):
        try:
            s = 0.45*max(0.,self._cos(a["lbp"],b["lbp"])) + \
                0.45*max(0.,self._cos(a["mhist"],b["mhist"]))
            da,db=a.get("des"),b.get("des")
            if da is not None and db is not None and len(da)>4 and len(db)>4:
                try:
                    ms=self.bf.knnMatch(da,db,k=2)
                    good=sum(1 for m in ms if len(m)==2 and m[0].distance<0.75*m[1].distance)
                    s += 0.10*(good/max(len(ms),1))
                except Exception: pass
            return s
        except Exception:
            return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  USER DATABASE
# ══════════════════════════════════════════════════════════════════════════════
class UserDB:
    def __init__(self):
        os.makedirs(FACES_DIR, exist_ok=True)
        self._data: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(USERS_FILE)\]
            try:
                with open(USERS_FILE, "rb") as f:
                    self._data = pickle.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        with open(USERS_FILE, "wb") as f:
            pickle.dump(self._data, f)

    def names(self) -> List[str]:
        return [v["display"] for v in self._data.values()]

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def has(self, name: str) -> bool:
        return name.strip().lower() in self._data

    def add(self, name: str, features: list):
        self._data[name.strip().lower()] = {
            "display": name.strip(), "features": features
        }
        self._save()

    def delete(self, key: str):
        self._data.pop(key, None)
        self._save()

    def display(self, key: str) -> str:
        return self._data.get(key, {}).get("display", key)

    def all_features(self):
        for v in self._data.values():
            yield v["display"], v["features"]

    def count(self) -> int:
        return len(self._data)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA THREAD
# ══════════════════════════════════════════════════════════════════════════════
class CamThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame = None
        self.gray  = None
        self.ok    = False
        self._stop = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        self.ok = True
        while not self._stop.is_set():
            ret, frame = cap.read()
            if ret:
                self.frame = frame
                self.gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            time.sleep(0.033)
        cap.release()

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════════════
#  UI WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
class Btn:
    def __init__(self, rect, label, primary=False, danger=False):
        self.r       = pygame.Rect(rect)
        self.label   = label
        self.primary = primary
        self.danger  = danger
        self._a      = 0.0
        self.hover   = False

    def draw(self, surf, font):
        if self.danger:
            bg, hi, bo = (55,8,8), (120,20,20), (200,35,35)
        elif self.primary:
            bg, hi, bo = (50,10,130), (90,30,200), PURPLE
        else:
            bg, hi, bo = (16,15,38), (28,25,65), (42,40,82)
        self._a += 0.20 * ((1.0 if self.hover else 0.0) - self._a)
        pygame.draw.rect(surf, _blend(bg, hi, self._a), self.r, border_radius=10)
        pygame.draw.rect(surf, bo, self.r, 2, border_radius=10)
        lbl = font.render(self.label, True, TEXT)
        surf.blit(lbl, lbl.get_rect(center=self.r.center))

    def hit(self, events) -> bool:
        self.hover = self.r.collidepoint(pygame.mouse.get_pos())
        return any(
            e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.hover
            for e in events
        )


class TextBox:
    def __init__(self, rect, placeholder=""):
        self.r    = pygame.Rect(rect)
        self.text = ""
        self.ph   = placeholder
        self.active = False
        self._t   = 0.0

    def handle(self, events, dt):
        self._t += dt
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN:
                self.active = self.r.collidepoint(e.pos)
            if e.type == pygame.KEYDOWN and self.active:
                if e.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif e.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    self.active = False
                elif e.unicode.isprintable() and len(self.text) < 22:
                    self.text += e.unicode

    def draw(self, surf, font):
        bo = PURPLE if self.active else MUTED
        pygame.draw.rect(surf, (18,16,42), self.r, border_radius=8)
        pygame.draw.rect(surf, bo, self.r, 2, border_radius=8)
        display = self.text if self.text else self.ph
        col     = TEXT if self.text else MUTED
        lbl = font.render(display, True, col)
        surf.blit(lbl, (self.r.x+12, self.r.centery - lbl.get_height()//2))
        if self.active and int(self._t*2) % 2 == 0:
            cx = self.r.x + 12 + (lbl.get_width() if self.text else 0) + 2
            pygame.draw.line(surf, TEXT,
                             (cx, self.r.centery-10), (cx, self.r.centery+10), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  FACE AUTH SYSTEM  (public API)
# ══════════════════════════════════════════════════════════════════════════════
class FaceAuthSystem:
    def __init__(self):
        self.engine = FaceEngine()
        self.db     = UserDB()
        self._cam: Optional[CamThread] = None

    # ── camera ────────────────────────────────────────────────────────────────
    def _start_cam(self):
        if self._cam is None or not self._cam.is_alive():
            self._cam = CamThread()
            self._cam.start()
            time.sleep(0.5)

    def shutdown(self):
        if self._cam:
            self._cam.stop()
            self._cam = None

    # ── helpers ───────────────────────────────────────────────────────────────
    def _cam_surface(self, w, h) -> Tuple[Optional[pygame.Surface], list]:
        if not self._cam or not self._cam.ok or self._cam.frame is None:
            return None, []
        frame = cv2.resize(self._cam.frame.copy(), (w, h))
        gray  = cv2.resize(self._cam.gray, (w, h))
        rects = self.engine.detect(gray)
        for (x, y, fw, fh) in rects:
            cv2.rectangle(frame, (x,y), (x+fw, y+fh), (130,55,240), 2)
        self._last_gray       = gray
        self._last_brightness = self.engine.brightness(gray)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(rgb.swapaxes(0,1)), rects

    def _identify(self, rect) -> Tuple[Optional[str], float]:
        try:
            gray = getattr(self, '_last_gray', None)
            if gray is None: return None, 0.0
            feat = self.engine.extract(gray, rect)
            if feat is None: return None, 0.0
            best_name, best_sim = None, 0.0
            for dname, features in self.db.all_features():
                sims = sorted((self.engine.similarity(feat,f) for f in features),reverse=True)
                top  = sims[:max(1,len(sims)//2)]
                avg  = sum(top)/len(top)
                if avg > best_sim: best_sim, best_name = avg, dname
            return best_name, best_sim
        except Exception:
            return None, 0.0

    def _header(self, surf, W, t, title, f_big, f_sm):
        pulse = _blend(PURPLE, BLUE, 0.5 + 0.5*math.sin(t*1.8))
        pygame.draw.rect(surf, pulse, (0, 0, W, 5))
        tl = f_big.render(title, True, TEXT)
        surf.blit(tl, tl.get_rect(centerx=W//2, y=18))
        sl = f_sm.render("AI Maze  ·  Facial Recognition", True, MUTED)
        surf.blit(sl, sl.get_rect(centerx=W//2, y=68))
        pygame.draw.line(surf, _blend(PURPLE, BG, 0.55),
                         (W//2-280, 96), (W//2+280, 96), 1)

    # ══════════════════════════════════════════════════════════════════════════
    #  PUBLIC ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════
    def run(self, screen: pygame.Surface, clock: pygame.time.Clock) -> Optional[str]:
        self._start_cam()
        return self._hub(screen, clock)

    # ─────────────────────────────────────────────────────────────────────────
    #  HUB
    # ─────────────────────────────────────────────────────────────────────────
    def _hub(self, screen, clock) -> Optional[str]:
        W, H  = screen.get_size()
        f_big = _font(38, bold=True)
        f_med = _font(21, bold=True)
        f_ui  = _font(16)
        f_sm  = _font(13)

        CAM_W, CAM_H = 350, 262
        cam_x = W//2 - CAM_W//2
        cam_y = 116

        BW, BX = 320, W//2 - 160
        btn_login  = Btn((BX, cam_y+CAM_H+30,  BW, 52), "FACE LOGIN",        primary=True)
        btn_reg    = Btn((BX, cam_y+CAM_H+96,  BW, 52), "REGISTER NEW FACE")
        btn_manage = Btn((BX, cam_y+CAM_H+162, BW, 52), "MANAGE USERS")
        btn_guest  = Btn((BX, cam_y+CAM_H+232, BW, 46), "PLAY AS GUEST")

        t = 0.0
        msg, mc, mt = "", GREEN, 0.0

        while True:
            dt = clock.tick(60) / 1000.0
            t += dt
            mt = max(0.0, mt - dt)
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit

            cam_surf, rects = self._cam_surface(CAM_W, CAM_H)

            screen.fill(BG)
            self._header(screen, W, t, "FACE RECOGNITION LOGIN", f_big, f_sm)

            # camera panel
            cam_rect = pygame.Rect(cam_x, cam_y, CAM_W, CAM_H)
            pygame.draw.rect(screen, PANEL, cam_rect.inflate(6,6), border_radius=12)
            if cam_surf:
                screen.blit(cam_surf, (cam_x, cam_y))
            else:
                pygame.draw.rect(screen, (18,16,40), cam_rect, border_radius=8)
                nl = f_ui.render("Camera unavailable", True, MUTED)
                screen.blit(nl, nl.get_rect(center=cam_rect.center))
            pygame.draw.rect(screen, PURPLE, cam_rect, 2, border_radius=8)

            dot_col = GREEN if rects else MUTED
            dot_txt = f"  {len(rects)} face detected" if rects else "  No face detected"
            screen.blit(f_sm.render(dot_txt, True, dot_col), (cam_x+6, cam_y+CAM_H+6))
            uc = f_sm.render(f"Registered: {self.db.count()}", True, MUTED)
            screen.blit(uc, (cam_x+CAM_W-uc.get_width()-4, cam_y+CAM_H+6))

            for btn in (btn_login, btn_reg, btn_manage, btn_guest):
                btn.draw(screen, f_med)

            if msg and mt > 0:
                ml = f_ui.render(msg, True, _blend(BG, mc, min(1.0, mt/0.5)))
                screen.blit(ml, ml.get_rect(centerx=W//2, y=H-46))

            pygame.display.flip()

            if btn_login.hit(events):
                result = self._login(screen, clock)
                if result:
                    return result
                msg, mc, mt = "Face not recognised — try again or register.", RED, 3.5

            if btn_reg.hit(events):
                result = self._register(screen, clock)
                if result:
                    msg, mc, mt = f"Registered '{result}'!  You can now face-login.", GREEN, 4.0

            if btn_manage.hit(events):
                self._manage(screen, clock)

            if btn_guest.hit(events):
                return None

    # ─────────────────────────────────────────────────────────────────────────
    #  FACE LOGIN
    # ─────────────────────────────────────────────────────────────────────────
    def _login(self, screen, clock) -> Optional[str]:
        W, H  = screen.get_size()
        f_big = _font(34, bold=True)
        f_med = _font(20, bold=True)
        f_ui  = _font(16)
        f_sm  = _font(13)

        CAM_W, CAM_H = 400, 300
        cam_x = W//2 - CAM_W//2
        cam_y = 112

        btn_back = Btn((W//2-160, cam_y+CAM_H+170, 320, 50), "BACK")

        progress   = 0.0
        FILL_RATE  = 0.85   # fills in ~1.2 s of consistent match
        hit_name   = None
        locked     = False
        lock_timer = 0.0
        t          = 0.0
        msg, mc    = "Position your face — hold still", CYAN
        torch_on   = False

        while True:
            dt = clock.tick(60) / 1000.0
            t += dt
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    return None

            cam_surf, rects = self._cam_surface(CAM_W, CAM_H)

            if not locked:
                if rects and getattr(self, '_last_gray', None) is not None:
                    name, sim = self._identify(rects[0])
                    if sim >= MATCH_THRESHOLD and name:
                        progress = min(1.0, progress + FILL_RATE * dt)
                        msg, mc  = f"Scanning…  {int(progress*100)}%", CYAN
                        hit_name = name
                    else:
                        progress = max(0.0, progress - FILL_RATE * dt)
                        msg, mc  = "No match yet — keep looking at camera", YELLOW
                else:
                    progress = max(0.0, progress - FILL_RATE * 0.6 * dt)
                    msg, mc  = "No face detected", MUTED

                if progress >= 1.0 and hit_name:
                    locked, lock_timer = True, 1.6
                    msg, mc = f"Welcome back, {hit_name}!", GREEN
            else:
                lock_timer -= dt
                if lock_timer <= 0:
                    return hit_name

            # draw
            if torch_on:
                screen.fill(TORCH)
            else:
                screen.fill(BG)
                self._header(screen, W, t, "FACE LOGIN", f_big, f_sm)

            cam_rect = pygame.Rect(cam_x, cam_y, CAM_W, CAM_H)
            if not torch_on:
                pygame.draw.rect(screen, PANEL, cam_rect.inflate(6,6), border_radius=12)
            if cam_surf:
                screen.blit(cam_surf, (cam_x, cam_y))
            else:
                pygame.draw.rect(screen, (18,16,40), cam_rect, border_radius=8)
            bo = GREEN if locked else (CYAN if progress > 0.05 else PURPLE)
            pygame.draw.rect(screen, bo, cam_rect, 2, border_radius=8)

            if not torch_on:
                bar = pygame.Rect(cam_x, cam_y+CAM_H+14, CAM_W, 13)
                pygame.draw.rect(screen, DARK_PU, bar, border_radius=6)
                fw = int(CAM_W * progress)
                if fw > 0:
                    fc = GREEN if locked else _blend(BLUE, GREEN, progress)
                    pygame.draw.rect(screen, fc,
                                     pygame.Rect(cam_x, cam_y+CAM_H+14, fw, 13),
                                     border_radius=6)
                pygame.draw.rect(screen, MUTED, bar, 1, border_radius=6)

                ml = f_ui.render(msg, True, mc)
                screen.blit(ml, ml.get_rect(centerx=W//2, y=cam_y+CAM_H+36))

                names = self.db.names()
                if names:
                    hint = f_sm.render("Registered: " + ", ".join(names[:6]), True, MUTED)
                else:
                    hint = f_sm.render("No users registered — go back and register first.", True, RED)
                screen.blit(hint, hint.get_rect(centerx=W//2, y=cam_y+CAM_H+62))

            # torch button (always visible)
            t_label = "[TORCH ON]" if not torch_on else "[TORCH OFF]"
            t_col   = (255, 200, 0) if not torch_on else (60, 45, 0)
            t_bg    = (60,  45,  0) if not torch_on else (220,180, 0)
            t_rect  = pygame.Rect(cam_x + CAM_W + 14, cam_y, 110, 46)
            pygame.draw.rect(screen, t_bg,  t_rect, border_radius=10)
            pygame.draw.rect(screen, t_col, t_rect, 2, border_radius=10)
            screen.blit(f_sm.render(t_label, True, t_col),
                        f_sm.render(t_label, True, t_col).get_rect(center=t_rect.center))
            if any(e.type==pygame.MOUSEBUTTONDOWN and e.button==1
                   and t_rect.collidepoint(e.pos) for e in events):
                torch_on = not torch_on

            # low-light warning
            if not torch_on:
                bv = getattr(self, '_last_brightness', 255)
                if bv < LOW_LIGHT_THRESH:
                    warn = f_sm.render(
                        f"⚠  Too dark ({int(bv)}) — press TORCH ON or turn on a light",
                        True, AMBER)
                    wr = warn.get_rect(centerx=W//2, y=H-30)
                    pygame.draw.rect(screen, (40,25,0), wr.inflate(16,8), border_radius=6)
                    screen.blit(warn, wr)

            if not locked:
                btn_back.draw(screen, f_med)
                if btn_back.hit(events):
                    return None

            pygame.display.flip()

    # ─────────────────────────────────────────────────────────────────────────
    #  REGISTER
    # ─────────────────────────────────────────────────────────────────────────
    def _register(self, screen, clock) -> Optional[str]:
        W, H  = screen.get_size()
        f_big = _font(34, bold=True)
        f_med = _font(20, bold=True)
        f_ui  = _font(16)
        f_sm  = _font(13)

        CAM_W, CAM_H = 360, 270
        cam_x = W//2 - CAM_W//2
        cam_y = 168

        name_box    = TextBox((W//2-180, cam_y-52, 360, 40), "Enter your name…")
        name_box.active = True
        btn_capture = Btn((W//2-160, cam_y+CAM_H+24, 320, 52), "CAPTURE FACE", primary=True)
        btn_back    = Btn((W//2-160, cam_y+CAM_H+90, 320, 46), "BACK")

        phase   = "name"
        samples = []
        samp_t  = 0.0
        SAMP_IV = 0.32

        msg, mc, mt = "Type your name, then press Capture.", CYAN, 0.0
        torch_on    = False
        t = 0.0

        while True:
            dt = clock.tick(60) / 1000.0
            t += dt
            mt = max(0.0, mt - dt)
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    return None

            name_box.handle(events, dt)
            cam_surf, rects = self._cam_surface(CAM_W, CAM_H)

            if phase == "capturing":
                if rects and self._cam.gray is not None:
                    samp_t += dt
                    if samp_t >= SAMP_IV:
                        samp_t = 0.0
                        _rg = getattr(self, '_last_gray', None)
                        feat = self.engine.extract(_rg, rects[0]) if _rg is not None else None
                        if feat:
                            samples.append(feat)
                            msg, mc, mt = f"Capturing… {len(samples)}/{SAMPLES_NEEDED}", CYAN, 2.0
                        if len(samples) >= SAMPLES_NEEDED:
                            uname = name_box.text.strip()
                            self.db.add(uname, samples)
                            phase = "done"
                            msg, mc, mt = f"'{uname}' registered successfully!", GREEN, 99.0
                else:
                    msg, mc, mt = "No face visible — look at the camera.", RED, 2.0

            # draw
            if torch_on:
                screen.fill(TORCH)
            else:
                screen.fill(BG)
                self._header(screen, W, t, "REGISTER FACE", f_big, f_sm)
                name_box.draw(screen, f_ui)

            cam_rect = pygame.Rect(cam_x, cam_y, CAM_W, CAM_H)
            if not torch_on:
                pygame.draw.rect(screen, PANEL, cam_rect.inflate(6,6), border_radius=12)
            if cam_surf:
                screen.blit(cam_surf, (cam_x, cam_y))
            else:
                pygame.draw.rect(screen, (18,16,40), cam_rect, border_radius=8)
            bo = GREEN if phase=="done" else (CYAN if phase=="capturing" else PURPLE)
            pygame.draw.rect(screen, bo, cam_rect, 2, border_radius=8)

            if not torch_on:
                dot_y = cam_y + CAM_H + 12
                for i in range(SAMPLES_NEEDED):
                    col = GREEN if i < len(samples) else DARK_PU
                    pygame.draw.circle(screen, col,  (cam_x+20+i*38, dot_y), 8)
                    pygame.draw.circle(screen, MUTED, (cam_x+20+i*38, dot_y), 8, 1)
                if mt > 0:
                    ml = f_ui.render(msg, True, mc)
                    screen.blit(ml, ml.get_rect(centerx=W//2, y=cam_y+CAM_H+30))

            # torch button
            t_label = "[TORCH ON]" if not torch_on else "[TORCH OFF]"
            t_col   = (255, 200, 0) if not torch_on else (60, 45, 0)
            t_bg    = (60,  45,  0) if not torch_on else (220,180, 0)
            t_rect  = pygame.Rect(cam_x + CAM_W + 14, cam_y, 110, 46)
            pygame.draw.rect(screen, t_bg,  t_rect, border_radius=10)
            pygame.draw.rect(screen, t_col, t_rect, 2, border_radius=10)
            screen.blit(f_sm.render(t_label, True, t_col),
                        f_sm.render(t_label, True, t_col).get_rect(center=t_rect.center))
            if any(e.type==pygame.MOUSEBUTTONDOWN and e.button==1
                   and t_rect.collidepoint(e.pos) for e in events):
                torch_on = not torch_on

            if not torch_on:
                bv = getattr(self, '_last_brightness', 255)
                if bv < LOW_LIGHT_THRESH:
                    warn = f_sm.render(
                        f"⚠  Very dark ({int(bv)}) — use TORCH ON for best results",
                        True, AMBER)
                    wr = warn.get_rect(centerx=W//2, y=cam_y+CAM_H+52)
                    pygame.draw.rect(screen, (40,25,0), wr.inflate(16,8), border_radius=6)
                    screen.blit(warn, wr)

            if phase == "done":
                cont = Btn((W//2-160, cam_y+CAM_H+90, 320, 52), "CONTINUE  →", primary=True)
                cont.draw(screen, f_med)
                if cont.hit(events):
                    return name_box.text.strip()
            else:
                btn_capture.draw(screen, f_med)
                btn_back.draw(screen, f_med)
                if btn_back.hit(events):
                    return None
                if btn_capture.hit(events):
                    uname = name_box.text.strip()
                    if not uname:
                        msg, mc, mt = "Enter your name first.", RED, 2.5
                    elif not rects:
                        msg, mc, mt = "No face detected — look at the camera.", RED, 2.5
                    elif self.db.has(uname):
                        msg, mc, mt = (f"'{uname}' already exists. "
                                       "Delete them in Manage Users to re-register."), ORANGE, 3.0
                    else:
                        phase, samples, samp_t = "capturing", [], 0.0
                        msg, mc, mt = "Hold still — capturing…", CYAN, 99.0

            pygame.display.flip()

    # ─────────────────────────────────────────────────────────────────────────
    #  MANAGE USERS
    # ─────────────────────────────────────────────────────────────────────────
    def _manage(self, screen, clock):
        W, H  = screen.get_size()
        f_big = _font(34, bold=True)
        f_med = _font(20, bold=True)
        f_ui  = _font(16)
        f_sm  = _font(13)

        btn_back = Btn((W//2-160, H-76, 320, 50), "BACK")
        t = 0.0
        msg, mc, mt = "", GREEN, 0.0

        while True:
            dt = clock.tick(60) / 1000.0
            t += dt
            mt = max(0.0, mt - dt)
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    return

            screen.fill(BG)
            self._header(screen, W, t, "MANAGE USERS", f_big, f_sm)

            keys = self.db.keys()
            if not keys:
                nl = f_ui.render("No registered users.", True, MUTED)
                screen.blit(nl, nl.get_rect(centerx=W//2, y=240))
            else:
                for i, key in enumerate(keys):
                    disp = self.db.display(key)
                    ry   = 148 + i * 58
                    pygame.draw.rect(screen, PANEL, (W//2-270, ry, 540, 46), border_radius=8)
                    pygame.draw.rect(screen, MUTED,  (W//2-270, ry, 540, 46), 1, border_radius=8)
                    nl = f_med.render(disp, True, TEXT)
                    screen.blit(nl, (W//2-250, ry+11))
                    del_btn = Btn((W//2+170, ry+7, 90, 32), "DELETE", danger=True)
                    del_btn.draw(screen, f_sm)
                    if del_btn.hit(events):
                        self.db.delete(key)
                        msg, mc, mt = f"Deleted '{disp}'.", ORANGE, 3.0

            if msg and mt > 0:
                ml = f_ui.render(msg, True, mc)
                screen.blit(ml, ml.get_rect(centerx=W//2, y=H-110))

            btn_back.draw(screen, f_med)
            if btn_back.hit(events):
                return

            pygame.display.flip()
