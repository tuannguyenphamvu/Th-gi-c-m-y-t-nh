import pygame
import random
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pygame.locals import *

import os

# Đảm bảo working directory luôn là thư mục chứa script này
# (tránh lỗi khi chạy từ thư mục khác)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── GAME VARIABLES ──────────────────────────────────────────────────────────
SCREEN_WIDTH  = 400
SCREEN_HEIGHT = 600
GRAVITY       = 1
GAME_SPEED    = 4
GROUND_WIDTH  = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100
PIPE_WIDTH    = 80
PIPE_HEIGHT   = 500
PIPE_GAP      = 150

wing = 'assets/audio/wing.wav'
hit  = 'assets/audio/hit.wav'

pygame.init()
pygame.mixer.init()

# ─── CAMERA ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

# ─── HAND (ngón cái) ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1)
mp_draw  = mp.solutions.drawing_utils

# ─── FACE LANDMARKER dùng file face_landmarker.task ──────────────────────────
# Model này có sẵn blendshapes: eyeBlinkLeft, eyeBlinkRight
# Chính xác hơn nhiều so với tự tính EAR thủ công

# MediaPipe trên Windows có bug với absolute path chứa ':\'.
# Fix: đọc model thành bytes rồi truyền qua model_asset_buffer.
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker_v2_with_blendshapes.task')
with open(_MODEL_PATH, 'rb') as _f:
    _MODEL_BYTES = _f.read()
base_options = python.BaseOptions(model_asset_buffer=_MODEL_BYTES)
face_options  = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,               # bật blendshape để lấy eyeBlink
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE        # xử lý từng frame
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# Ngưỡng blendshape eyeBlink (0.0 = mở hoàn toàn, 1.0 = nhắm hoàn toàn)
BLINK_THRESHOLD = 0.40
BLINK_FRAMES    = 2      # số frame nhắm liên tiếp để xác nhận nháy

blink_counter = 0


def detect_controls():
    """Đọc camera, trả về (thumb_up, blink)."""
    global blink_counter

    ret, frame = cap.read()
    if not ret:
        return False, False

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Ngón cái (Hand Landmarks) ─────────────────────────────────────────
    hand_result = hands.process(rgb)
    thumb_up = False
    if hand_result.multi_hand_landmarks:
        for hlm in hand_result.multi_hand_landmarks:
            # landmark 4 = đầu ngón cái, 3 = khớp trên ngón cái
            if hlm.landmark[4].y < hlm.landmark[3].y:
                thumb_up = True
            mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

    # ── Nháy mắt (Face Landmarker .task) ─────────────────────────────────
    mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    face_result = face_landmarker.detect(mp_image)

    blink      = False
    eye_closed = False
    hud_text   = "NO FACE"
    hud_color  = (120, 120, 120)

    if face_result.face_landmarks and face_result.face_blendshapes:
        # Lấy blendshape score eyeBlinkLeft và eyeBlinkRight
        bs_dict = {b.category_name: b.score
                   for b in face_result.face_blendshapes[0]}

        blink_l = bs_dict.get('eyeBlinkLeft',  0.0)
        blink_r = bs_dict.get('eyeBlinkRight', 0.0)
        avg     = (blink_l + blink_r) / 2.0

        eye_closed = avg > BLINK_THRESHOLD
        hud_text   = f"eyeBlink L:{blink_l:.2f} R:{blink_r:.2f}"
        hud_color  = (0, 0, 255) if eye_closed else (0, 255, 0)

        # Xác nhận nháy khi mắt mở lại sau ít nhất BLINK_FRAMES frame nhắm
        if eye_closed:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_FRAMES:
                blink = True
            blink_counter = 0

        # Vẽ một số landmark mắt cho trực quan
        lm = face_result.face_landmarks[0]
        for idx in [33, 133, 159, 145, 263, 362, 386, 374]:
            px, py = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

    # ── HUD trên cửa sổ camera ────────────────────────────────────────────
    cv2.putText(frame, hud_text,
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
    status = "BLINK!" if blink else ("CLOSED" if eye_closed else "OPEN")
    cv2.putText(frame, status,
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)
    if thumb_up:
        cv2.putText(frame, "THUMB UP",
                    (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(frame, "Blink or Thumb UP = JUMP",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)

    cv2.imshow("Flappy Bird - Camera", frame)
    cv2.waitKey(1)

    return thumb_up, blink


# ─── SPRITE CLASSES ──────────────────────────────────────────────────────────

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha(),
        ]
        self.current_image = 0
        self.image = self.images[0]
        self.mask  = pygame.mask.from_surface(self.image)
        self.rect  = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2
        self.speed = 0

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed
        if self.rect[1] < 0:
            self.rect[1] = 0

    def bump(self):
        self.speed = -10


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        if inverted:
            self.image   = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask   = pygame.mask.from_surface(self.image)
        self.passed = False

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))
        self.mask  = pygame.mask.from_surface(self.image)
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):
    size = random.randint(100, 300)
    return (Pipe(False, xpos, size),
            Pipe(True,  xpos, SCREEN_HEIGHT - size - PIPE_GAP))


def reset_game(bird, pipe_group):
    bird.rect[1] = SCREEN_HEIGHT / 2
    bird.speed   = 0
    pipe_group.empty()
    for i in range(2):
        p, pi = get_random_pipes(SCREEN_WIDTH * i + 600)
        pipe_group.add(p, pi)


# ─── PYGAME SETUP ────────────────────────────────────────────────────────────

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Face Landmarker + Hand")

BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))

font       = pygame.font.SysFont("Arial", 40)
small_font = pygame.font.SysFont("Arial", 18)

bird = Bird()
bird_group   = pygame.sprite.Group(bird)
ground_group = pygame.sprite.Group(*[Ground(GROUND_WIDTH * i) for i in range(2)])
pipe_group   = pygame.sprite.Group()
for i in range(2):
    p, pi = get_random_pipes(SCREEN_WIDTH * i + 600)
    pipe_group.add(p, pi)

clock      = pygame.time.Clock()
score      = 0
last_thumb = False
last_blink = False
game_over  = False

# ─── GAME LOOP ───────────────────────────────────────────────────────────────

while True:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            face_landmarker.close()
            exit()
        if event.type == KEYDOWN:
            if event.key == K_SPACE and not game_over:
                bird.bump()
            if event.key == K_r and game_over:
                reset_game(bird, pipe_group)
                score = 0
                game_over = False

    thumb_up, blink = detect_controls()

    # Kích hoạt nhảy ở cạnh lên của tín hiệu (tránh giữ liên tục)
    jump = (thumb_up and not last_thumb) or (blink and not last_blink)

    # Restart khi game over
    if game_over and thumb_up:
        reset_game(bird, pipe_group)
        score = 0
        game_over = False

    if jump and not game_over:
        bird.bump()
        pygame.mixer.music.load(wing)
        pygame.mixer.music.play()

    last_thumb = thumb_up
    last_blink = blink

    # ── Render ──────────────────────────────────────────────────────────────
    screen.blit(BACKGROUND, (0, 0))

    if not game_over:
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            ground_group.add(Ground(GROUND_WIDTH - 20))

        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])
            p, pi = get_random_pipes(SCREEN_WIDTH * 2)
            pipe_group.add(p, pi)

        bird_group.update()
        ground_group.update()
        pipe_group.update()

        for pipe in pipe_group:
            if not pipe.passed and pipe.rect[0] < bird.rect[0]:
                pipe.passed = True
                score += 0.5

    bird_group.draw(screen)
    pipe_group.draw(screen)
    ground_group.draw(screen)

    sc = font.render(str(int(score)), True, (255, 255, 255))
    screen.blit(sc, (SCREEN_WIDTH // 2 - sc.get_width() // 2, 50))

    # Indicator điều khiển
    screen.blit(small_font.render("👍 Thumb", True,
                (0, 255, 100) if thumb_up else (180, 180, 180)), (10, 10))
    screen.blit(small_font.render("😉 Blink", True,
                (0, 200, 255) if blink    else (180, 180, 180)), (10, 32))

    if game_over:
        ot = font.render("GAME OVER", True, (255, 50, 50))
        rt = small_font.render("Raise thumb or press R to restart", True, (255, 255, 255))
        screen.blit(ot, (SCREEN_WIDTH//2 - ot.get_width()//2, SCREEN_HEIGHT//2 - 40))
        screen.blit(rt, (SCREEN_WIDTH//2 - rt.get_width()//2, SCREEN_HEIGHT//2 + 10))

    pygame.display.update()

    if not game_over and (
        pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask)
        or
        pygame.sprite.groupcollide(bird_group, pipe_group,   False, False, pygame.sprite.collide_mask)
    ):
        pygame.mixer.music.load(hit)
        pygame.mixer.music.play()
        game_over = True