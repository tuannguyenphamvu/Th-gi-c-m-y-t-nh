import pygame
import random
import time
import cv2
import mediapipe as mp

from pygame.locals import *

# GAME VARIABLES
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

SPEED = 10
GRAVITY = 1
GAME_SPEED = 4

GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100

PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150

wing = 'assets/audio/wing.wav'
hit = 'assets/audio/hit.wav'

pygame.init()
pygame.mixer.init()

# CAMERA SETUP
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


def finger_up():
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    thumb_up = False

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            thumb_tip = hand.landmark[4]
            thumb_ip = hand.landmark[3]

            # nếu ngón cái đưa lên
            if thumb_tip.y < thumb_ip.y:
                thumb_up = True

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Camera Control", frame)
    cv2.waitKey(1)

    return thumb_up


class Bird(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()
        ]

        self.current_image = 0
        self.image = self.images[0]

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
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

        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:

            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)

        else:

            self.rect[1] = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

        self.passed = False

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):

    def __init__(self, xpos):

        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])


def get_random_pipes(xpos):

    size = random.randint(100, 300)

    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)

    return pipe, pipe_inverted


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird Hand Control")

BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))

font = pygame.font.SysFont("Arial", 40)

bird_group = pygame.sprite.Group()
bird = Bird()
bird_group.add(bird)

ground_group = pygame.sprite.Group()

for i in range(2):
    ground = Ground(GROUND_WIDTH * i)
    ground_group.add(ground)

pipe_group = pygame.sprite.Group()

for i in range(2):
    pipes = get_random_pipes(SCREEN_WIDTH * i + 600)
    pipe_group.add(pipes[0])
    pipe_group.add(pipes[1])

clock = pygame.time.Clock()

score = 0
last_finger = False
game_over = False

while True:

    clock.tick(30)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            exit()

    finger = finger_up()
    # restart game bằng tay
    if game_over and finger:
        
        bird.rect[1] = SCREEN_HEIGHT / 2
        bird.speed = 0
        
        pipe_group.empty()
        
        for i in range(2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 600)
            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])

        score = 0
        game_over = False

    if finger and not last_finger and not game_over:
        bird.bump()
        pygame.mixer.music.load(wing)
        pygame.mixer.music.play()

    last_finger = finger

    screen.blit(BACKGROUND, (0, 0))

    if not game_over:

        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDTH - 20)
            ground_group.add(new_ground)

        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])

            pipes = get_random_pipes(SCREEN_WIDTH * 2)

            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])

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

    score_text = font.render(str(int(score)), True, (255, 255, 255))
    screen.blit(score_text, (SCREEN_WIDTH / 2, 50))

    pygame.display.update()

    if not game_over and (
        pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask)
        or
        pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)
    ):

        pygame.mixer.music.load(hit)
        pygame.mixer.music.play()

        game_over = True