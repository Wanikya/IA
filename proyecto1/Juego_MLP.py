#!/usr/bin/env python3
"""
Juego de esquivar pelotas usando un Perceptrón Multicapa (MLPClassifier)
"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import pygame
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE_W, BASE_H = 1280, 800
WINDOW_FRACTION = 0.90
N_PELOTAS = 2
CSV_NAME = "datos_esquivar.csv"

ACCION_QUIETO = 0
ACCION_ARR = 1
ACCION_ABA = 2
ACCION_IZQ = 3
ACCION_DER = 4
ACCION_ARR_IZQ = 5
ACCION_ARR_DER = 6
ACCION_ABA_IZQ = 7
ACCION_ABA_DER = 8


def dirs_a_accion(dx: int, dy: int) -> int:
    return {
        (0, 0): ACCION_QUIETO,
        (0, -1): ACCION_ARR,
        (0, 1): ACCION_ABA,
        (-1, 0): ACCION_IZQ,
        (1, 0): ACCION_DER,
        (-1, -1): ACCION_ARR_IZQ,
        (1, -1): ACCION_ARR_DER,
        (-1, 1): ACCION_ABA_IZQ,
        (1, 1): ACCION_ABA_DER,
    }.get((dx, dy), ACCION_QUIETO)


def accion_a_dirs(accion: int) -> Tuple[int, int]:
    return {
        ACCION_QUIETO: (0, 0),
        ACCION_ARR: (0, -1),
        ACCION_ABA: (0, 1),
        ACCION_IZQ: (-1, 0),
        ACCION_DER: (1, 0),
        ACCION_ARR_IZQ: (-1, -1),
        ACCION_ARR_DER: (1, -1),
        ACCION_ABA_IZQ: (-1, 1),
        ACCION_ABA_DER: (1, 1),
    }.get(accion, (0, 0))


def tamano_ventana() -> Tuple[int, int]:
    pygame.display.init()
    info = pygame.display.Info()
    sw = info.current_w or BASE_W
    sh = info.current_h or BASE_H
    w = min(max(BASE_W, int(sw * WINDOW_FRACTION)), max(640, sw - 40))
    h = min(max(BASE_H, int(sh * WINDOW_FRACTION)), max(480, sh - 80))
    return w, h


@dataclass
class Sample:
    ball1_dx: float
    ball1_dy: float
    ball1_vx: float
    ball1_vy: float
    ball2_dx: float
    ball2_dy: float
    ball2_vx: float
    ball2_vy: float
    chase_dx: float
    chase_dy: float
    player_x: float
    player_y: float
    move_dx: int
    move_dy: int
    accion: int


FEATURE_NAMES = [
    "ball1_dx", "ball1_dy", "ball1_vx", "ball1_vy",
    "ball2_dx", "ball2_dy", "ball2_vx", "ball2_vy",
    "chase_dx", "chase_dy", "player_x", "player_y",
    "move_dx", "move_dy", "accion",
]


@dataclass
class Pelota:
    rect: pygame.Rect
    vx: float
    vy: float


@dataclass
class Perseguidor:
    x: float
    y: float
    radio: float
    velocidad: float

    def hitbox(self) -> pygame.Rect:
        r = int(self.radio)
        return pygame.Rect(int(self.x) - r, int(self.y) - r, r * 2, r * 2)


class Juego:
    def __init__(self) -> None:
        pygame.init()
        self.w, self.h = tamano_ventana()
        self._ventana = (self.w, self.h)
        self.pantalla = pygame.display.set_mode(self._ventana)
        pygame.display.set_caption("Esquivar pelotas — MLP (Red Neuronal)")
        self._fullscreen = False

        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        self.scale = max(1.0, min(self.w / BASE_W, self.h / BASE_H) * 1.15)
        self.speed = max(3, int(6 * self.scale))
        self.player_size = (int(36 * self.scale), int(52 * self.scale))
        self.ball_size = (int(22 * self.scale), int(22 * self.scale))

        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))
        self._cargar_assets()

        self.corriendo = True
        self.modo_auto = False
        self.datos: List[Sample] = []
        self.colisiones = 0
        self.frames_vivos = 0
        self.frame_i = 0
        self.frame_tick = 0

        # Componentes de Aprendizaje Profundo
        self.scaler = StandardScaler()
        self.modelo: Optional[MLPClassifier] = None
        self.modelo_entrenado = False

        self.jugador = pygame.Rect(0, 0, *self.player_size)
        self.pelotas: List[Pelota] = []
        self.perseguidor: Optional[Perseguidor] = None
        self._reset()

    def _cargar_assets(self) -> None:
        def load(path: str, size: Tuple[int, int], color=(180, 180, 180)) -> pygame.Surface:
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                s = pygame.Surface(size, pygame.SRCALPHA)
                s.fill((*color, 255))
                return s

        base = os.path.dirname(os.path.abspath(__file__))
        sp = os.path.join(base, "assets", "sprites")
        gm = os.path.join(base, "assets", "game")
        self.frames = [
            load(os.path.join(sp, f"mono_frame_{i}.png"), self.player_size) for i in (1, 2, 3, 4)
        ]
        self.ball_img = load(os.path.join(sp, "purple_ball.png"), self.ball_size, (160, 100, 255))
        self.ball_img2 = self.ball_img.copy()
        self.ball_img2.fill((255, 120, 40, 80), special_flags=pygame.BLEND_RGBA_ADD)
        self.fondo = load(os.path.join(gm, "fondo2.png"), (self.w, self.h), (30, 35, 50))

    def _reset(self) -> None:
        self.jugador.center = (self.w // 2, self.h // 2)
        self.frames_vivos = 0
        self.pelotas.clear()
        for i in range(N_PELOTAS):
            side = -1 if i % 2 == 0 else 1
            x = int(self.w * 0.1) if side < 0 else int(self.w * 0.9)
            y = random.randint(40, self.h - 40)
            vx = random.uniform(3.5, 7.5) * self.scale * side
            vy = random.uniform(-5.0, 5.0) * self.scale
            if abs(vy) < 1.5 * self.scale:
                vy = 2.5 * self.scale * (1 if i % 2 == 0 else -1)
            self.pelotas.append(Pelota(pygame.Rect(x, y, *self.ball_size), vx, vy))

        ang = random.uniform(0, math.tau)
        dist = min(self.w, self.h) * 0.35
        self.perseguidor = Perseguidor(
            x=self.w / 2 + math.cos(ang) * dist,
            y=self.h / 2 + math.sin(ang) * dist,
            radio=28 * self.scale,
            velocidad=2.8 * self.scale,
        )

    def _dir_teclado(self, keys) -> Tuple[int, int]:
        dx = dy = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += 1
        return max(-1, min(1, dx)), max(-1, min(1, dy))

    def _mover_jugador(self, dx: int, dy: int) -> None:
        self.jugador.x += dx * self.speed
        self.jugador.y += dy * self.speed
        self.jugador.clamp_ip(self.pantalla.get_rect())

    def _mover_pelotas(self) -> None:
        for p in self.pelotas:
            p.rect.x += int(p.vx)
            p.rect.y += int(p.vy)
            if p.rect.left <= 0 or p.rect.right >= self.w:
                p.vx *= -1
                p.rect.x = max(0, min(p.rect.x, self.w - p.rect.w))
            if p.rect.top <= 0 or p.rect.bottom >= self.h:
                p.vy *= -1
                p.rect.y = max(0, min(p.rect.y, self.h - p.rect.h))

    def _mover_perseguidor(self) -> None:
        assert self.perseguidor is not None
        dx = self.jugador.centerx - self.perseguidor.x
        dy = self.jugador.centery - self.perseguidor.y
        dist = math.hypot(dx, dy) or 1.0
        self.perseguidor.x += self.perseguidor.velocidad * dx / dist
        self.perseguidor.y += self.perseguidor.velocidad * dy / dist
        r = self.perseguidor.radio
        self.perseguidor.x = max(r, min(self.w - r, self.perseguidor.x))
        self.perseguidor.y = max(r, min(self.h - r, self.perseguidor.y))

    def _colision(self) -> bool:
        assert self.perseguidor is not None
        if self.jugador.colliderect(self.perseguidor.hitbox()):
            return True
        return any(self.jugador.colliderect(p.rect) for p in self.pelotas)

    def _feat_pelota(self, p: Pelota) -> Tuple[float, float, float, float]:
        return (
            float(p.rect.centerx - self.jugador.centerx),
            float(p.rect.centery - self.jugador.centery),
            float(p.vx),
            float(p.vy),
        )

    def construir_sample(self, move_dx: int, move_dy: int) -> Sample:
        assert self.perseguidor is not None
        f: List[float] = []
        for i in range(2):
            if i < len(self.pelotas):
                f.extend(self._feat_pelota(self.pelotas[i]))
            else:
                f.extend([0.0, 0.0, 0.0, 0.0])
        return Sample(
            ball1_dx=f[0],
            ball1_dy=f[1],
            ball1_vx=f[2],
            ball1_vy=f[3],
            ball2_dx=f[4],
            ball2_dy=f[5],
            ball2_vx=f[6],
            ball2_vy=f[7],
            chase_dx=float(self.perseguidor.x - self.jugador.centerx),
            chase_dy=float(self.perseguidor.y - self.jugador.centery),
            player_x=float(self.jugador.centerx),
            player_y=float(self.jugador.centery),
            move_dx=int(move_dx),
            move_dy=int(move_dy),
            accion=dirs_a_accion(move_dx, move_dy),
        )

    def registrar(self, move_dx: int, move_dy: int) -> None:
        self.datos.append(self.construir_sample(move_dx, move_dy))

    def exportar_csv(self) -> str:
        if not self.datos:
            return "No hay datos."
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_NAME)
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FEATURE_NAMES)
            w.writeheader()
            for s in self.datos:
                w.writerow(asdict(s))
        return f"Guardado {CSV_NAME} ({len(self.datos)} filas)."

    # ---- ENTRENAMIENTO PERCEPTRÓN MULTICAPA (MLP) ----
    def entrenar_modelo(self) -> str:
        if len(self.datos) < 100:
            return "Se necesitan más datos para la red neuronal (mínimo 100-200 frames)."

        # 1. Convertir la memoria a DataFrame
        df = pd.DataFrame([asdict(s) for s in self.datos])

        feature_cols = [
            "ball1_dx", "ball1_dy", "ball1_vx", "ball1_vy",
            "ball2_dx", "ball2_dy", "ball2_vx", "ball2_vy",
            "chase_dx", "chase_dy", "player_x", "player_y"
        ]
        X = df[feature_cols]
        y = df["accion"]

        # 2. Escalar características (Crucial para redes neuronales)
        X_scaled = self.scaler.fit_transform(X)

        # 3. Configurar MLP (2 capas ocultas: 64 y 32 neuronas)
        self.modelo = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=400,
            random_state=42
        )
        self.modelo.fit(X_scaled, y)
        self.modelo_entrenado = True

        return f"¡MLP entrenado con {len(df)} registros!"

    # ---- DECISIÓN AUTOMÁTICA CON RED NEURONAL ----
    def decision_auto(self) -> Tuple[int, int]:
        if not self.modelo_entrenado or self.modelo is None:
            return 0, 0

        # Formatear el estado actual
        s_act = self.construir_sample(0, 0)
        features = [[
            s_act.ball1_dx, s_act.ball1_dy, s_act.ball1_vx, s_act.ball1_vy,
            s_act.ball2_dx, s_act.ball2_dy, s_act.ball2_vx, s_act.ball2_vy,
            s_act.chase_dx, s_act.chase_dy,
            s_act.player_x, s_act.player_y
        ]]

        # Aplicar el mismo escalador del entrenamiento
        features_scaled = self.scaler.transform(features)

        # Predecir con la red neuronal
        accion_predicha = int(self.modelo.predict(features_scaled)[0])
        return accion_a_dirs(accion_predicha)

    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ — Esquivar (Red Neuronal MLP)", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(50 * self.scale)))

        lineas = [
            "M — Manual (reinicia dataset)",
            "A — Auto (juega con el MLP)",
            "T — Entrenar MLP (Red Neuronal)",
            "C — Exportar CSV",
            "F — Fullscreen",
            "Q — Salir",
        ]
        x0 = int(60 * self.scale)
        y = int(120 * self.scale)
        for op in lineas:
            self.pantalla.blit(self.fuente_chica.render(op, True, self.BLANCO), (x0, y))
            y += self.fuente_chica.get_linesize() + 4

        y += 10
        estado_mod = "LISTO" if self.modelo_entrenado else "SIN ENTRENAR"
        self.pantalla.blit(
            self.fuente_chica.render(f"Memoria: {len(self.datos)} frames | Modelo: {estado_mod}", True, self.GRIS),
            (x0, y),
        )
        if msg:
            self.pantalla.blit(
                self.fuente_chica.render(msg, True, self.AMARILLO),
                (x0, y + self.fuente_chica.get_linesize() + 8),
            )
        pygame.display.flip()

    def mostrar_menu(self) -> None:
        msg = ""
        esperando = True
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos.clear()
                        self.modelo = None
                        self.modelo_entrenado = False
                        self.colisiones = 0
                        self._reset()
                        esperando = False
                    elif e.key == pygame.K_a:
                        self.modo_auto = True
                        self._reset()
                        esperando = False
                    elif e.key == pygame.K_t:
                        msg = self.entrenar_modelo()
                    elif e.key == pygame.K_c:
                        msg = self.exportar_csv()
                    elif e.key == pygame.K_f:
                        self._toggle_fs()
                    elif e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False

    def _toggle_fs(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            self.pantalla = pygame.display.set_mode(
                (info.current_w, info.current_h), pygame.FULLSCREEN
            )
        else:
            self.pantalla = pygame.display.set_mode(self._ventana)
        self.w, self.h = self.pantalla.get_size()
        self.fondo = pygame.transform.smoothscale(self.fondo, (self.w, self.h))
        self._reset()

    def _dibujar(self) -> None:
        self.pantalla.blit(self.fondo, (0, 0))
        pygame.draw.line(
            self.pantalla, (255, 255, 255), (self.w // 2, self.h // 2 - 10), (self.w // 2, self.h // 2 + 10), 1
        )
        pygame.draw.line(
            self.pantalla, (255, 255, 255), (self.w // 2 - 10, self.h // 2), (self.w // 2 + 10, self.h // 2), 1
        )

        self.frame_tick += 1
        if self.frame_tick >= 8:
            self.frame_i = (self.frame_i + 1) % len(self.frames)
            self.frame_tick = 0
        self.pantalla.blit(self.frames[self.frame_i], self.jugador.topleft)

        for i, p in enumerate(self.pelotas):
            self.pantalla.blit(self.ball_img if i % 2 == 0 else self.ball_img2, p.rect.topleft)

        assert self.perseguidor is not None
        pygame.draw.circle(
            self.pantalla,
            (220, 60, 60),
            (int(self.perseguidor.x), int(self.perseguidor.y)),
            int(self.perseguidor.radio),
        )
        pygame.draw.circle(
            self.pantalla,
            (255, 200, 200),
            (int(self.perseguidor.x), int(self.perseguidor.y)),
            int(self.perseguidor.radio),
            3,
        )

        modo = "AUTO (MLP)" if self.modo_auto else "MANUAL"
        hud = f"[{modo}] {len(self.datos)}  golpes={self.colisiones}"
        self.pantalla.blit(self.fuente_chica.render(hud, True, self.AMARILLO), (10, 8))

    def jugar(self) -> None:
        reloj = pygame.time.Clock()
        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_p):
                        return
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        return
                    if e.key == pygame.K_f:
                        self._toggle_fs()

            keys = pygame.key.get_pressed()
            if self.modo_auto:
                dx, dy = self.decision_auto()
            else:
                dx, dy = self._dir_teclado(keys)
                self.registrar(dx, dy)

            self._mover_jugador(dx, dy)
            self._mover_pelotas()
            self._mover_perseguidor()
            self.frames_vivos += 1

            if self._colision():
                self.colisiones += 1
                self._reset()

            self._dibujar()
            pygame.display.flip()
            reloj.tick(45)

    def loop(self) -> None:
        while self.corriendo:
            self.mostrar_menu()
            if self.corriendo:
                self.jugar()
        pygame.quit()


if __name__ == "__main__":
    Juego().loop()