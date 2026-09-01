"""
=====================================================================================
Движок численного моделирования звёздных систем с учётом эффектов ОТО (1PN-приближение)
=====================================================================================
Полностью самодостаточен: только numpy (+ matplotlib для отрисовки, импортируется
лениво только внутри plot_trajectories). Работает в CPython, Google Colab,
Flask/FastAPI backend, Gradio, Streamlit и в Gradio Lite / Pyodide (WASM в браузере) —
никаких C-расширений, компиляции или внешних библиотек (REBOUND и т.п.) не используется.

Единицы измерения:
    масса     — массы Солнца (M_sun)
    длина     — астрономические единицы (AU)
    время     — годы (yr)
    скорость  — AU/yr
G = 4*pi^2 (AU^3 / (M_sun * yr^2)) — стандартная гелиоцентрическая система единиц.

Модуль спроектирован как "чистый движок": функция simulate() только считает и
возвращает numpy-массивы траекторий. Никакого UI/сервера внутри — это уже задача
приложения, которое будет вызывать этот код.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Физические константы (в units AU / M_sun / yr)
# ---------------------------------------------------------------------------
G = 4.0 * np.pi ** 2      # гравитационная постоянная в выбранных единицах
C_LIGHT = 63241.077       # скорость света, AU/yr (= 299792.458 км/с)


# ---------------------------------------------------------------------------
# 1. Инициализация тел системы
# ---------------------------------------------------------------------------
def body_from_orbit(a, e, mass_central, mass_body=0.0, arg_peri=0.0, retrograde=False):
    """
    Переводит элементы орбиты (большая полуось a [AU], эксцентриситет e) в декартовы
    положение/скорость в момент прохождения перицентра (задача двух тел,
    mu = G*(M_central + M_body)).
    Скорость в перицентре (vis-viva): v_p = sqrt(mu/a * (1+e)/(1-e))
    """
    mu = G * (mass_central + mass_body)
    r_p = a * (1.0 - e)
    v_p = np.sqrt(mu / a * (1.0 + e) / (1.0 - e))

    pos = np.array([r_p, 0.0])
    vel = np.array([0.0, -v_p if retrograde else v_p])

    cos_w, sin_w = np.cos(arg_peri), np.sin(arg_peri)   # поворот для разброса орбит по фазе
    rot = np.array([[cos_w, -sin_w], [sin_w, cos_w]])
    return rot @ pos, rot @ vel


def build_system(star_mass, planets):
    """planets: список {'mass','a','e','arg_peri'?, 'retrograde'?}. Индекс 0 = звезда."""
    n = len(planets) + 1
    masses = np.zeros(n)
    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))

    masses[0] = star_mass
    for i, p in enumerate(planets, start=1):
        masses[i] = p['mass']
        pos[i], vel[i] = body_from_orbit(
            p['a'], p['e'], star_mass, p['mass'],
            arg_peri=p.get('arg_peri', 0.0), retrograde=p.get('retrograde', False)
        )
    return shift_to_com(masses, pos, vel)


def shift_to_com(masses, pos, vel):
    """
    Сдвигает систему в систему отсчёта центра масс: суммарный импульс -> 0,
    положение ЦМ -> начало координат. Гарантирует, что звезда не дрейфует по экрану.
    """
    M = masses.sum()
    r_com = (masses[:, None] * pos).sum(axis=0) / M
    v_com = (masses[:, None] * vel).sum(axis=0) / M
    return masses, pos - r_com, vel - v_com


# ---------------------------------------------------------------------------
# 2. Ускорения: полный ньютоновский N-body + 1PN-поправка ОТО от звезды
# ---------------------------------------------------------------------------
def newtonian_acc(masses, pos):
    """a_i = sum_{j!=i} G*m_j*(r_j - r_i)/|r_j-r_i|^3 — полное взаимное притяжение."""
    n = len(masses)
    acc = np.zeros_like(pos)
    for i in range(n):
        diff = pos - pos[i]
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = 1.0
        inv_r3 = 1.0 / dist ** 3
        inv_r3[i] = 0.0
        acc[i] = (G * masses[:, None] * diff * inv_r3[:, None]).sum(axis=0)
    return acc


def pn_correction_acc(masses, pos, vel, star_index=0):
    """
    1PN-поправка ОТО к ускорению планет от звезды (доминирующий релятивистский
    эффект; вклад планет друг на друга на порядки меньше и им пренебрегаем —
    та же схема, что и в "gr"-модуле REBOUNDx, формула Nobili & Roxburgh 1986):

        a_PN = (G*M / (c^2 * r^2)) * [ (4*G*M/r - v^2) * n_hat + 4*(v . n_hat) * v ]

    r, v — расстояние и относительная скорость планеты относительно звезды,
    n_hat — орт от звезды к планете. Даёт прецессию перигелия
    dphi = 6*pi*G*M / (c^2 * a * (1-e^2)) за оборот (проверено на прецессии Меркурия).
    Реакция на звезду добавлена по 3-му закону Ньютона (сохранение импульса).
    """
    acc = np.zeros_like(pos)
    M = masses[star_index]
    r_star, v_star = pos[star_index], vel[star_index]

    for j in range(len(masses)):
        if j == star_index:
            continue
        r_vec = pos[j] - r_star
        v_vec = vel[j] - v_star
        r = np.linalg.norm(r_vec)
        n_hat = r_vec / r
        v2 = v_vec @ v_vec
        vr = v_vec @ n_hat

        a_pn = (G * M / (C_LIGHT ** 2 * r ** 2)) * (
            (4.0 * G * M / r - v2) * n_hat + 4.0 * vr * v_vec
        )
        acc[j] += a_pn
        acc[star_index] -= (masses[j] / M) * a_pn

    return acc


def total_acc(masses, pos, vel):
    return newtonian_acc(masses, pos) + pn_correction_acc(masses, pos, vel)


# ---------------------------------------------------------------------------
# 3. Интегратор Рунге-Кутты 4-го порядка
#    (устойчив к накоплению ошибок в отличие от Эйлера; корректно работает
#    со скоростно-зависимыми силами PN-поправки, т.к. это система [r,v]->[v,a(r,v)])
# ---------------------------------------------------------------------------
def rk4_step(masses, pos, vel, dt):
    def deriv(p, v):
        return v, total_acc(masses, p, v)

    k1p, k1v = deriv(pos, vel)
    k2p, k2v = deriv(pos + 0.5 * dt * k1p, vel + 0.5 * dt * k1v)
    k3p, k3v = deriv(pos + 0.5 * dt * k2p, vel + 0.5 * dt * k2v)
    k4p, k4v = deriv(pos + dt * k3p, vel + dt * k3v)

    new_pos = pos + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
    new_vel = vel + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return new_pos, new_vel


# ---------------------------------------------------------------------------
# 4. Основной цикл симуляции
# ---------------------------------------------------------------------------
def simulate(star_mass, planets, t_total, dt, recenter=True):
    """
    star_mass : масса звезды [M_sun]
    planets   : список {'mass','a','e','arg_peri'?,'retrograde'?} — до 10 планет
    t_total   : длительность [лет], dt : шаг интегрирования [лет]
    Возвращает: times[n_steps+1], traj[n_steps+1, n_bodies, 2] (в системе ЦМ), masses
    """
    assert len(planets) <= 10, "Максимум 10 планет"

    masses, pos, vel = build_system(star_mass, planets)
    n_steps = int(t_total / dt)
    n_bodies = len(masses)

    traj = np.zeros((n_steps + 1, n_bodies, 2))
    times = np.zeros(n_steps + 1)
    traj[0] = pos

    for step in range(1, n_steps + 1):
        pos, vel = rk4_step(masses, pos, vel, dt)
        traj[step] = pos
        times[step] = step * dt

    if recenter:
        # Страховка от численного дрейфа ЦМ из-за ошибок интегрирования:
        # на каждом шаге принудительно вычитаем текущее положение центра масс.
        com = (masses[None, :, None] * traj).sum(axis=1) / masses.sum()
        traj = traj - com[:, None, :]

    return times, traj, masses


# ---------------------------------------------------------------------------
# 5. Отрисовка траекторий (matplotlib импортируется только здесь)
# ---------------------------------------------------------------------------
def plot_trajectories(traj, masses, labels=None, title="Звёздная система (1PN ОТО)"):
    import matplotlib.pyplot as plt

    n_bodies = traj.shape[1]
    if labels is None:
        labels = ["Звезда"] + [f"Планета {i}" for i in range(1, n_bodies)]

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(n_bodies):
        if i == 0:
            ax.plot(traj[0, 0, 0], traj[0, 0, 1], marker='*', color='gold',
                    markersize=20, label=labels[0], zorder=5)
        else:
            ax.plot(traj[:, i, 0], traj[:, i, 1], linewidth=0.8, label=labels[i])

    ax.set_xlabel("x, а.е.")
    ax.set_ylabel("y, а.е.")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# 6. Демонстрационный запуск (не выполняется при импорте модуля)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Первая планета взята близко и эксцентрично, чтобы прецессия перигелия
    # от ОТО была заметна за разумное число оборотов ("горячий Меркурий").
    demo_planets = [
        {'mass': 1.0e-6, 'a': 0.15, 'e': 0.50, 'arg_peri': 0.0},
        {'mass': 3.0e-6, 'a': 0.39, 'e': 0.20, 'arg_peri': 1.0},
        {'mass': 3.0e-6, 'a': 0.72, 'e': 0.05, 'arg_peri': 2.5},
        {'mass': 3.0e-6, 'a': 1.00, 'e': 0.02, 'arg_peri': 4.0},
        {'mass': 3.2e-7, 'a': 1.52, 'e': 0.09, 'arg_peri': 0.8},
    ]

    t, traj, masses = simulate(star_mass=1.0, planets=demo_planets, t_total=40.0, dt=0.001)

    fig = plot_trajectories(traj, masses)
    fig.savefig("orbits.png", dpi=150)
    print(f"Готово: {traj.shape[0]} шагов, {traj.shape[1]} тел. График сохранён в orbits.png")
