"""
gestures.py — turn body keypoints into gesture events
=====================================================

WHY THIS FILE EXISTS:
    MoveNet tells us where 17 body points are in every frame.  This module
    watches the two wrists over time and decides when the person made one
    of four gestures:

        open_palm    hand held still at shoulder/face height ("stop sign")
        palm_up      hand pushed up above the head
        swipe_left   hand moved quickly to the user's left
        swipe_right  hand moved quickly to the user's right

    It is pure Python (no camera, no model, no MQTT) and receives the time
    as an argument, so every rule is unit-tested with synthetic poses.

HOW IT WORKS:
    1. Presence gate: only a person who is close and centred counts (both
       shoulders visible, shoulders wide enough, near the middle of the
       image).  Passers-by in the background are ignored.
    2. Body coordinates: each wrist is measured from the middle of the
       shoulders, in shoulder widths (SW), with y pointing up.  Walking
       across the image therefore does not move the wrists in body
       coordinates, so it cannot look like a swipe.
    3. Detectors: simple geometric rules over the last fraction of a second
       of wrist samples (thresholds live in config.yaml).
    4. Latch: after an event nothing else fires until the cooldown has
       passed; palm gestures also wait until that hand is lowered.  Holding
       a palm for 10 s therefore produces exactly one event.

LIMITATION:
    MoveNet has no finger keypoints, so it cannot tell an open palm from a
    fist.  "open_palm" means "a raised, still hand with the forearm up".
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple, Optional

# MoveNet keypoint indices (COCO order).  Each keypoint is (y, x, score)
# with x and y in 0..1 of the model's square input image.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

# (wrist, elbow) keypoint indices of each arm.
ARMS = {"left": (9, 7), "right": (10, 8)}

OPPOSITE_SWIPE = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}


@dataclass(frozen=True)
class GestureConfig:
    """Rule thresholds; positions in shoulder widths, times in seconds.

    Each field is documented in config.yaml (section "gestures").
    """
    min_keypoint_score: float = 0.3
    min_shoulder_width: float = 0.12
    max_center_offset: float = 0.35
    max_wrist_speed: float = 15.0
    max_gap_s: float = 0.3
    lost_s: float = 0.5
    cooldown_s: float = 1.0
    opposite_swipe_block_s: float = 1.5
    release_y: float = -1.0
    max_off_axis_ratio: float = 0.5
    palm_hold_s: float = 1.0
    palm_zone_min_y: float = -0.6
    palm_zone_max_y: float = 0.8
    palm_still_radius: float = 0.2
    palm_min_forearm_rise: float = 0.3
    up_min_rise: float = 1.2
    up_window_s: float = 0.7
    up_end_y: float = 1.0
    swipe_min_dx: float = 1.5
    swipe_window_s: float = 0.6
    swipe_band_min_y: float = -1.2
    swipe_band_max_y: float = 1.0


@dataclass(frozen=True)
class GestureEvent:
    gesture: str
    confidence: float   # mean keypoint score of the samples used, NOT a probability


class _Sample(NamedTuple):
    t: float
    x: float            # SW, positive = user's right (with mirror on)
    y: float            # SW, positive = up, 0 = shoulder line
    quality: float      # mean score of this wrist and the shoulders
    forearm_up: bool    # elbow visible and the wrist clearly above it
    frame: int          # number of the update() call that saw it


class GestureEngine:
    """Feed one frame of keypoints at a time; get at most one event back."""

    def __init__(self, config: GestureConfig, mirror: bool = True):
        self.config = config
        self.mirror = mirror
        # WHY a bit more than the longest rule window: the palm rule must
        # see that the hand has been tracked for the whole hold time.
        self._history_s = max(config.palm_hold_s, config.up_window_s, config.swipe_window_s) + 0.5
        self._frame = 0
        self.reset()

    def reset(self) -> None:
        """Forget everything (camera reconnect, person left)."""
        self._tracks = {side: deque() for side in ARMS}
        self._person_seen: Optional[float] = None
        self._center: Optional[tuple] = None
        self._latch: Optional[tuple] = None        # (gesture, side, t)
        self._last_swipe: Optional[tuple] = None   # (gesture, t)
        self._blocked_stroke = {side: 0 for side in ARMS}   # +1/-1: blocked stroke moving right/left

    @property
    def person_present(self) -> bool:
        return self._person_seen is not None

    def update(self, t: float, keypoints) -> Optional[GestureEvent]:
        """Process one frame taken at time t (seconds, monotonic)."""
        c = self.config
        self._frame += 1
        if self._person_seen is not None and t - self._person_seen >= c.lost_s:
            self.reset()

        kps = [tuple(float(v) for v in kp) for kp in keypoints]
        body = self._gate(kps)
        if body is None:
            return None
        cx, cy, sw = body
        # WHY: the shoulders jumping by more than a shoulder width between
        # frames means a different person (or a glitch); old history is useless.
        if self._center is not None and math.hypot(cx - self._center[0], cy - self._center[1]) > sw:
            self._clear_tracks()
        self._center = (cx, cy)
        self._person_seen = t

        shoulder_score = (kps[LEFT_SHOULDER][2] + kps[RIGHT_SHOULDER][2]) / 2
        fresh = [side for side in ARMS if self._add_sample(side, t, kps, cx, cy, sw, shoulder_score)]

        if self._latch is not None:
            if not self._released(t):
                return None
            self._latch = None
            self._trim_to_newest(*ARMS)   # motion made during the latch never counts

        best = None
        for side in fresh:
            found = self._detect(self._tracks[side])
            if found is None:
                continue
            gesture, window = found
            if self._blocked(gesture, window[0].t):
                # WHY: consume the natural return stroke after a swipe, now and
                # while it goes on (see _add_sample), so it cannot fire later.
                self._trim_to_newest(side)
                self._blocked_stroke[side] = 1 if gesture == "swipe_right" else -1
                continue
            confidence = round(sum(s.quality for s in window) / len(window), 2)
            if best is None or confidence > best[2]:
                best = (gesture, side, confidence)
        if best is None:
            return None

        gesture, side, confidence = best
        self._latch = (gesture, side, t)
        if gesture in OPPOSITE_SWIPE:
            self._last_swipe = (gesture, t)
        self._trim_to_newest(*ARMS)   # the same motion can never trigger twice
        return GestureEvent(gesture, confidence)

    # --- presence and coordinates ---------------------------------------------

    def _gate(self, kps):
        """Return (cx, cy, shoulder_width) of a close, centred person, else None."""
        c = self.config
        ly, lx, ls = kps[LEFT_SHOULDER]
        ry, rx, rs = kps[RIGHT_SHOULDER]
        if min(ls, rs) < c.min_keypoint_score:
            return None
        sw = math.hypot(lx - rx, ly - ry)
        cx, cy = (lx + rx) / 2, (ly + ry) / 2
        if sw < c.min_shoulder_width or abs(cx - 0.5) > c.max_center_offset:
            return None
        return cx, cy, sw

    def _add_sample(self, side, t, kps, cx, cy, sw, shoulder_score) -> bool:
        c = self.config
        wrist, elbow = ARMS[side]
        wy, wx, ws = kps[wrist]
        if ws < c.min_keypoint_score:
            return False
        x = (wx - cx) / sw
        if self.mirror:
            x = -x
        y = (cy - wy) / sw
        ey, _, es = kps[elbow]
        forearm_up = es >= c.min_keypoint_score and y - (cy - ey) / sw >= c.palm_min_forearm_rise

        track = self._tracks[side]
        if track:
            last = track[-1]
            dt = t - last.t
            # WHY: a movement must be seen to count.  Motion is measured only
            # across consecutive frames that saw this wrist: a wrist missed in
            # a frame (blur, low score, person not gated) or after a camera
            # hiccup may reappear elsewhere (e.g. MoveNet swapping left and
            # right), and a wrist that "teleports" between two frames is the
            # same glitch.  The last sighting stays in the track until then,
            # so a hand unseen for a moment does not count as lowered.
            if (last.frame != self._frame - 1 or dt >= c.max_gap_s
                    or math.hypot(x - last.x, y - last.y) > c.max_wrist_speed * max(dt, 1e-3)):
                track.clear()
            if dt >= c.lost_s:
                self._blocked_stroke[side] = 0   # a lost hand has ended its stroke (see below)
        track.append(_Sample(t, x, y, (ws + shoulder_score) / 2, forearm_up, self._frame))
        while track[0].t < t - self._history_s:
            track.popleft()
        direction = self._blocked_stroke[side]
        if direction and len(track) > 1:
            # WHY: a blocked return stroke stays blocked until the hand stops,
            # turns or is lost, even if it goes on past opposite_swipe_block_s.
            if (track[-1].x - track[-2].x) * direction > 0:
                self._trim_to_newest(side)
            else:
                self._blocked_stroke[side] = 0
        return True

    # --- latch ----------------------------------------------------------------

    def _released(self, t) -> bool:
        c = self.config
        gesture, side, t_event = self._latch
        if t - t_event < c.cooldown_s:
            return False
        if gesture in OPPOSITE_SWIPE:
            return True
        track = self._tracks[side]
        hand_up = bool(track) and t - track[-1].t < c.lost_s and track[-1].y >= c.release_y
        return not hand_up

    def _blocked(self, gesture, t_start) -> bool:
        if self._last_swipe is None:
            return False
        last_gesture, t_last = self._last_swipe
        # WHY the start time: a slow return stroke that began inside the block
        # time is still the return stroke, even if it ends after it.
        return (gesture == OPPOSITE_SWIPE[last_gesture]
                and t_start - t_last < self.config.opposite_swipe_block_s)

    def _trim_to_newest(self, *sides) -> None:
        for side in sides:
            track = self._tracks[side]
            while len(track) > 1:
                track.popleft()

    def _clear_tracks(self) -> None:
        for track in self._tracks.values():
            track.clear()

    # --- detectors --------------------------------------------------------------
    # Each returns (gesture, samples used) or None.  They are mutually
    # exclusive: a swipe is mostly horizontal, palm_up mostly vertical and
    # ends above the head, open_palm is still and below the head.

    def _detect(self, track):
        samples = list(track)
        return self._swipe(samples) or self._palm_up(samples) or self._open_palm(samples)

    def _swipe(self, samples):
        c = self.config
        newest = samples[-1]
        # WHY backwards from the newest sample: the first start point far
        # enough away is where the stroke began (the opposite-swipe block is
        # timed from there), and stopping at the first sample outside the band
        # keeps the WHOLE stroke between chest and head, so an arm swinging
        # through hip height is not a swipe.
        for i in range(len(samples) - 1, -1, -1):
            s = samples[i]
            if newest.t - s.t > c.swipe_window_s or not c.swipe_band_min_y <= s.y <= c.swipe_band_max_y:
                return None
            dx, dy = newest.x - s.x, newest.y - s.y
            if abs(dx) >= c.swipe_min_dx and abs(dy) <= c.max_off_axis_ratio * abs(dx):
                return ("swipe_right" if dx > 0 else "swipe_left"), samples[i:]
        return None

    def _palm_up(self, samples):
        c = self.config
        newest = samples[-1]
        if newest.y < c.up_end_y:
            return None
        for i, s in enumerate(samples):
            rise = newest.y - s.y
            if (newest.t - s.t <= c.up_window_s and rise >= c.up_min_rise
                    and abs(newest.x - s.x) <= c.max_off_axis_ratio * rise):
                return "palm_up", samples[i:]
        return None

    def _open_palm(self, samples):
        c = self.config
        newest = samples[-1]
        if newest.t - samples[0].t < c.palm_hold_s:
            return None
        window = [s for s in samples if s.t >= newest.t - c.palm_hold_s]
        if not all(s.forearm_up and c.palm_zone_min_y <= s.y <= c.palm_zone_max_y for s in window):
            return None
        mx = sum(s.x for s in window) / len(window)
        my = sum(s.y for s in window) / len(window)
        if any(math.hypot(s.x - mx, s.y - my) > c.palm_still_radius for s in window):
            return None
        return "open_palm", window
