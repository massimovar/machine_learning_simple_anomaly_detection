"""
Tests for gestures.py — synthetic keypoint trajectories, time injected.

Each test builds MoveNet-style (17, 3) arrays from hand positions given in
body coordinates (shoulder widths from the shoulder midpoint, y up, +x =
user's right) and feeds them at 20 FPS into one GestureEngine.
"""

import os
import random
import sys
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gestures import GestureConfig, GestureEngine  # noqa: E402

FPS = 20
DT = 1.0 / FPS

REST_R = (0.6, -1.6)       # right hand hanging by the side
REST_L = (-0.6, -1.6)
PALM = (0.4, 0.2)          # "stop sign" in front of the right shoulder
CHEST_L = (-0.5, -0.2)     # right hand across the body, chest height
CHEST_R = (1.2, -0.2)      # right hand out to the right, chest height
ABOVE_HEAD = (0.4, 1.4)


def make_pose(right=REST_R, left=REST_L, right_elbow=None, left_elbow=None,
              wrist_score=0.9, body_score=0.9, center=(0.5, 0.45), sw=0.25, mirror=True):
    """Body coordinates -> (17, 3) array of (y, x, score) in the model's square input."""
    cx, cy = center
    sign = -1.0 if mirror else 1.0   # a camera sees the user's right on the image left

    def put(kps, index, body_xy, score):
        bx, by = body_xy
        kps[index] = (cy - by * sw, cx + sign * bx * sw, score)

    kps = np.zeros((17, 3), dtype=np.float32)
    put(kps, 5, (-0.5, 0.0), body_score)   # left shoulder
    put(kps, 6, (0.5, 0.0), body_score)    # right shoulder
    # Default elbow: below the wrist, i.e. forearm raised.
    put(kps, 7, left_elbow or (left[0], left[1] - 0.8), body_score)
    put(kps, 8, right_elbow or (right[0], right[1] - 0.8), body_score)
    put(kps, 9, left, wrist_score)
    put(kps, 10, right, wrist_score)
    return kps


def move(start, end, seconds):
    """Positions moving linearly from start to end (start excluded)."""
    n = max(1, round(seconds * FPS))
    return [(start[0] + (end[0] - start[0]) * i / n,
             start[1] + (end[1] - start[1]) * i / n) for i in range(1, n + 1)]


def stay(position, seconds):
    return [position] * round(seconds * FPS)


def right_hand(positions, **pose_kwargs):
    return [make_pose(right=p, **pose_kwargs) for p in positions]


def raise_and_hold(seconds=1.2):
    return move(REST_R, PALM, 0.3) + stay(PALM, seconds)


def lower(start=PALM):
    return move(start, REST_R, 0.2) + stay(REST_R, 0.3)


def swipe_right_motion():
    return move(REST_R, CHEST_L, 0.4) + stay(CHEST_L, 0.3) + move(CHEST_L, CHEST_R, 0.4)


def swipe_left_motion():
    return move(REST_R, CHEST_R, 0.4) + stay(CHEST_R, 0.3) + move(CHEST_R, CHEST_L, 0.4)


class Session:
    """Feeds frames at FPS into one engine and records the events."""

    def __init__(self, config=None, mirror=True):
        self.engine = GestureEngine(config or GestureConfig(), mirror=mirror)
        self.t = 100.0
        self.events = []

    def play(self, poses, dt=DT):
        for kps in poses:
            self.t += dt
            event = self.engine.update(self.t, kps)
            if event is not None:
                self.events.append(event)
        return self

    def rest(self, seconds=0.5):
        return self.play(right_hand(stay(REST_R, seconds)))

    @property
    def names(self):
        return [e.gesture for e in self.events]


class TestRecognition(unittest.TestCase):

    def test_open_palm_fires_when_lowered_after_the_hold(self):
        s = Session().rest().play(right_hand(raise_and_hold()))
        self.assertEqual(s.names, [], "must not fire while the hand is still up")
        s.play(right_hand(lower()))
        self.assertEqual(s.names, ["open_palm"])

    def test_hold_shorter_than_palm_hold_s_is_not_open_palm(self):
        s = Session().rest().play(right_hand(raise_and_hold(0.8) + lower()))
        self.assertEqual(s.names, [])

    def test_hand_lowered_long_after_the_hold_is_not_open_palm(self):
        # The hold ends when the hand starts moving; reaching release_y
        # takes longer than palm_lower_s.
        s = Session().rest().play(right_hand(raise_and_hold() + move(PALM, REST_R, 1.5)))
        self.assertEqual(s.names, [])

    def test_open_palm_hold_survives_one_missed_frame(self):
        s = Session().rest().play(right_hand(raise_and_hold(0.6)))
        s.play(right_hand(stay(PALM, 0.05), wrist_score=0.1))
        s.play(right_hand(stay(PALM, 0.5) + lower()))
        self.assertEqual(s.names, ["open_palm"])

    def test_open_palm_hold_restarts_after_two_missed_frames(self):
        # The hand must stay visible for palm_hold_s.
        s = Session().rest().play(right_hand(raise_and_hold(0.6)))
        s.play(right_hand(stay(PALM, 0.1), wrist_score=0.1))
        s.play(right_hand(stay(PALM, 0.6) + lower()))
        self.assertEqual(s.names, [], "the hold must restart after the missed frames")
        s.play(right_hand(raise_and_hold() + lower()))
        self.assertEqual(s.names, ["open_palm"])

    def test_swipe_from_a_held_hand_is_only_a_swipe(self):
        # Raising the hand and pausing before a swipe looks exactly like a
        # stop sign; the swipe that follows must win.
        s = Session().rest().play(right_hand(move(REST_R, CHEST_L, 0.4) + stay(CHEST_L, 1.5)
                                             + move(CHEST_L, CHEST_R, 0.4) + lower(CHEST_R)))
        self.assertEqual(s.names, ["swipe_right"])

    def test_palm_up_from_a_held_hand_is_only_palm_up(self):
        s = Session().rest().play(right_hand(raise_and_hold(1.5) + move(PALM, ABOVE_HEAD, 0.4)
                                             + stay(ABOVE_HEAD, 0.5) + lower(ABOVE_HEAD)))
        self.assertEqual(s.names, ["palm_up"])

    def test_arm_held_out_after_a_swipe_is_not_open_palm(self):
        s = Session().rest().play(right_hand(swipe_right_motion() + stay(CHEST_R, 3.0) + lower(CHEST_R)))
        self.assertEqual(s.names, ["swipe_right"])

    def test_swipe_across_one_missed_frame_fires(self):
        # A single blurred frame during a fast move is normal on the device.
        stroke = right_hand(move(CHEST_L, CHEST_R, 0.4))
        stroke[3] = make_pose(right=CHEST_R, wrist_score=0.1)
        s = Session().rest().play(right_hand(move(REST_R, CHEST_L, 0.4) + stay(CHEST_L, 0.3)) + stroke)
        self.assertEqual(s.names, ["swipe_right"])

    def test_palm_up_across_one_missed_frame_fires(self):
        rise = right_hand(move(REST_R, ABOVE_HEAD, 0.5) + stay(ABOVE_HEAD, 0.3))
        rise[5] = make_pose(right=ABOVE_HEAD, wrist_score=0.1)
        s = Session().rest().play(rise)
        self.assertEqual(s.names, ["palm_up"])

    def test_fast_swipe_in_frames_arriving_in_a_burst_fires(self):
        # Frames 20 ms apart (camera burst): the wrist jumps 0.57 SW per
        # frame, about 28 SW/s, yet it is real motion.
        s = Session().rest().play(right_hand(move(REST_R, CHEST_L, 0.4) + stay(CHEST_L, 0.3)))
        s.play(right_hand(move(CHEST_L, CHEST_R, 0.15)), dt=0.02)
        self.assertEqual(s.names, ["swipe_right"])

    def test_palm_up(self):
        s = Session().rest()
        s.play(right_hand(move(REST_R, ABOVE_HEAD, 0.5) + stay(ABOVE_HEAD, 0.5)))
        self.assertEqual(s.names, ["palm_up"])

    def test_swipe_right(self):
        s = Session().rest().play(right_hand(swipe_right_motion()))
        self.assertEqual(s.names, ["swipe_right"])

    def test_swipe_left(self):
        s = Session().rest().play(right_hand(swipe_left_motion()))
        self.assertEqual(s.names, ["swipe_left"])

    def test_mirror_off_reverses_direction(self):
        poses = right_hand(swipe_right_motion())   # built as seen in a mirror
        s = Session(mirror=False).rest().play(poses)
        self.assertEqual(s.names, ["swipe_left"])

    def test_confidence_is_mean_keypoint_quality(self):
        s = Session().rest().play(right_hand(swipe_right_motion(), wrist_score=0.8, body_score=0.9))
        self.assertEqual(len(s.events), 1)
        self.assertEqual(s.events[0].confidence, 0.85)
        self.assertIsInstance(s.events[0].confidence, float)


class TestNoRepeatActivation(unittest.TestCase):

    def test_palm_held_ten_seconds_fires_once(self):
        s = Session().rest().play(right_hand(raise_and_hold(10.0) + lower()))
        self.assertEqual(s.names, ["open_palm"])

    def test_palm_held_through_a_short_dropout_fires_once(self):
        # A hand unseen for a moment is not a lowered hand.
        s = Session().rest().play(right_hand(raise_and_hold(2.0)))
        s.play(right_hand(stay(PALM, 0.15), wrist_score=0.1))
        s.play(right_hand(stay(PALM, 3.0) + lower()))
        self.assertEqual(s.names, ["open_palm"])

    def test_palm_moving_while_raised_does_not_fire_twice(self):
        s = Session().rest().play(right_hand(raise_and_hold(2.0)))
        s.play(right_hand(move(PALM, (0.0, 0.4), 0.3) + stay((0.0, 0.4), 3.0) + lower((0.0, 0.4))))
        self.assertEqual(s.names, ["open_palm"])

    def test_palm_refires_after_lowering(self):
        s = Session().rest().play(right_hand(raise_and_hold(2.0) + lower()))
        s.rest(1.0)                                   # past cooldown_s
        s.play(right_hand(raise_and_hold(1.5) + lower()))
        self.assertEqual(s.names, ["open_palm", "open_palm"])

    def test_raising_hand_to_hold_is_not_palm_up(self):
        s = Session().rest().play(right_hand(move(REST_R, PALM, 0.3) + stay(PALM, 0.5)))
        self.assertEqual(s.names, [])

    def test_palm_up_then_hold_above_head_is_not_open_palm(self):
        s = Session().rest().play(right_hand(move(REST_R, ABOVE_HEAD, 0.5) + stay(ABOVE_HEAD, 3.0)))
        self.assertEqual(s.names, ["palm_up"])

    def test_hand_raised_slowly_above_head_is_neither_gesture(self):
        s = Session().rest().play(right_hand(move(REST_R, ABOVE_HEAD, 2.0) + stay(ABOVE_HEAD, 2.0)))
        self.assertEqual(s.names, [])

    def test_swipe_with_fast_return_fires_once(self):
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(move(CHEST_R, CHEST_L, 0.4) + move(CHEST_L, REST_R, 0.4) + stay(REST_R, 2.0)))
        self.assertEqual(s.names, ["swipe_right"])

    def test_return_stroke_after_cooldown_is_blocked(self):
        far_left = (-0.8, -0.2)
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(stay(CHEST_R, 1.0) + move(CHEST_R, far_left, 0.3)
                          + move(far_left, REST_R, 0.4) + stay(REST_R, 1.0)))
        self.assertEqual(s.names, ["swipe_right"])

    def test_slow_return_starting_inside_the_block_is_blocked(self):
        # The return starts 1.2 s after the swipe but crosses swipe_min_dx
        # only at 1.6 s, after opposite_swipe_block_s has expired.
        far_left = (-0.8, -0.2)
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(stay(CHEST_R, 1.2) + move(CHEST_R, far_left, 0.5)
                          + move(far_left, REST_R, 0.4) + stay(REST_R, 1.0)))
        self.assertEqual(s.names, ["swipe_right"])

    def test_long_return_stroke_is_blocked_even_past_the_block_time(self):
        # The return starts 1.25 s after the swipe and goes on until 1.85 s:
        # the part after opposite_swipe_block_s is still the same stroke.
        far_left = (-1.9, -0.2)
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(stay(CHEST_R, 1.25) + move(CHEST_R, far_left, 0.6) + stay(far_left, 0.5)))
        self.assertEqual(s.names, ["swipe_right"])

    def test_swipe_after_the_hand_was_lost_is_not_the_return_stroke(self):
        # The blocked return stroke leaves the view still moving left; 1 s
        # later the hand comes back already moving left: a new stroke.
        far_left = (-1.9, -0.2)
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(stay(CHEST_R, 1.25) + move(CHEST_R, far_left, 0.6)))
        s.play(right_hand(stay(far_left, 1.0), wrist_score=0.1))
        s.play(right_hand([CHEST_R] + move(CHEST_R, CHEST_L, 0.3)))
        self.assertEqual(s.names, ["swipe_right", "swipe_left"])

    def test_second_swipe_within_cooldown_is_ignored(self):
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(move(CHEST_R, CHEST_L, 0.15) + move(CHEST_L, CHEST_R, 0.25)
                          + move(CHEST_R, REST_R, 0.4) + stay(REST_R, 1.0)))
        self.assertEqual(s.names, ["swipe_right"])

    def test_two_swipes_after_cooldown_fire_twice(self):
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(move(CHEST_R, REST_R, 0.4)))
        s.play(right_hand(swipe_right_motion()))
        self.assertEqual(s.names, ["swipe_right", "swipe_right"])

    def test_opposite_swipe_allowed_after_block(self):
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(move(CHEST_R, REST_R, 0.4) + stay(REST_R, 1.0)))
        s.play(right_hand(swipe_left_motion()))
        self.assertEqual(s.names, ["swipe_right", "swipe_left"])

    def test_opposite_swipe_after_holding_still_past_the_block_fires(self):
        # The stroke starts when the hand leaves the end point (1.6 s), not
        # at the oldest still sample in the swipe window (1.3 s).
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(stay(CHEST_R, 1.6) + move(CHEST_R, CHEST_L, 0.3)))
        self.assertEqual(s.names, ["swipe_right", "swipe_left"])

    def test_both_hands_swiping_together_fire_once(self):
        left_start, left_end = (-1.2, -0.2), (0.5, -0.2)
        poses = [make_pose(right=r, left=l) for r, l in zip(
            move(REST_R, CHEST_L, 0.4) + stay(CHEST_L, 0.3) + move(CHEST_L, CHEST_R, 0.4),
            move(REST_L, left_start, 0.4) + stay(left_start, 0.3) + move(left_start, left_end, 0.4))]
        s = Session().rest().play(poses)
        self.assertEqual(s.names, ["swipe_right"])


class TestFalsePositives(unittest.TestCase):

    def test_walking_across_the_image_is_not_a_swipe(self):
        low_forearm = {"right_elbow": (0.5, -0.5)}   # wrist barely above elbow: not a palm
        hand = (0.4, -0.4)
        poses = [make_pose(right=hand, center=(0.3 + 0.4 * i / 12, 0.45), **low_forearm)
                 for i in range(13)]
        poses += [make_pose(right=hand, center=(0.7, 0.45), **low_forearm)] * 40
        s = Session().play(poses)
        self.assertEqual(s.names, [])

    def test_hip_level_arm_swing_is_ignored(self):
        a, b = (-0.8, -1.4), (0.9, -1.4)
        s = Session().rest().play(right_hand(
            move(REST_R, a, 0.4) + move(a, b, 0.4) + move(b, a, 0.4) + move(a, b, 0.4)))
        self.assertEqual(s.names, [])

    def test_arm_swing_through_hip_height_is_not_a_swipe(self):
        # A pendulum swing: both ends inside the swipe band, the middle below it.
        a, bottom, b = (-0.9, -1.1), (0.0, -1.6), (0.9, -1.1)
        s = Session().rest().play(right_hand(move(REST_R, a, 0.4) + stay(a, 0.2)
                                             + move(a, bottom, 0.2) + move(bottom, b, 0.2)
                                             + move(b, REST_R, 0.4)))
        self.assertEqual(s.names, [])

    def test_low_wrist_scores_are_ignored(self):
        s = Session().rest().play(right_hand(swipe_right_motion(), wrist_score=0.2))
        self.assertEqual(s.names, [])

    def test_swipe_not_seen_between_two_sure_samples_is_ignored(self):
        # Wrist sure on the left, unsure for 0.4 s (e.g. motion blur), then
        # sure on the right: the movement itself was never observed.
        s = Session().rest().play(right_hand(move(REST_R, CHEST_L, 0.4) + stay(CHEST_L, 0.3)))
        s.play(right_hand(stay(CHEST_L, 0.4), wrist_score=0.1))
        s.play(right_hand(stay(CHEST_R, 0.5)))
        self.assertEqual(s.names, [])

    def test_swipe_across_a_few_missed_frames_is_ignored(self):
        # Three unsure frames (0.15 s): shorter than max_gap_s, and the jump
        # is small enough for max_wrist_jump, yet the movement was never seen
        # (it may be MoveNet swapping left and right).
        near_right = (0.9, -0.2)
        s = Session().rest().play(right_hand(move(REST_R, CHEST_L, 0.4) + stay(CHEST_L, 0.3)))
        s.play(right_hand(stay(CHEST_L, 0.15), wrist_score=0.1))
        s.play(right_hand(stay(near_right, 0.5)))
        self.assertEqual(s.names, [])

    def test_swipe_across_frames_without_a_gated_person_is_ignored(self):
        near_right = (0.9, -0.2)
        s = Session().rest().play(right_hand(move(REST_R, CHEST_L, 0.4) + stay(CHEST_L, 0.3)))
        s.play(right_hand(stay(CHEST_L, 0.15), body_score=0.1))   # shoulders unsure
        s.play(right_hand(stay(near_right, 0.5)))
        self.assertEqual(s.names, [])

    def test_rise_not_seen_between_two_sure_samples_is_not_palm_up(self):
        s = Session().rest()
        s.play(right_hand(stay(REST_R, 0.4), wrist_score=0.1))
        s.play(right_hand(stay(ABOVE_HEAD, 0.5)))
        self.assertEqual(s.names, [])

    def test_crossed_arms_are_not_open_palm(self):
        r_elbow, l_elbow = (0.6, -0.6), (-0.6, -0.6)
        r_end, l_end = (-0.5, -0.5), (0.5, -0.5)
        poses = [make_pose(right=r, left=l, right_elbow=r_elbow, left_elbow=l_elbow)
                 for r, l in zip(move(REST_R, r_end, 0.4) + stay(r_end, 3.0),
                                 move(REST_L, l_end, 0.4) + stay(l_end, 3.0))]
        s = Session().play(poses)
        self.assertEqual(s.names, [])

    def test_left_right_wrist_swap_glitch_is_not_a_swipe(self):
        r, l = (0.8, -0.8), (-0.8, -0.8)
        raise_hands = [make_pose(right=rp, left=lp)
                       for rp, lp in zip(move(REST_R, r, 0.4), move(REST_L, l, 0.4))]
        steady = [make_pose(right=r, left=l)] * 10
        swapped = [make_pose(right=l, left=r)] * 3
        s = Session().play(raise_hands + steady + swapped + steady * 2)
        self.assertEqual(s.names, [])

    def test_waving_hand_is_not_open_palm(self):
        a, b = (-0.2, 0.0), (0.6, 0.0)
        wave = (move(a, b, 0.5) + move(b, a, 0.5)) * 3
        s = Session().rest().play(right_hand(move(REST_R, a, 0.3) + wave))
        self.assertEqual(s.names, [])

    def test_jittery_palm_still_fires_once(self):
        rng = random.Random(42)
        jitter = [(PALM[0] + rng.uniform(-0.05, 0.05), PALM[1] + rng.uniform(-0.05, 0.05))
                  for _ in range(60)]
        s = Session().rest().play(right_hand(move(REST_R, PALM, 0.3) + jitter + lower()))
        self.assertEqual(s.names, ["open_palm"])


class TestPresence(unittest.TestCase):

    def test_far_person_is_gated(self):
        s = Session().play(right_hand(stay(REST_R, 0.5) + swipe_right_motion(), sw=0.05))
        self.assertEqual(s.names, [])
        self.assertFalse(s.engine.person_present)

    def test_off_center_person_is_gated(self):
        s = Session().play(right_hand(stay(REST_R, 0.5) + swipe_right_motion(), center=(0.9, 0.45)))
        self.assertEqual(s.names, [])

    def test_person_swap_restarts_the_hold(self):
        other = {"center": (0.8, 0.45)}                     # shoulders jump 1.2 SW
        s = Session().rest().play(right_hand(raise_and_hold(0.7)))
        s.play(right_hand(stay(PALM, 0.5) + lower(), **other))
        self.assertEqual(s.names, [], "a different person must start a new hold")
        s.play(right_hand(raise_and_hold() + lower(), **other))
        self.assertEqual(s.names, ["open_palm"])

    def test_person_lost_mid_gesture_restarts_the_hold(self):
        s = Session().rest().play(right_hand(raise_and_hold(0.6)))
        self.assertTrue(s.engine.person_present)
        s.play(right_hand(stay(PALM, 0.6), body_score=0.1))   # shoulders not visible
        self.assertFalse(s.engine.person_present)
        s.play(right_hand(stay(PALM, 0.6) + lower()))
        self.assertEqual(s.names, [], "the hold must restart after the person was lost")
        s.play(right_hand(raise_and_hold() + lower()))
        self.assertEqual(s.names, ["open_palm"])


class TestConfigFile(unittest.TestCase):

    def test_config_yaml_matches_code_defaults(self):
        path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        with open(path, encoding="utf-8") as f:
            section = yaml.safe_load(f)["gestures"]
        self.assertEqual(GestureConfig(**section), GestureConfig(),
                         "config.yaml gestures must equal GestureConfig defaults "
                         "so the unit tests cover the deployed thresholds")


if __name__ == "__main__":
    unittest.main()
