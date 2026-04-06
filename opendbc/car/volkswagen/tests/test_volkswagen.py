import math
import random
import re
import unittest
from types import SimpleNamespace

from opendbc.car import DT_CTRL, structs
from opendbc.car.structs import CarParams
from opendbc.car.volkswagen.carcontroller import HCAMitigation, MQBStandstillManager
from opendbc.car.volkswagen.mqbcan import ESPOverride
from opendbc.car.volkswagen.values import CAR, CarControllerParams as CCP, FW_QUERY_CONFIG, WMI
from opendbc.car.volkswagen.fingerprints import FW_VERSIONS

Ecu = CarParams.Ecu

CHASSIS_CODE_PATTERN = re.compile('[A-Z0-9]{2}')
# TODO: determine the unknown groups
SPARE_PART_FW_PATTERN = re.compile(b'\xf1\x87(?P<gateway>[0-9][0-9A-Z]{2})(?P<unknown>[0-9][0-9A-Z][0-9])(?P<unknown2>[0-9A-Z]{2}[0-9])([A-Z0-9]| )')


class TestVolkswagenHCAMitigation(unittest.TestCase):
  STUCK_TORQUE_FRAMES = round(CCP.STEER_TIME_STUCK_TORQUE / (DT_CTRL * CCP.STEER_STEP))

  def test_same_torque_mitigation(self):
    """Same-torque nudge fires at the threshold, in the correct direction, and resets cleanly."""
    hca_mitigation = HCAMitigation(CCP)

    for actuator_value in (-CCP.STEER_MAX, -1, 0, 1, CCP.STEER_MAX):
      hca_mitigation.update(0, 0)  # Reset mitigation state
      for frame in range(self.STUCK_TORQUE_FRAMES + 2):
        should_nudge = actuator_value != 0 and frame == self.STUCK_TORQUE_FRAMES
        expected_torque = actuator_value - (1, -1)[actuator_value < 0] if should_nudge else actuator_value
        assert hca_mitigation.update(actuator_value, actuator_value) == expected_torque, f"{frame=}"

class TestVolkswagenMQBStandstillManager(unittest.TestCase):
  HOLD_MAX_FRAMES = MQBStandstillManager.HOLD_MAX_FRAMES
  HOLD_RELEASE_TOTAL_FRAMES = MQBStandstillManager.HOLD_RELEASE_TOTAL_FRAMES
  RELEASE_START_FRAME = HOLD_MAX_FRAMES - HOLD_RELEASE_TOTAL_FRAMES + 1  # first frame of release window
  START_INTENT_ACCEL_THRESHOLD = MQBStandstillManager.START_INTENT_ACCEL_THRESHOLD
  START_INTENT_MIN_FRAMES = MQBStandstillManager.START_INTENT_MIN_FRAMES
  START_COMMIT_ACCEL_MIN = MQBStandstillManager.START_COMMIT_ACCEL_MIN
  WEGIMPULSE_STILLNESS_FRAMES = MQBStandstillManager.WEGIMPULSE_STILLNESS_FRAMES

  def _cs(self, *, esp_hold_confirmation=False, esp_stopping=False, rolling_backward=False,
          rolling_forward=False, grade=0.0, brake_pressed=False, standstill=True, v_ego=0.0, sum_wegimpulse=0):
    out = SimpleNamespace(brakePressed=brake_pressed, standstill=standstill, vEgo=v_ego)
    return SimpleNamespace(out=out, esp_hold_confirmation=esp_hold_confirmation,
                           esp_stopping=esp_stopping, rolling_backward=rolling_backward,
                           rolling_forward=rolling_forward, grade=grade, sum_wegimpulse=sum_wegimpulse)

  def _prime_can_stop_forever(self, mgr, **cs_kwargs):
    """Run WEGIMPULSE_STILLNESS_FRAMES+1 frames with constant wegimpulse and esp_stopping to set can_stop_forever."""
    for _ in range(self.WEGIMPULSE_STILLNESS_FRAMES + 1):
      mgr.update(self._cs(esp_stopping=True, **cs_kwargs), long_active=True, accel=-1.0, stopping=True, starting=False)
    assert mgr.can_stop_forever

  def test_brake_pressed_disables_long_active(self):
    """Brake input overrides long_active to prevent faults when pre-enabled."""
    mgr = MQBStandstillManager()
    long_active, *_ = mgr.update(self._cs(brake_pressed=True), long_active=True, accel=0.0, stopping=True, starting=False)
    assert not long_active

  def test_hold_max_frames_disables_long_active(self):
    """long_active is suppressed after HOLD_MAX_FRAMES of confirmed hold to avoid a cruise fault."""
    mgr = MQBStandstillManager()
    cs = self._cs(esp_hold_confirmation=True)
    for _ in range(self.HOLD_MAX_FRAMES + 1):
      long_active, *_ = mgr.update(cs, long_active=True, accel=-1.0, stopping=True, starting=False)
    assert not long_active

  def test_can_stop_forever_flat_stop(self):
    """can_stop_forever is set when esp_stopping is active and wheels have been still for WEGIMPULSE_STILLNESS_FRAMES."""
    mgr = MQBStandstillManager()
    self._prime_can_stop_forever(mgr)
    *_, esp_override = mgr.update(self._cs(), long_active=True, accel=-1.0, stopping=True, starting=False)
    assert mgr.can_stop_forever
    assert esp_override == ESPOverride.START

  def test_can_stop_forever_cleared_by_hold_confirmation(self):
    """can_stop_forever is cleared when ESP confirms a hold (indefinite hold is no longer possible)."""
    mgr = MQBStandstillManager()
    self._prime_can_stop_forever(mgr)
    mgr.update(self._cs(esp_hold_confirmation=True), long_active=True, accel=-1.0, stopping=True, starting=False)
    assert not mgr.can_stop_forever

  def test_can_stop_forever_not_cleared_by_steep_grade(self):
    """Steep grades no longer disable can_stop_forever; rollback handling is managed separately."""
    mgr = MQBStandstillManager()
    self._prime_can_stop_forever(mgr)
    mgr.update(self._cs(grade=10), long_active=True, accel=-1.0, stopping=True, starting=False)
    assert mgr.can_stop_forever

  def test_can_stop_forever_persists_until_hold_confirmation(self):
    """can_stop_forever persists while long control stays active and no hold has been confirmed yet."""
    mgr = MQBStandstillManager()
    self._prime_can_stop_forever(mgr)
    *_, esp_override = mgr.update(self._cs(), long_active=True, accel=0.5, stopping=False, starting=False)
    assert mgr.can_stop_forever
    assert esp_override == ESPOverride.START

  def test_can_stop_forever_cleared_when_long_inactive(self):
    """can_stop_forever is cleared when long control is inactive."""
    mgr = MQBStandstillManager()
    self._prime_can_stop_forever(mgr)
    mgr.update(self._cs(), long_active=False, accel=-1.0, stopping=True, starting=False)
    assert not mgr.can_stop_forever

  def test_can_stop_forever_requires_wegimpulse_stillness(self):
    """can_stop_forever is not set if esp_stopping is active but wheels are still ticking."""
    mgr = MQBStandstillManager()
    for i in range(self.WEGIMPULSE_STILLNESS_FRAMES + 1):
      mgr.update(self._cs(esp_stopping=True, sum_wegimpulse=i), long_active=True, accel=-1.0, stopping=True, starting=False)
    assert not mgr.can_stop_forever

  def test_can_stop_forever_requires_esp_stopping(self):
    """can_stop_forever is not set by wheel stillness alone — esp_stopping must also be active."""
    mgr = MQBStandstillManager()
    for _ in range(self.WEGIMPULSE_STILLNESS_FRAMES + 1):
      mgr.update(self._cs(esp_stopping=False), long_active=True, accel=-1.0, stopping=True, starting=False)
    assert not mgr.can_stop_forever

  def test_stopping_override_fires_continuously_from_stillness_threshold(self):
    """STOP override fires on every frame once wheels have been still for >= WEGIMPULSE_STILLNESS_FRAMES while braking.
    Uses standstill=False to isolate the wegimpulse trigger from the cycling hold path."""
    mgr = MQBStandstillManager()
    cs = self._cs(standstill=False)  # keep cycling hold inactive so only wegimpulse trigger fires STOP
    for i in range(self.WEGIMPULSE_STILLNESS_FRAMES):
      *_, esp_override = mgr.update(cs, long_active=True, accel=-1.0, stopping=True, starting=False)
      assert esp_override != ESPOverride.STOP, f"STOP override fired early at frame {i}"
    for _ in range(3):  # fires on frame 10, 11, 12 — continuously, not just once
      *_, esp_override = mgr.update(cs, long_active=True, accel=-1.0, stopping=True, starting=False)
      assert esp_override == ESPOverride.STOP

  def test_stopping_override_suppressed_on_positive_accel(self):
    """STOP override does not fire when accel is positive, even if wheels have been still long enough."""
    mgr = MQBStandstillManager()
    cs = self._cs(standstill=False)
    for _ in range(self.WEGIMPULSE_STILLNESS_FRAMES + 1):
      *_, esp_override = mgr.update(cs, long_active=True, accel=0.5, stopping=False, starting=True)
    assert esp_override != ESPOverride.STOP

  def test_wegimpulse_change_resets_stillness_counter(self):
    """A wegimpulse change resets the stillness counter, delaying the STOP trigger."""
    mgr = MQBStandstillManager()
    for _ in range(self.WEGIMPULSE_STILLNESS_FRAMES - 1):
      mgr.update(self._cs(sum_wegimpulse=0), long_active=True, accel=-1.0, stopping=True, starting=False)
    # wheel ticks just before threshold — counter resets
    mgr.update(self._cs(sum_wegimpulse=1), long_active=True, accel=-1.0, stopping=True, starting=False)
    assert mgr.frames_since_wegimpulse_change == 0
    assert not mgr.can_stop_forever

  def test_can_stop_forever_cleared_on_wegimpulse_change(self):
    """can_stop_forever is immediately cleared when any wheel tick is detected."""
    mgr = MQBStandstillManager()
    self._prime_can_stop_forever(mgr)
    mgr.update(self._cs(sum_wegimpulse=1), long_active=True, accel=-1.0, stopping=True, starting=False)
    assert not mgr.can_stop_forever

  def test_can_stop_forever_persists_after_esp_stopping_clears(self):
    """can_stop_forever remains set even if esp_stopping is no longer active, until hold confirms."""
    mgr = MQBStandstillManager()
    self._prime_can_stop_forever(mgr)
    *_, esp_override = mgr.update(self._cs(esp_stopping=False), long_active=True, accel=0.5, stopping=False, starting=False)
    assert mgr.can_stop_forever
    assert esp_override == ESPOverride.START

  def test_low_speed_uphill_enters_stop_commit_before_launch(self):
    """Below the theoretical safe speed, uphill launch requests first commit to stopping to prevent rollback."""
    grade = 10.0
    mgr = MQBStandstillManager()
    _, accel, stopping, starting, esp_override = \
      mgr.update(self._cs(grade=grade), long_active=True, accel=0.5, stopping=False, starting=True)
    assert mgr.stop_commit_active
    assert accel == -3.5
    assert stopping is True
    assert starting is False
    assert esp_override == ESPOverride.STOP

  def test_launch_boost_not_applied_above_safe_speed(self):
    """Once speed exceeds the theoretical safe speed, uphill launch requests pass through unchanged."""
    mgr = MQBStandstillManager()
    _, accel, stopping, starting, esp_override = \
      mgr.update(self._cs(grade=10.0, v_ego=0.8, standstill=False), long_active=True, accel=0.5, stopping=False, starting=True)
    assert accel == 0.5
    assert stopping is False
    assert starting is True
    assert esp_override is None

  def test_rollback_brake_protection(self):
    """Accel is forced to -3.5 when rollback is active and accel is not positive."""
    mgr = MQBStandstillManager()
    mgr.update(self._cs(rolling_backward=True), long_active=True, accel=0.0, stopping=True, starting=False)
    assert mgr.rollback_detected
    _, accel, *_ = mgr.update(self._cs(), long_active=True, accel=-0.5, stopping=True, starting=False)
    assert accel == -3.5

  def test_rollback_protection_clears_on_rolling_forward(self):
    """rollback_detected is cleared when the car rolls forward."""
    mgr = MQBStandstillManager()
    mgr.update(self._cs(rolling_backward=True), long_active=True, accel=0.0, stopping=True, starting=False)
    assert mgr.rollback_detected
    mgr.update(self._cs(rolling_forward=True), long_active=True, accel=0.0, stopping=True, starting=False)
    assert not mgr.rollback_detected

  def test_hill_hold_first_frame_skip(self):
    """First frame with hold confirmed skips hill_accel to avoid a check engine light, but still sets overrides."""
    grade = 8.0
    cs = self._cs(esp_hold_confirmation=True, grade=grade, v_ego=1.0)
    mgr = MQBStandstillManager()
    _, accel, _, _, esp_override = \
      mgr.update(cs, long_active=True, accel=-1.0, stopping=True, starting=False)
    assert accel == -1.0  # hill_accel not yet applied
    assert esp_override == ESPOverride.STOP

  def test_hill_hold_accel_and_overrides(self):
    """Engine torque is built via hill_accel and ESP braking is held when stopped on a grade."""
    grade = 8.0
    cs = self._cs(esp_hold_confirmation=True, grade=grade, v_ego=1.0)
    mgr = MQBStandstillManager()
    for _ in range(2):
      mgr.update(cs, long_active=True, accel=-1.0, stopping=True, starting=False)
    _, accel, stopping, starting, esp_override = \
      mgr.update(cs, long_active=True, accel=-1.0, stopping=True, starting=False)
    assert accel == 0.045 * grade + 0.0625
    assert starting is True
    assert stopping is False
    assert esp_override == ESPOverride.STOP

  def test_hill_hold_accel_suppressed_on_shallow_grades(self):
    """hill_accel stays suppressed through 3% grade when the committed-stop path is not active."""
    for grade in (0.0, 1.0, 2.0, 3.0):
      cs = self._cs(esp_hold_confirmation=True, grade=grade, v_ego=1.0)
      mgr = MQBStandstillManager()
      for _ in range(2):
        mgr.update(cs, long_active=True, accel=-1.0, stopping=True, starting=False)
      _, accel, *_ = mgr.update(cs, long_active=True, accel=-1.0, stopping=True, starting=False)
      assert accel == -1.0, f"Expected no hill_accel boost at {grade=}"

  def test_hill_hold_progressive_release_pattern(self):
    """Progressive release pulses follow the 1-on/3-off, 2-on/3-off, 3-on/3-off, hold pattern near HOLD_MAX_FRAMES."""
    cs = self._cs(esp_hold_confirmation=True, grade=0.0)
    mgr = MQBStandstillManager()
    # Expected esp_override=START for each frame 1..HOLD_MAX_FRAMES
    # Frames before release window: always False
    # Release window phases: 0=T, 1-3=F, 4-5=T, 6-8=F, 9-11=T, 12-14=F, 15+=T
    excluded_phases = {1, 2, 3, 6, 7, 8, 12, 13, 14}
    for frame in range(1, self.HOLD_MAX_FRAMES + 1):
      *_, esp_override = mgr.update(cs, long_active=True, accel=-1.0, stopping=True, starting=False)
      phase = frame - self.RELEASE_START_FRAME
      expected = ESPOverride.START if phase >= 0 and phase not in excluded_phases else ESPOverride.STOP
      assert esp_override is expected, f"{frame=} {phase=}"

  def test_timer_resets_when_moving_without_hold(self):
    """Hold frame counter resets when wheels move without an ESP hold confirmation."""
    mgr = MQBStandstillManager()
    cs_held = self._cs(esp_hold_confirmation=True)
    for _ in range(5):
      mgr.update(cs_held, long_active=True, accel=-1.0, stopping=True, starting=False)
    assert mgr.esp_hold_frames == 5
    mgr.update(self._cs(v_ego=1.0, standstill=False), long_active=True, accel=0.5, stopping=False, starting=True)
    assert mgr.esp_hold_frames == 0

  def test_timer_resets_after_starting_with_hold(self):
    """Hold frame counter resets when hold drops after a starting attempt was sent."""
    mgr = MQBStandstillManager()
    cs_held = self._cs(esp_hold_confirmation=True, grade=0.0)
    # Run into release window so esp_override=START, setting hold_timer_can_reset
    for _ in range(self.RELEASE_START_FRAME):
      mgr.update(cs_held, long_active=True, accel=-1.0, stopping=True, starting=False)
    assert mgr.hold_timer_can_reset
    # Hold releases while car remains at standstill
    mgr.update(self._cs(esp_hold_confirmation=False), long_active=True, accel=-1.0, stopping=True, starting=False)
    assert mgr.esp_hold_frames == 1

  def test_hold_timer_can_reset_clears_on_inactive(self):
    """hold_timer_can_reset is cleared when long control disengages, preventing a stale reset on re-engagement."""
    mgr = MQBStandstillManager()
    cs_held = self._cs(esp_hold_confirmation=True, grade=0.0)
    for _ in range(self.RELEASE_START_FRAME):
      mgr.update(cs_held, long_active=True, accel=-1.0, stopping=True, starting=False)
    assert mgr.hold_timer_can_reset
    mgr.update(cs_held, long_active=False, accel=-1.0, stopping=True, starting=False)
    assert not mgr.hold_timer_can_reset

  def test_theoretical_safe_speed_zero_off_uphill(self):
    """The theoretical rollback-prevention threshold is only active on positive grades."""
    mgr = MQBStandstillManager(vehicle_mass=1540.0)
    assert mgr.get_theoretical_safe_speed(0.0, 0.0) == 0.0
    assert mgr.get_theoretical_safe_speed(-5.0, 0.0) == 0.0

  def test_theoretical_safe_speed_scales_with_grade(self):
    """The zero-rollback threshold grows quickly with grade for a typical MQB mass."""
    mgr = MQBStandstillManager(vehicle_mass=1540.0)
    expected_10 = (1.5 * (MQBStandstillManager.GRAVITY * 10.0 / math.sqrt(10.0**2 + 10000.0))**2 /
                   (MQBStandstillManager.BRAKE_TORQUE_RAMP_RATE / (1540.0 * MQBStandstillManager.ASSUMED_WHEEL_RADIUS)))
    assert abs(mgr.get_theoretical_safe_speed(10.0, 0.0) - expected_10) < 1e-9
    assert mgr.get_theoretical_safe_speed(12.0, 0.0) > mgr.get_theoretical_safe_speed(10.0, 0.0)

  def test_stop_commit_enters_below_safe_speed_on_uphill(self):
    """Below the theoretical safe speed on an uphill, negative accel commits to stopping with max brake."""
    mgr = MQBStandstillManager(vehicle_mass=1540.0)
    _, accel, stopping, starting, *_ = mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False),
                                                  long_active=True, accel=-0.1, stopping=False, starting=False)
    assert mgr.stop_commit_active
    assert not mgr.start_commit_active
    assert accel == -3.5
    assert stopping is True
    assert starting is False

  def test_stop_commit_flips_to_start_commit_on_positive_accel(self):
    """Sustained strong accel intent leaves committed stop and enters committed start."""
    mgr = MQBStandstillManager(vehicle_mass=1540.0)
    mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False), long_active=True, accel=-0.1, stopping=False, starting=False)
    for _ in range(self.START_INTENT_MIN_FRAMES - 1):
      mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False), long_active=True,
                 accel=self.START_INTENT_ACCEL_THRESHOLD + 0.1, stopping=True, starting=False)
    _, accel, stopping, starting, *_ = mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False),
                                                  long_active=True, accel=self.START_INTENT_ACCEL_THRESHOLD + 0.1,
                                                  stopping=True, starting=False)
    assert not mgr.stop_commit_active
    assert mgr.start_commit_active
    assert accel == 1.0
    assert stopping is False
    assert starting is True

  def test_start_commit_ignores_stop_intent_until_safe_speed(self):
    """Committed start persists below safe speed even if the requested accel changes its mind."""
    mgr = MQBStandstillManager(vehicle_mass=1540.0)
    mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False), long_active=True, accel=-0.1, stopping=False, starting=False)
    for _ in range(self.START_INTENT_MIN_FRAMES):
      mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False), long_active=True,
                 accel=self.START_INTENT_ACCEL_THRESHOLD + 0.1, stopping=True, starting=False)
    _, accel, stopping, starting, *_ = mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False),
                                                  long_active=True, accel=-0.1, stopping=True, starting=False)
    assert mgr.start_commit_active
    assert accel == 1.0
    assert stopping is False
    assert starting is True

  def test_start_commit_keeps_positive_accel_on_shallow_grade(self):
    """Committed start still enforces positive accel on shallow grades where hill accel alone is non-positive."""
    mgr = MQBStandstillManager(vehicle_mass=1540.0)
    for _ in range(self.START_INTENT_MIN_FRAMES):
      mgr.update(self._cs(grade=4.0, v_ego=0.01, standstill=False), long_active=True,
                 accel=self.START_INTENT_ACCEL_THRESHOLD + 0.1, stopping=False, starting=False)
    _, accel, stopping, starting, *_ = mgr.update(self._cs(grade=4.0, v_ego=0.01, standstill=False),
                                                  long_active=True, accel=-0.1, stopping=True, starting=False)
    assert mgr.start_commit_active
    assert accel == self.START_COMMIT_ACCEL_MIN
    assert stopping is False
    assert starting is True

  def test_weak_positive_accel_below_safe_speed_keeps_stop_commit(self):
    """Below the theoretical safe speed, weak positive accel is treated like stop intent to protect hold."""
    mgr = MQBStandstillManager(vehicle_mass=1540.0)
    _, accel, stopping, starting, *_ = mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False),
                                                  long_active=True, accel=self.START_INTENT_ACCEL_THRESHOLD - 0.05,
                                                  stopping=False, starting=False)
    assert mgr.stop_commit_active
    assert not mgr.start_commit_active
    assert accel == -3.5
    assert stopping is True
    assert starting is False

  def test_start_commit_enters_directly_below_safe_speed_on_strong_accel(self):
    """Below the theoretical safe speed on an uphill, sustained strong accel enters committed start directly."""
    mgr = MQBStandstillManager(vehicle_mass=1540.0)
    for _ in range(self.START_INTENT_MIN_FRAMES - 1):
      mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False), long_active=True,
                 accel=self.START_INTENT_ACCEL_THRESHOLD + 0.1, stopping=False, starting=False)
    _, accel, stopping, starting, *_ = mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False),
                                                  long_active=True, accel=self.START_INTENT_ACCEL_THRESHOLD + 0.1,
                                                  stopping=False, starting=False)
    assert not mgr.stop_commit_active
    assert mgr.start_commit_active
    assert accel == 1.0
    assert stopping is False
    assert starting is True

  def test_start_commit_clears_above_safe_speed(self):
    """Committed start clears once measured speed exceeds the theoretical threshold."""
    mgr = MQBStandstillManager(vehicle_mass=1540.0)
    mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False), long_active=True, accel=-0.1, stopping=False, starting=False)
    for _ in range(self.START_INTENT_MIN_FRAMES):
      mgr.update(self._cs(grade=10.0, v_ego=0.15, standstill=False), long_active=True,
                 accel=self.START_INTENT_ACCEL_THRESHOLD + 0.1, stopping=True, starting=False)
    _, accel, stopping, starting, *_ = mgr.update(self._cs(grade=10.0, v_ego=0.8, standstill=False),
                                                  long_active=True, accel=self.START_INTENT_ACCEL_THRESHOLD + 0.1,
                                                  stopping=False, starting=True)
    assert not mgr.start_commit_active
    assert accel == self.START_INTENT_ACCEL_THRESHOLD + 0.1
    assert stopping is False
    assert starting is True


class TestVolkswagenPlatformConfigs(unittest.TestCase):
  def test_mqb_gap_adjust_button_held_states(self):
    cp = structs.CarParams()
    cp.carFingerprint = next(iter(CAR))
    cp.transmissionType = CarParams.TransmissionType.automatic

    gap_button = next(b for b in CCP(cp).BUTTONS if b.event_type == structs.CarState.ButtonEvent.Type.gapAdjustCruise)
    assert gap_button.values == [1, 2, 3]

    state = False
    events = []
    for raw in (0, 1, 2, 3, 0):
      pressed = raw in gap_button.values
      if state != pressed:
        events.append((raw, pressed))
      state = pressed

    assert events == [(1, True), (0, False)]

  def test_spare_part_fw_pattern(self):
    # Relied on for determining if a FW is likely VW
    for platform, ecus in FW_VERSIONS.items():
      with self.subTest(platform=platform.value):
        for fws in ecus.values():
          for fw in fws:
            assert SPARE_PART_FW_PATTERN.match(fw) is not None, f"Bad FW: {fw}"

  def test_chassis_codes(self):
    for platform in CAR:
      with self.subTest(platform=platform.value):
        assert len(platform.config.wmis) > 0, "WMIs not set"
        assert len(platform.config.chassis_codes) > 0, "Chassis codes not set"
        assert all(CHASSIS_CODE_PATTERN.match(cc) for cc in
                   platform.config.chassis_codes), "Bad chassis codes"

        # No two platforms should share chassis codes
        for comp in CAR:
          if platform == comp:
            continue
          assert set() == platform.config.chassis_codes & comp.config.chassis_codes, \
                           f"Shared chassis codes: {comp}"

  def test_custom_fuzzy_fingerprinting(self):
    all_radar_fw = list({fw for ecus in FW_VERSIONS.values() for fw in ecus[Ecu.fwdRadar, 0x757, None]})

    for platform in CAR:
      with self.subTest(platform=platform.name):
        for wmi in WMI:
          for chassis_code in platform.config.chassis_codes | {"00"}:
            vin = ["0"] * 17
            vin[0:3] = wmi
            vin[6:8] = chassis_code
            vin = "".join(vin)

            # Check a few FW cases - expected, unexpected
            for radar_fw in random.sample(all_radar_fw, 5) + [b'\xf1\x875Q0907572G \xf1\x890571', b'\xf1\x877H9907572AA\xf1\x890396']:
              should_match = ((wmi in platform.config.wmis and chassis_code in platform.config.chassis_codes) and
                              radar_fw in all_radar_fw)

              live_fws = {(0x757, None): [radar_fw]}
              matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy(live_fws, vin, FW_VERSIONS)

              expected_matches = {platform} if should_match else set()
              assert expected_matches == matches, "Bad match"
