from enum import Enum
from dataclasses import dataclass
from opendbc.car.volkswagen.mqbcan import ResetSignal


class EspHoldState(Enum):
  DRIVING = "driving"           # car is moving
  STANDSTILL = "standstill"     # car stopped, hold not confirmed
  HOLD_ACTIVE = "hold_active"   # ESP hold confirmed, timer counting
  HOLD_FAILED = "hold_failed"   # timer exceeded threshold, must soft-disable


@dataclass
class EspHoldOutput:
  """Output from EspHoldController.update()"""
  long_active_override: bool | None  # None = no change, False = force disable
  reset_signal: ResetSignal
  steep_grade_warning: bool


class EspHoldController:
  """
  Manages ESP hold for VW MQB vehicles with acc_type == 1.

  The ESP hold will fault after ~65 frames of continuous hold. To prevent this,
  we periodically send reset signals to cycle the hold. This controller:

  1. Tracks time at standstill to send periodic reset signals
  2. Tracks ESP hold timer to detect impending faults
  3. Detects successful reset cycles to reset the hold timer
  4. Triggers soft-disable when the hold timer exceeds threshold (steep grade)

  State machine:
    DRIVING -> STANDSTILL: when car stops (esp_standstill or esp_hold confirmed)
    STANDSTILL -> HOLD_ACTIVE: when esp_hold confirmed
    HOLD_ACTIVE -> HOLD_FAILED: when hold timer >= threshold
    HOLD_ACTIVE -> DRIVING: when car moves (esp_standstill and esp_hold both false)
    STANDSTILL -> DRIVING: when car moves
    HOLD_FAILED -> DRIVING: when car rolls or user brakes
  """

  FAULT_THRESHOLD = 50    # frames before ESP hold faults (around 65 is the limit)
  RESET_START_DELAY = 10  # wait this many frames before first reset
  RESET_INTERVAL = 10     # send reset signal every N frames

  def __init__(self):
    self.state = EspHoldState.DRIVING
    self.frames_at_standstill = 0
    self.esp_hold_frames = 0

    # reset cycle detection: we need to track whether a reset cycle completed successfully
    # successful cycle = sent reset while engaged, hold dropped, remained engaged, hold reacquired
    self._esp_hold_prev = False
    self._reset_sent_while_engaged = False
    self._remained_engaged_during_release = False

    self._steep_grade_warning = False

  def update(
    self,
    esp_hold: bool,
    esp_standstill: bool,
    long_active: bool,
    brake_pressed: bool,
    accel: float,
  ) -> EspHoldOutput:
    """
    Update the ESP hold state machine.

    Args:
      esp_hold: ESP_Haltebestaetigung - ESP is actively holding the car
      esp_standstill: ESP_v_Signal == 0 - car is stationary per ESP
      long_active: openpilot longitudinal control is active
      brake_pressed: driver is pressing brake
      accel: commanded acceleration

    Returns:
      EspHoldOutput with override signals and warning state
    """
    # track frames at standstill for reset signal timing
    self._update_standstill_counter(esp_hold, esp_standstill)

    # update the ESP hold timer with complex pause/reset semantics
    self._update_hold_timer(esp_hold, esp_standstill, long_active)

    # determine current state
    new_state = self._compute_state(esp_hold, esp_standstill)

    # check for hold failure
    hold_failed = self.esp_hold_frames >= self.FAULT_THRESHOLD
    if hold_failed:
      new_state = EspHoldState.HOLD_FAILED
      self._steep_grade_warning = True

    self.state = new_state

    # determine outputs
    long_active_override = self._compute_long_active_override(hold_failed, brake_pressed)
    reset_signal = self._compute_reset_signal(long_active, hold_failed)
    self._update_warning_state(hold_failed, brake_pressed, long_active, accel)

    return EspHoldOutput(
      long_active_override=long_active_override,
      reset_signal=reset_signal,
      steep_grade_warning=self._steep_grade_warning,
    )

  def _update_standstill_counter(self, esp_hold: bool, esp_standstill: bool) -> None:
    """Update the frames_at_standstill counter."""
    if esp_standstill or esp_hold:
      self.frames_at_standstill += 1
    else:
      self.frames_at_standstill = 0

  def _update_hold_timer(self, esp_hold: bool, esp_standstill: bool, long_active: bool) -> None:
    """
    Update the ESP hold timer. This timer:
    - increments while esp_hold is True
    - pauses (holds value) while esp_hold is False but car is stationary
    - resets to 0 when:
      a) car actually moves while esp_hold is False, OR
      b) a successful reset cycle completes
    """
    if esp_hold:
      # check for rising edge (hold just acquired)
      if not self._esp_hold_prev:
        # check if this is a successful reset cycle
        if self._reset_sent_while_engaged and self._remained_engaged_during_release and long_active:
          # we stayed engaged throughout the cycle - timer resets
          self.esp_hold_frames = 0
        # always clear reset tracking on rising edge
        self._reset_sent_while_engaged = False
      self.esp_hold_frames += 1
    else:
      # hold is released
      if not esp_standstill:
        # car is actually moving - resets the ESP's internal timer
        self.esp_hold_frames = 0
      # track if we disengage during the release period
      if not long_active:
        self._remained_engaged_during_release = False

    self._esp_hold_prev = esp_hold

  def _compute_state(self, esp_hold: bool, esp_standstill: bool) -> EspHoldState:
    """Determine the current state based on inputs."""
    if not esp_standstill and not esp_hold:
      return EspHoldState.DRIVING
    elif esp_hold:
      return EspHoldState.HOLD_ACTIVE
    else:
      return EspHoldState.STANDSTILL

  def _compute_long_active_override(self, hold_failed: bool, brake_pressed: bool) -> bool | None:
    """Determine if we need to override long_active."""
    if brake_pressed:
      # user braking - force disable
      return False
    if hold_failed:
      # approaching fault threshold - force disable to let car roll
      return False
    return None

  def _compute_reset_signal(self, long_active: bool, hold_failed: bool) -> ResetSignal:
    """Determine what reset signal to send, if any."""
    if self.frames_at_standstill == 0:
      return ResetSignal.NONE
    if hold_failed:
      return ResetSignal.NONE

    # periodic reset signals while at standstill
    if self.frames_at_standstill > self.RESET_START_DELAY and self.frames_at_standstill % self.RESET_INTERVAL == 0:
      # full cycle + torque reset
      if long_active:
        # track that we sent a reset while engaged
        self._reset_sent_while_engaged = True
        self._remained_engaged_during_release = True
      return ResetSignal.CYCLE_AND_TORQUE
    else:
      # keep brake applied, torque engine to prevent rollback
      return ResetSignal.MAKE_OR_RELEASE_TORQUE

  def _update_warning_state(
    self,
    hold_failed: bool,
    brake_pressed: bool,
    long_active: bool,
    accel: float,
  ) -> None:
    """Update the steep grade warning state."""
    # warning clears when:
    # - user takes over with brake
    # - openpilot wants to drive away (positive accel while engaged)
    if self._steep_grade_warning and not hold_failed:
      if brake_pressed:
        self._steep_grade_warning = False
      elif long_active and accel > 0:
        self._steep_grade_warning = False
