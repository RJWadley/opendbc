import math
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.volkswagen import mlbcan, mqbcan, pqcan
from opendbc.car.volkswagen.values import CanBus, CarControllerParams, VolkswagenFlags

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState


class HCAMitigation:
  """
  Manages HCA fault mitigations for VW/Audi EPS racks:
    * Reduces torque by 1 for a single frame after commanding the same torque value for too long
  """

  def __init__(self, CCP):
    self._max_same_torque_frames = CCP.STEER_TIME_STUCK_TORQUE / (DT_CTRL * CCP.STEER_STEP)
    self._same_torque_frames = 0

  def update(self, apply_torque, apply_torque_last):
    if apply_torque != 0 and apply_torque_last == apply_torque:
      self._same_torque_frames += 1
      if self._same_torque_frames > self._max_same_torque_frames:
        apply_torque -= (1, -1)[apply_torque < 0]
        self._same_torque_frames = 0
    else:
      self._same_torque_frames = 0

    return apply_torque


class MQBStandstillManager:
  """
  Extended standstill for MQB w/ ACC type 1. There are three strategies.

  simple hold (best)
  Normally brake is commanded by the TSK. During a stopping procedure the ESP handles brake autonomously.
  If we exit the stopping procedure at the perfect moment, the ESP will hold indefinitely without complaining.

  cycling hold
  if our simple hold fails, use ACC_06 to build engine torque while simultaneously using ACC_07 to hold the brake.
  If the engine is producing enough torque to prevent rollback the ESP will happily cycle its timer when we ask it to.

  last resort
  if all else fails, disable long control and let the car creep naturally to avoid faulting cruise.
  """

  # simple hold
  BRAKE_TORQUE_RAMP_RATE = 2800.0     # Nm/s
  ASSUMED_WHEEL_RADIUS = 0.328        # m, typical MQB tire rolling radius
  PERMITTED_ROLLBACK_DISTANCE = 0.0   # m, kept at zero for now but could be relaxed
  GRAVITY = 9.81                      # m/s^2
  START_INTENT_ACCEL_THRESHOLD = 0.2  # m/s^2, accel must exceed this to roll on a hill
  START_INTENT_MIN_FRAMES = 5         # 100 ms at 50 Hz ACC update rate
  START_COMMIT_ACCEL_MIN = 0.2        # m/s^2, ensure committed launch still rolls forward
  WEGIMPULSE_STILLNESS_FRAMES = 10    # frames of no wheel tick change before triggering stop and allowing indefinite hold
  # cycling hold
  HOLD_RELEASE_TOTAL_FRAMES = 20      # total time allotted for progressive pulses during a cycling hold
  # last resort
  HOLD_MAX_FRAMES = 50                # frames to hold before disabling long control to avoid a fault

  def __init__(self, vehicle_mass: float = 1540.0):
    self.vehicle_mass = vehicle_mass
    self.esp_hold_frames = 0
    self.can_stop_forever = False
    self.rollback_detected = False
    self.hold_timer_can_reset = False
    self.stop_commit_active = False
    self.start_commit_active = False
    self.start_intent_frames = 0
    self.frames_since_wegimpulse_change = 0
    self._prev_sum_wegimpulse: int | None = None

  def get_theoretical_safe_speed(self, grade_pct: float, v_ego: float) -> float:
    # Because brake torque is based off a jerk-limited speed target even at standstill, the TSK may
    # not be able to build torque fast enough to prevent rollback when the car is moving slowly. If
    # the car is moving fast enough, we can rely on momentum to prevent rollback while the TSK is
    # building brake torque. Below this speed we lose our momentum buffer and risk rollback, so we
    # must force the car to stop prematurely. Higher grades require a higher minimum safe speed.
    if grade_pct <= 0 or self.vehicle_mass <= 0:
      return 0.0

    sin_theta = grade_pct / math.sqrt(grade_pct ** 2 + 10000.0)
    grade_accel = self.GRAVITY * sin_theta
    brake_decel_build_rate = self.BRAKE_TORQUE_RAMP_RATE / (self.vehicle_mass * self.ASSUMED_WHEEL_RADIUS)

    return 1.5 * grade_accel ** 2 / brake_decel_build_rate

  def update(self, CS, long_active: bool, accel: float, stopping: bool, starting: bool
             ) -> tuple[bool, float, bool, bool, "mqbcan.ESPOverride | None"]:
    esp_override: mqbcan.ESPOverride | None = None
    theoretical_safe_speed = self.get_theoretical_safe_speed(CS.grade, CS.out.vEgo)
    if CS.esp_hold_confirmation:
      self.esp_hold_frames += 1
    if CS.rolling_backward:
      self.rollback_detected = True
    elif CS.rolling_forward:
      self.rollback_detected = False

    # acc type 1 is sensitive to control signals when brake is pressed (when preEnabled)
    if CS.out.brakePressed:
      long_active = False

    # last resort: avoid a cruise fault if a hold is confirmed for too long and cannot be cycled
    if self.esp_hold_frames > self.HOLD_MAX_FRAMES:
      long_active = False

    # simple hold: detect strong start intent for use on hills
    strong_start_intent = False
    if long_active and theoretical_safe_speed > 0 and CS.out.vEgo < theoretical_safe_speed and accel > self.START_INTENT_ACCEL_THRESHOLD:
      self.start_intent_frames += 1
      strong_start_intent = self.start_intent_frames >= self.START_INTENT_MIN_FRAMES
    else:
      self.start_intent_frames = 0

    # simple hold: If we drop below our safe speed, we must force the car to stop. We remain stopped until the
    # vehicle has strong intent to drive away to prevent a scenario where we want to stop but cannot build
    # brake torque fast enough to prevent rollback.
    if not long_active or theoretical_safe_speed <= 0:
      self.stop_commit_active = False
      self.start_commit_active = False
    else:
      if self.start_commit_active:
        if CS.out.vEgo > theoretical_safe_speed:
          self.start_commit_active = False
      elif self.stop_commit_active:
        if strong_start_intent:
          self.stop_commit_active = False
          self.start_commit_active = True
      elif CS.out.vEgo < theoretical_safe_speed and strong_start_intent:
        self.start_commit_active = True
      elif CS.out.vEgo < theoretical_safe_speed and accel <= 0:
        self.stop_commit_active = True
      elif CS.out.vEgo < theoretical_safe_speed:
        self.stop_commit_active = True

    # simple hold: If needed, adjust acceleration to prevent rollback. In order of priority:
    # 1. if the car is actively rolling backward, crank brakes to max
    # 2. if we are committed to stopping due to low speed, crank brakes to max
    # 3. if we are committed to driving away on a hill, adjust accel to ensure we roll forward
    desired_launch_accel = 0.2 * CS.grade - 1
    if long_active and self.rollback_detected and accel <= 0:
      accel = -3.5
      stopping = True
      starting = False
    elif long_active and self.stop_commit_active:
      accel = -3.5
      stopping = True
      starting = False
    elif long_active and self.start_commit_active:
      accel = max(accel, desired_launch_accel, self.START_COMMIT_ACCEL_MIN)
      stopping = False
      starting = True

    # simple hold: track wheel stillness via wegimpulse counters every frame
    # any wheel movement immediately invalidates the hold
    if CS.sum_wegimpulse != self._prev_sum_wegimpulse:
      self.frames_since_wegimpulse_change = 0
      self.can_stop_forever = False
    else:
      self.frames_since_wegimpulse_change += 1
    self._prev_sum_wegimpulse = CS.sum_wegimpulse

    # simple hold: continuously assert stopping procedure while wheels are confirmed still and we're braking.
    # end the stopping procedure right after it starts, before any hold has been confirmed (hold only confirms at low speed).
    # if a hold is confirmed before we end the stopping procedure we won't be able to hold indefinitely.
    if long_active:
      if self.frames_since_wegimpulse_change >= self.WEGIMPULSE_STILLNESS_FRAMES and accel <= 0:
        esp_override = mqbcan.ESPOverride.STOP
      if CS.esp_stopping and self.frames_since_wegimpulse_change >= self.WEGIMPULSE_STILLNESS_FRAMES:
        self.can_stop_forever = True
      if self.esp_hold_frames > 0:
        self.can_stop_forever = False
      if self.can_stop_forever:
        esp_override = mqbcan.ESPOverride.START
    else:
      self.can_stop_forever = False


    # cycling hold: build engine torque via ACC_06 as rollback prevention if needed, ESP braking held via ACC_07
    # if long_active and accel <= 0 and not self.can_stop_forever and (CS.esp_hold_confirmation or CS.out.standstill):
    #   # skip torque management for one frame each cycle to avoid check engine light
    #   if self.esp_hold_frames > 1:
    #     # too much torque and the car moves, too little and the ESP won't cycle its timer
    #     # targets 80% of torque needed to hold the car at stop, derived from ESP_15 and some experimentation
    #     if CS.grade > 3:
    #       hill_accel = 0.045 * CS.grade + 0.0625
    #       accel = max(accel, hill_accel)
    #     starting = True
    #     stopping = False
    #   # Near the counter limit, send progressively longer starting pulses:
    #   # 1 frame, wait 3, 2 frames, wait 3, 3 frames, wait 3, then hold starting until cutoff.
    #   release_phase = self.esp_hold_frames - (self.HOLD_MAX_FRAMES - self.HOLD_RELEASE_TOTAL_FRAMES + 1)
    #   is_release_attempt = release_phase >= 0 and release_phase not in (1, 2, 3, 6, 7, 8, 12, 13, 14)
    #   esp_override = mqbcan.ESPOverride.START if is_release_attempt else mqbcan.ESPOverride.STOP

    # cycling hold: standstill timer resets under two conditions:
    # - wheels move while hold is not confirmed
    if not CS.out.standstill and not CS.esp_hold_confirmation:
      self.esp_hold_frames = 0
    # - we drop a hold confirmation after sending a start request
    esp_is_starting = long_active and (starting if esp_override is None else esp_override == mqbcan.ESPOverride.START)
    esp_is_stopping = long_active and (stopping if esp_override is None else esp_override == mqbcan.ESPOverride.STOP)
    esp_inactive = not esp_is_starting and not esp_is_stopping
    if esp_is_starting and CS.esp_hold_confirmation:
      self.hold_timer_can_reset = True
    if esp_inactive:
      self.hold_timer_can_reset = False
    if long_active and self.hold_timer_can_reset and not CS.esp_hold_confirmation:
      self.esp_hold_frames = 1 # don't switch hold strategies mid hold, that's jank

    return long_active, accel, stopping, starting, esp_override


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.CCP = CarControllerParams(CP)
    self.CAN = CanBus(CP)
    self.packer_pt = CANPacker(dbc_names[Bus.pt])
    self.aeb_available = not CP.flags & VolkswagenFlags.PQ

    if CP.flags & VolkswagenFlags.PQ:
      self.CCS = pqcan
    elif CP.flags & VolkswagenFlags.MLB:
      self.CCS = mlbcan
    else:
      self.CCS = mqbcan

    self.apply_torque_last = 0
    self.gra_acc_counter_last = None
    self.hca_mitigation = HCAMitigation(self.CCP)
    self.standstill_manager = MQBStandstillManager(CP.mass)

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    can_sends = []

    # **** Steering Controls ************************************************ #

    if self.frame % self.CCP.STEER_STEP == 0:
      apply_torque = 0
      if CC.latActive:
        new_torque = int(round(actuators.torque * self.CCP.STEER_MAX))
        apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.CCP)

      apply_torque = self.hca_mitigation.update(apply_torque, self.apply_torque_last)
      hca_enabled = apply_torque != 0
      self.apply_torque_last = apply_torque
      can_sends.append(self.CCS.create_steering_control(self.packer_pt, self.CAN.pt, apply_torque, hca_enabled))

      if self.CP.flags & VolkswagenFlags.STOCK_HCA_PRESENT:
        # Pacify VW Emergency Assist driver inactivity detection by changing its view of driver steering input torque
        # to the greatest of actual driver input or 2x openpilot's output (1x openpilot output is not enough to
        # consistently reset inactivity detection on straight level roads). See commaai/openpilot#23274 for background.
        ea_simulated_torque = float(np.clip(apply_torque * 2, -self.CCP.STEER_MAX, self.CCP.STEER_MAX))
        if abs(CS.out.steeringTorque) > abs(ea_simulated_torque):
          ea_simulated_torque = CS.out.steeringTorque
        can_sends.append(self.CCS.create_eps_update(self.packer_pt, self.CAN.cam, CS.eps_stock_values, ea_simulated_torque))

    # **** Acceleration Controls ******************************************** #

    if self.CP.openpilotLongitudinalControl:
      if self.frame % self.CCP.ACC_CONTROL_STEP == 0:
        long_active = CC.longActive
        accel = actuators.accel
        esp_override = None
        stopping = actuators.longControlState == LongCtrlState.stopping and CS.out.vEgo < self.CP.vEgoStopping
        starting = CS.out.vEgo < self.CCP.VW_LOW_SPEED_STATE_SPEED and not stopping

        if self.CCS == mqbcan and CS.acc_type == 1:
          long_active, accel, stopping, starting, esp_override = \
            self.standstill_manager.update(CS, long_active, accel, stopping, starting)

        acc_control = self.CCS.acc_control_value(CS.out.cruiseState.available, CS.out.accFaulted, long_active)
        accel = float(np.clip(accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if long_active else 0)

        can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, self.CAN.pt, CS.acc_type, long_active, accel,
                                                           acc_control, stopping, starting, CS.esp_hold_confirmation,
                                                           esp_override))

      #if self.aeb_available:
      #  if self.frame % self.CCP.AEB_CONTROL_STEP == 0:
      #    can_sends.append(self.CCS.create_aeb_control(self.packer_pt, False, False, 0.0))
      #  if self.frame % self.CCP.AEB_HUD_STEP == 0:
      #    can_sends.append(self.CCS.create_aeb_hud(self.packer_pt, False, False))

    # **** HUD Controls ***************************************************** #

    if self.frame % self.CCP.LDW_STEP == 0:
      hud_alert = 0
      if hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw):
        hud_alert = self.CCP.LDW_MESSAGES["laneAssistTakeOver"]
      can_sends.append(self.CCS.create_lka_hud_control(self.packer_pt, self.CAN.pt, CS.ldw_stock_values, CC.latActive,
                                                       CS.out.steeringPressed, hud_alert, hud_control))

    if self.frame % self.CCP.ACC_HUD_STEP == 0 and self.CP.openpilotLongitudinalControl:
      lead_distance = 0
      if hud_control.leadVisible and self.frame * DT_CTRL > 1.0:  # Don't display lead until we know the scaling factor
        lead_distance = 512 if CS.upscale_lead_car_signal else 8
      acc_hud_status = self.CCS.acc_hud_status_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive)
      # FIXME: PQ may need to use the on-the-wire mph/kmh toggle to fix rounding errors
      # FIXME: Detect clusters with vEgoCluster offsets and apply an identical vCruiseCluster offset
      set_speed = hud_control.setSpeed * CV.MS_TO_KPH
      can_sends.append(self.CCS.create_acc_hud_control(self.packer_pt, self.CAN.pt, acc_hud_status, set_speed,
                                                       lead_distance, hud_control.leadDistanceBars))

    # **** Stock ACC Button Controls **************************************** #

    gra_send_ready = self.CP.pcmCruise and CS.gra_stock_values["COUNTER"] != self.gra_acc_counter_last
    if gra_send_ready and (CC.cruiseControl.cancel or CC.cruiseControl.resume):
      can_sends.append(self.CCS.create_acc_buttons_control(self.packer_pt, self.CAN.ext, CS.gra_stock_values,
                                                           cancel=CC.cruiseControl.cancel, resume=CC.cruiseControl.resume))

    new_actuators = actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / self.CCP.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last

    self.gra_acc_counter_last = CS.gra_stock_values["COUNTER"]
    self.frame += 1
    return new_actuators, can_sends
