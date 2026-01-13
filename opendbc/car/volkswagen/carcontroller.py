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

# esp hold will fault after ~65 frames. we stop sending active at this threshold
# to give margin before the fault occurs
ESP_HOLD_FAULT_THRESHOLD = 50


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
    self.eps_timer_soft_disable_alert = False
    self.hca_frame_timer_running = 0
    self.hca_frame_same_torque = 0
    self.frames_at_standstill = 0
    self.esp_hold_frames = 0  # tracks ESP's internal hold timer
    self.esp_hold_prev = False
    self.reset_sent_while_engaged = False  # did we send a reset signal while engaged?
    self.remained_engaged_during_release = False  # did we stay engaged during hold release?
    self.steep_grade_hold_warning = False

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    can_sends = []

    # **** Steering Controls ************************************************ #

    if self.frame % self.CCP.STEER_STEP == 0:
      # Logic to avoid HCA state 4 "refused":
      #   * Don't steer unless HCA is in state 3 "ready" or 5 "active"
      #   * Don't steer at standstill
      #   * Don't send > 3.00 Newton-meters torque
      #   * Don't send the same torque for > 6 seconds
      #   * Don't send uninterrupted steering for > 360 seconds
      # MQB racks reset the uninterrupted steering timer after a single frame
      # of HCA disabled; this is done whenever output happens to be zero.

      if CC.latActive:
        new_torque = int(round(actuators.torque * self.CCP.STEER_MAX))
        apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.CCP)
        self.hca_frame_timer_running += self.CCP.STEER_STEP
        if self.apply_torque_last == apply_torque:
          self.hca_frame_same_torque += self.CCP.STEER_STEP
          if self.hca_frame_same_torque > self.CCP.STEER_TIME_STUCK_TORQUE / DT_CTRL:
            apply_torque -= (1, -1)[apply_torque < 0]
            self.hca_frame_same_torque = 0
        else:
          self.hca_frame_same_torque = 0
        hca_enabled = abs(apply_torque) > 0
      else:
        hca_enabled = False
        apply_torque = 0

      if not hca_enabled:
        self.hca_frame_timer_running = 0

      self.eps_timer_soft_disable_alert = self.hca_frame_timer_running > self.CCP.STEER_TIME_ALERT / DT_CTRL
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
        needs_cycle = self.CCS == mqbcan and CS.acc_type == 1
        force_disable = needs_cycle and CS.out.brakePressed
        long_active = False if force_disable else CC.longActive

        if (CS.esp_standstill_confirmation or CS.esp_hold_confirmation):
          self.frames_at_standstill += 1
        else:
          self.frames_at_standstill = 0

        # track ESP's internal hold timer. this timer:
        # - increments while esp_hold_confirmation is True
        # - pauses (holds value) while esp_hold_confirmation is False but car is stationary
        # - resets to 0 when:
        #   a) car actually moves while esp_hold_confirmation is False, OR
        #   b) a successful reset cycle: we sent reset while engaged, hold dropped, we remained engaged, hold reacquired
        # note: disengagements during hold release just pause the timer, they don't reset it.
        if CS.esp_hold_confirmation:
          # check if hold just reacquired after a valid reset cycle
          if not self.esp_hold_prev:
            if self.reset_sent_while_engaged and self.remained_engaged_during_release and long_active:
              # successful reset: we stayed engaged throughout the reset cycle
              self.esp_hold_frames = 0
            # clear reset tracking state
            self.reset_sent_while_engaged = False
          self.esp_hold_frames += 1
        else:
          # hold is released
          if not CS.esp_standstill_confirmation:
            # car is moving while hold is released - this resets the ESP's internal timer
            self.esp_hold_frames = 0
          # track if we remain engaged during the release period
          if not long_active:
            self.remained_engaged_during_release = False
        self.esp_hold_prev = CS.esp_hold_confirmation

        # if we're approaching the fault threshold without a successful reset, soft-disable
        # the car will start rolling, which resets our timer and allows re-braking
        steep_grade_soft_disable = needs_cycle and self.esp_hold_frames >= ESP_HOLD_FAULT_THRESHOLD
        if steep_grade_soft_disable:
          long_active = False
          self.steep_grade_hold_warning = True

        # clear warning when:
        # - user presses brake (taking over manually)
        # - openpilot wants to drive away (positive accel while engaged)
        if self.steep_grade_hold_warning and not steep_grade_soft_disable:
          if CS.out.brakePressed:
            self.steep_grade_hold_warning = False
          elif CC.longActive and actuators.accel > 0:
            self.steep_grade_hold_warning = False

        # expose warning state to carstate for event generation
        CS.steep_grade_hold_warning = self.steep_grade_hold_warning

        reset_signal = mqbcan.ResetSignal.NONE
        if (self.frames_at_standstill > 0 and needs_cycle and not steep_grade_soft_disable):
          if (self.frames_at_standstill > 10 and self.frames_at_standstill % 10 == 0):
            reset_signal = mqbcan.ResetSignal.CYCLE_AND_TORQUE
            # track that we sent a reset while engaged for valid reset detection
            if long_active:
              self.reset_sent_while_engaged = True
              self.remained_engaged_during_release = True
          else:
            reset_signal = mqbcan.ResetSignal.MAKE_OR_RELEASE_TORQUE

        acc_control = self.CCS.acc_control_value(CS.out.cruiseState.available, CS.out.accFaulted, long_active)
        accel = float(np.clip(actuators.accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if long_active else 0)

        stopping = actuators.longControlState == LongCtrlState.stopping
        starting = actuators.longControlState == LongCtrlState.pid and (CS.esp_hold_confirmation or CS.out.vEgo < self.CP.vEgoStopping)
        can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, self.CAN.pt, CS.acc_type, long_active, accel,
                                                           acc_control, stopping, starting, CS.esp_hold_confirmation, reset_signal))

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
                                                       lead_distance, hud_control.leadDistanceBars, self.steep_grade_hold_warning))

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
