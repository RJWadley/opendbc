import numpy as np
from enum import IntEnum
from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.volkswagen import mlbcan, mqbcan, pqcan
from opendbc.car.volkswagen.values import CanBus, CarControllerParams, VolkswagenFlags

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState

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
    self.braking_request_counter = 0
    self.gra_acc_counter_last = None
    self.esp_33_counter_last = None
    self.esp_05_counter_last = None
    self.eps_timer_soft_disable_alert = False
    self.distance_button_was_stopped = None
    self.hca_frame_timer_running = 0
    self.hca_frame_same_torque = 0

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
        esp_stopping_override = None
        esp_starting_override = None

        acc_control = self.CCS.acc_control_value(CS.out.cruiseState.available, CS.out.accFaulted, CC.longActive)
        accel = float(np.clip(actuators.accel, self.CCP.ACCEL_MIN, self.CCP.ACCEL_MAX) if CC.longActive else 0)
        stopping = actuators.longControlState == LongCtrlState.stopping if CC.longActive else False
        starting = actuators.longControlState == LongCtrlState.pid and (CS.esp_hold_confirmation or CS.out.vEgo < self.CP.vEgoStopping) if CC.longActive else False

        # distance button debug helper, force stop or start when distance button is pressed
        if CS.distance_button_pressed:
          if self.distance_button_was_stopped is None:
            self.distance_button_was_stopped = CS.esp_standstill_confirmation
          if CC.longActive:
            if self.distance_button_was_stopped:
              accel = 1
              stopping = False
              starting = CS.out.vEgo < self.CP.vEgoStopping if CC.longActive else False
            else:
              accel = min(-1.5, accel)
              stopping = CS.out.vEgo < self.CP.vEgoStopping if CC.longActive else False
              starting = False
        else:
          self.distance_button_was_stopped = None

        if CS.tsk_braking_request > 0:
          self.braking_request_counter += 1
        else:
          self.braking_request_counter = 0

        # for MQB type 1 acc, there are two timeouts we need to bypass.
        # the first (hold confirmation timeout) is around 60-70 frames of hold confirmation
        # the second (SRBM timeout) only applies on hills, and the timing varies between 1-3 seconds of SRBM active
        # note: the exact grade at which uphill logic applies may need tweaking
        if (CC.longActive and self.CCS == mqbcan and CS.acc_type == 1 and CS.esp_standstill_confirmation):

          # when stopped on a hill bypass second timer by restarting SRBM
          # if CS.tsk_grade > 2 and self.braking_request_counter >= 25:
          #   esp_stopping_override = True
          #   esp_starting_override = False

          # # bypass first timer by manually releasing the hold confirmation
          # # note that if we send more than one 'disabled' frame in a row, we'll lose brake pressure
          # elif CS.esp_hold_confirmation and CS.tsk_braking_request == 0 and self.frame % (5 * self.CCP.ACC_CONTROL_STEP) == 0:
          #   esp_stopping_override = False
          #   esp_starting_override = False
          # else:
          #   esp_stopping_override = False
          #   esp_starting_override = True

          # SRBM needs a change in accel to trigger a restart
          # if CS.esp_hold_confirmation:
          #   accel = -1.5
          # on hill, maximize braking to prevent accidental rollback during a hold
          if accel < 0:
            accel = self.CCP.ACCEL_MIN
          # on hill, prevent getting stuck during a takeoff attempt
          else:
            accel = max(accel, 1.5)

          if (CS.esp_hold_confirmation):
            esp_stopping_override = False
            esp_starting_override = False
          else:
            esp_stopping_override = False
            esp_starting_override = True

        can_sends.extend(self.CCS.create_acc_accel_control(self.packer_pt, self.CAN.pt, CS.acc_type, CC.longActive, accel,
                                                            acc_control, stopping, starting, CS.esp_hold_confirmation,
                                                            esp_stopping_override, esp_starting_override))

      #if self.aeb_available:
      #  if self.frame % self.CCP.AEB_CONTROL_STEP == 0:
      #    can_sends.append(self.CCS.create_aeb_control(self.packer_pt, False, False, 0.0))
      #  if self.frame % self.CCP.AEB_HUD_STEP == 0:
      #    can_sends.append(self.CCS.create_aeb_hud(self.packer_pt, False, False))

    # **** ESP_33 Counter Spoof ******************************************** #
    # spoof ESP_33 on ACAN (bus 1) so the ECU sees SRBM as available.
    # counter+1 supersedes the real message forwarded by the gateway from FCAN.
    # the ESP on FCAN never sees this because gateway doesn't forward ESP messages back.
    if self.CCS == mqbcan:
      if CS.esp_33_stock.get("COUNTER") != self.esp_33_counter_last:
        can_sends.append(self.CCS.create_esp_33_spoof(self.packer_pt, self.CAN.aux, CS.esp_33_stock))
      self.esp_33_counter_last = CS.esp_33_stock.get("COUNTER")

      if CS.esp_05_stock.get("COUNTER") != self.esp_05_counter_last:
        can_sends.append(self.CCS.create_esp_05_spoof(self.packer_pt, self.CAN.aux, CS.esp_05_stock))
      self.esp_05_counter_last = CS.esp_05_stock.get("COUNTER")

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
